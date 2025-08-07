"""Multi-stage learning with RL"""

from __future__ import annotations
import os
import logging
from functools import partial
from typing import List, TYPE_CHECKING

import torch

from reinvent.utils import setup_logger, CsvFormatter, config_parse
from reinvent.runmodes import Handler, RL, create_adapter
from reinvent.runmodes.setup_sampler import setup_sampler
from reinvent.runmodes.RL import terminators, memories
from reinvent.runmodes.RL.data_classes import WorkPackage, ModelState
from reinvent.runmodes.utils import disable_gradients
from reinvent.scoring import Scorer, ScoreResults
from .validation import RLConfig

if TYPE_CHECKING:
    from reinvent.runmodes.RL import terminator_callable
    from reinvent.models import ModelAdapter
    from .validation import (
        SectionDiversityFilter,
        SectionLearningStrategy,
        SectionInception,
        SectionStage,
    )
    
from molscore import MolScore, MolScoreBenchmark, MolScoreCurriculum

logger = logging.getLogger(__name__)

TRANSFORMERS = ["Mol2Mol", "LinkinventTransformer", "LibinventTransformer", "Pepinvent"]


def setup_diversity_filter(config: SectionDiversityFilter, rdkit_smiles_flags: dict):
    """Setup of the diversity filter

    Basic setup of the diversity filter memory.  The parameters are from a
    dict, so the keys (parameters) are hard-coded here.

    :param config: config parameter specific to the filter
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: the set up diversity filter
    """

    if config is None or not hasattr(config, "type"):
        return None

    diversity_filter = getattr(memories, config.type)

    logger.info(f"Using diversity filter {config.type}")

    return diversity_filter(
        bucket_size=config.bucket_size,
        minscore=config.minscore,
        minsimilarity=config.minsimilarity,
        penalty_multiplier=config.penalty_multiplier,
        rdkit_smiles_flags=rdkit_smiles_flags,
    )


def setup_reward_strategy(config: SectionLearningStrategy, agent: ModelAdapter):
    """Setup the Reinforcement Learning reward strategy

    Basic parameter setup for RL learning including the reward function. The
    parameters are from a dict, so the keys (parameters) are hard-coded here.

    DAP has been found to be the best choice, see https://doi.org/10.1021/acs.jcim.1c00469.
    SDAP seems to have a smaller learning rate while the other two (MAULI, MASCOF)
    do not seem to bes useful at all.

    :param config: the configuration
    :param agent: the agent model network
    :return: the set up RL strategy
    """

    learning_rate = config.rate
    sigma = config.sigma  # determines how dominant the score is

    reward_strategy_str = config.type

    try:
        reward_strategy = getattr(RL, f"{reward_strategy_str}_strategy")
    except AttributeError:
        msg = f"Unknown reward strategy {reward_strategy_str}"
        logger.critical(msg)
        raise RuntimeError(msg)

    torch_optim = torch.optim.Adam(agent.get_network_parameters(), lr=learning_rate)
    learning_strategy = RL.RLReward(torch_optim, sigma, reward_strategy)

    logger.info(f"Using reward strategy {reward_strategy_str}")

    return learning_strategy


def setup_inception(config: SectionInception, prior: ModelAdapter):
    """Setup inception memory

    :param config: the config specific to the inception memory
    :param prior: the prior network
    :return: the set up inception memory or None
    """

    smilies = []
    deduplicate = config.deduplicate
    smilies_filename = config.smiles_file

    if smilies_filename and os.path.exists(smilies_filename):
        smilies = config_parse.read_smiles_csv_file(smilies_filename, columns=0)

        if not smilies:
            msg = f"Inception SMILES could not be read from {smilies_filename}"
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.info(f"Inception SMILES read from {smilies_filename}")

    if not smilies:
        logger.info(f"No SMILES for inception. Populating from first sampled batch.")

    if deduplicate:
        logger.info("Global SMILES deduplication for inception memory")

    inception = memories.Inception(
        memory_size=config.memory_size,
        sample_size=config.sample_size,
        smilies=smilies,
        scoring_function=None,
        prior=prior,
        deduplicate=deduplicate,
    )

    logger.info(f"Using inception memory")

    return inception


def create_packages(
    reward_strategy: RL.RLReward, stages: List[SectionStage], rdkit_smiles_flags: dict
) -> List[WorkPackage]:
    """Create work packages

    Collect the stage parameters and build a work package for each stage.  The
    parameters are from a dict, so the keys (parameters) are hard-coded here.
    Each stage can define its own scoring function.

    :param reward_strategy: the reward strategy
    :param stages: the parameters for each work package
    :param rdkit_smiles_flags: RDKit flags for canonicalization
    :return: a list of work packages
    """
    packages = []

    for stage in stages:
        chkpt_filename = stage.chkpt_file

        scoring_function = Scorer(stage.scoring)

        max_score = stage.max_score
        min_steps = stage.min_steps
        max_steps = stage.max_steps

        terminator_param = stage.termination
        terminator_name = terminator_param.lower().title()

        try:
            terminator: terminator_callable = getattr(terminators, f"{terminator_name}Terminator")
        except KeyError:
            msg = f"Unknown termination criterion: {terminator_name}"
            logger.critical(msg)
            raise RuntimeError(msg)

        diversity_filter = None

        if stage.diversity_filter:
            diversity_filter = setup_diversity_filter(stage.diversity_filter, rdkit_smiles_flags)

        packages.append(
            WorkPackage(
                scoring_function,
                reward_strategy,
                max_steps,
                terminator(max_score, min_steps),
                diversity_filter,
                chkpt_filename,
            )
        )

    return packages


def run_molscore_learning(
    input_config: dict,
    device: torch.device,
    tb_logdir: str,
    responder_config: dict,
    write_config: str = None,
    molscore_config: str = None,
    *args,
    **kwargs,
):
    """Run Reinforcement Learning/Curriculum Learning with MolScore

    :param input_config: the run configuration
    :param device: torch device
    :param tb_logdir: TensorBoard log directory
    :param responder_config: responder configuration
    :param write_config: callable to write config
    :param molscore_config: MolScore configuration
    """
    
    # Read MolScore configuration
    if molscore_config is not None:
        molscore_cfg = config_parse.read_config(molscore_config, molscore_config.rsplit(".")[-1])
        _ = input_config.pop("molscore_config", None)  # Remove from input config
    else:
        assert "molscore_config" in input_config, "MolScore configuration is required"
        molscore_cfg = input_config.pop("molscore_config")
    
    # Read RL config
    config = RLConfig(**input_config)
    parameters = config.parameters

    # NOTE: The model files are a dictionary with model attributes from
    #       Reinvent and a set of tensors, each with an attribute for the
    #       device (CPU or GPU) and if gradients are required

    prior_model_filename = parameters.prior_file
    agent_model_filename = parameters.agent_file

    # NOTE: Inference mode means here that torch runs eval() on the network:
    #       switch off some specific layers (dropout, batch normal) but that
    #       does not affect autograd
    #       The gradients are switched off for the prior but the optimizer will
    #       not touch those anyway because we pass only the agent network to the
    #       optimizer, see above.
    adapter, _, model_type = create_adapter(prior_model_filename, "inference", device)
    prior = adapter
    disable_gradients(prior)

    rdkit_smiles_flags = dict(allowTautomers=True)

    if model_type in TRANSFORMERS:  # Transformer-based models
        agent_mode = "inference"
        rdkit_smiles_flags.update(sanitize=True, isomericSmiles=True)
        rdkit_smiles_flags2 = dict(isomericSmiles=True)
    else:
        agent_mode = "training"
        rdkit_smiles_flags2 = dict()

    adapter, agent_save_dict, agent_model_type = create_adapter(
        agent_model_filename, agent_mode, device
    )
    agent = adapter

    if model_type != agent_model_type:
        msg = f"Inconsistent model types: prior is {model_type} agent is {agent_model_type}"
        logger.critical(msg)
        raise RuntimeError(msg)

    logger.info(f"Using generator {model_type}")
    logger.info(f"Prior read from {prior_model_filename}")
    logger.info(f"Agent read from {agent_model_filename}")

    smilies = None

    if parameters.smiles_file:
        smilies = config_parse.read_smiles_csv_file(parameters.smiles_file, columns=0)
        logger.info(f"Input molecules/fragments read from file {parameters.smiles_file}")

    sampler, _ = setup_sampler(model_type, parameters.dict(), agent)
    reward_strategy = setup_reward_strategy(config.learning_strategy, agent)

    # NOTE we don't use the divesity filter from here, instead use MolScore

    if parameters.purge_memories:
        logger.info("Purging diversity filter memories after each stage")
    else:
        logger.info("Diversity filter memories are retained between stages")

    inception = None

    # Inception only set up for the very first step
    if config.inception and model_type == "Reinvent":
        inception = setup_inception(config.inception, prior)

    if not inception and model_type == "Reinvent":
        logger.warning("Inception disabled but may speed up convergence")

    #packages = create_packages(reward_strategy, stages, rdkit_smiles_flags2)

    # NOTE MolScore saves output by default
    
    # FIXME: is there a sensible default, this is only needed by Mol2Mol
    distance_threshold = parameters.distance_threshold

    model_learning = getattr(RL, f"{model_type}Learning")

    if callable(write_config):
        write_config(config.model_dump())
        
    # Setup MolScore
    def wrap_scoring_function(scoring_function, smiles, *args, **kwargs):
        """Wrap the MolScore scoring function as Reinvent4 expects"""
        scores = scoring_function(smiles)
        return ScoreResults(
            smilies=smiles,
            total_scores=scores,
            completed_components=[],
        )
    def wrap_terminator(scoring_function, *args, **kwargs):
        """Wrap MolScore.finished as Reinvent4 expects terminator"""
        return scoring_function.finished
    
    # Single mode
    if molscore_cfg['molscore_mode'] == "single":
        task = MolScore(
            model_name=molscore_cfg.get("model_name", "Reinvent4"),
            task_config=molscore_cfg['molscore_task'],
            budget=molscore_cfg['total_smiles'],
            output_dir=molscore_cfg['output_dir'],
            add_run_dir=True,
            **molscore_cfg.get("molscore_kwargs", {}),
        )
        with task as scoring_function:
                # Setup optimizer
                optimize = model_learning(
                    max_steps=int(1e9), # Set very high to rely on terminator.
                    stage_no=0,
                    prior=prior,
                    state=ModelState(agent, None), # No diversity filter, handled by MolScore
                    scoring_function=partial(wrap_scoring_function, scoring_function),
                    reward_strategy=reward_strategy,
                    sampling_model=sampler,
                    smilies=smilies,
                    distance_threshold=distance_threshold,
                    rdkit_smiles_flags=rdkit_smiles_flags,
                    inception=inception,
                    responder_config=responder_config,
                    tb_logdir=None,
                )
                # Optimize
                _ = optimize(partial(wrap_terminator, scoring_function))
                
    # Benchmark mode
    if molscore_cfg['molscore_mode'] == "benchmark":
        MSB = MolScoreBenchmark(
            model_name=molscore_cfg.get("model_name", "Reinvent4"),
            benchmark=molscore_cfg['molscore_task'],
            budget=molscore_cfg['total_smiles'],
            output_dir=molscore_cfg['output_dir'],
            add_run_dir=True,
            **molscore_cfg.get("molscore_kwargs", {}),
        )
        with MSB as benchmark:
            for task in benchmark:
                with task as scoring_function:
                        # Setup optimizer
                        optimize = model_learning(
                            max_steps=int(1e9), # Set very high to rely on terminator.
                            stage_no=0,
                            prior=prior,
                            state=ModelState(agent, None), # No diversity filter, handled by MolScore
                            scoring_function=partial(wrap_scoring_function, scoring_function),
                            reward_strategy=reward_strategy,
                            sampling_model=sampler,
                            smilies=smilies,
                            distance_threshold=distance_threshold,
                            rdkit_smiles_flags=rdkit_smiles_flags,
                            inception=inception,
                            responder_config=responder_config,
                            tb_logdir=None,
                        )
                        # Optimize
                        _ = optimize(partial(wrap_terminator, scoring_function))
                        
    # Curriculum mode
    if molscore_cfg['molscore_mode'] == "curriculum":
        MSB = MolScoreCurriculum(
            model_name=molscore_cfg.get("model_name", "Reinvent4"),
            benchmark=molscore_cfg['molscore_task'],
            budget=molscore_cfg['total_smiles'],
            output_dir=molscore_cfg['output_dir'],
            **molscore_cfg.get("molscore_kwargs", {}),
        )
        with task as scoring_function:
                # Setup optimizer
                optimize = model_learning(
                    max_steps=int(1e9), # Set very high to rely on terminator.
                    stage_no=0,
                    prior=prior,
                    state=ModelState(agent, None), # No diversity filter, handled by MolScore
                    scoring_function=partial(wrap_scoring_function, scoring_function),
                    reward_strategy=reward_strategy,
                    sampling_model=sampler,
                    smilies=smilies,
                    distance_threshold=distance_threshold,
                    rdkit_smiles_flags=rdkit_smiles_flags,
                    inception=inception,
                    responder_config=responder_config,
                    tb_logdir=None,
                )
                # Optimize
                _ = optimize(partial(wrap_terminator, scoring_function))
