import os
import pickle
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf


def load_checkpoint(cfg):
    """Load the training state from the given checkpoint.

    Args:
        cfg: The training config.

    Returns:
        encoder_params: The parameters of the encoder loaded from the checkpoint, encoder_params is None if 'restoring_encoder' in cfg is False.
        decoder_params: The parameters of the decoder loaded from the checkpoint, decoder_params is None if 'restoring_decoder' in cfg is False.
        optimizer_state: The parameters of the optimizer loaded from checkpoint,optimizer_state is None if 'restoring_optimizer' in cfg is False.
        keys: The PRNG keys used for each device as the training key, keys is None if restoring_decoder in cfg is False.
        num_steps: The number of steps executed in the loaded training state, num_steps is 0 if restoring_decoder in cfg is false.
        extras: A dictionary that contains additional variables needed during training/validation, extra is an empty dictionary if 'restore_path' in cfg is False.
    """
    if cfg.checkpointing.restore_path:
        cfg.checkpointing.checkpoint_fname_load = os.path.splitext(
            cfg.checkpointing.checkpoint_fname_load
        )[0]
        with open(
            os.path.join(
                cfg.checkpointing.restore_path,
                cfg.checkpointing.checkpoint_fname_load + ".pkl",
            ),
            "rb",
        ) as f:
            saved_state = pickle.load(f)
            saved_encoder, saved_decoder = hk.data_structures.partition(
                lambda m, n, p: "encoder" in m, saved_state.params
            )

    encoder_params = None
    if cfg.checkpointing.restore_path and cfg.checkpointing.restore_encoder:
        encoder_params = saved_encoder

    decoder_params = None
    num_steps = 0
    keys = None
    if cfg.checkpointing.restore_path and cfg.checkpointing.restore_decoder:
        # If the number of decoders is smaller than the population size, clone the decoders till
        # the population size, else, copy the decoders until the current population size.
        old_population_size = jax.tree_util.tree_leaves(saved_decoder)[0].shape[0]
        if old_population_size == cfg.pop_size:
            decoder_params = saved_decoder
        elif (
            old_population_size == 1
            and cfg.checkpointing.allow_cloned_across_population
        ):
            decoder_params = jax.tree_map(
                lambda x: jnp.concatenate([x] * cfg.pop_size, axis=0)[: cfg.pop_size],
                saved_decoder,
            )
        elif (
            old_population_size == 1
            and not cfg.checkpointing.allow_cloned_across_population
        ):
            raise ValueError(
                "Decoder contains params for 1 agent, current population size is {}."
                "Either clone the decoder parameters for all agents by setting "
                "'allow_cloned_across_population' in config to True, or load training"
                " state with decoder params for {} agents.".format(
                    cfg.pop_size, cfg.pop_size
                )
            )
        else:
            raise ValueError(
                "Decoder contains params for {} agents, current population size is {}."
                "Load a training state with decoder params for {} agents.".format(
                    old_population_size, cfg.pop_size, cfg.pop_size
                )
            )

        # If we have the same number of devices as the number of saved keys, use the saved keys,
        # else, generate random keys.
        if saved_state.key.shape[0] == cfg.num_devices:
            keys = [jnp.array(k) for k in saved_state.key]

        num_steps = saved_state.num_steps

    optimizer_state = None
    if cfg.checkpointing.restore_path and cfg.checkpointing.restore_optimizer:
        optimizer_state = saved_state.optimizer_state

    extras = {"best_reward": 1e-6}
    if cfg.checkpointing.restore_path:
        if "extras" in saved_state.keys():
            if "best_reward" not in saved_state.extras.keys():
                saved_state.extras.update({"best_reward": 1e-6})
            extras = saved_state.extras

    return encoder_params, decoder_params, optimizer_state, keys, num_steps, extras


def create_checkpoint_directory(cfg, logger=None):
    """Create the directory to save the checkpoints.

    Args:
        cfg: The training config.
        logger: The PoppyLogger object used for logging the training information.
    """
    directory = cfg.checkpointing.directory
    Path(directory).mkdir(parents=True, exist_ok=True)

    if logger:
        with Path(directory + "/config.yaml").open("w+") as f:
            OmegaConf.save(cfg, f)
        logger.write_artifact({"config": Path(directory + "/config.yaml").as_posix()})


def save_checkpoint(cfg, training_state, logger=None, fname_prefix=""):
    """Save the checkpoint.

    Args:
        cfg: The training config.
        training_state: The container used to store training data.
        logger: The PoppyLogger object used for logging the training information.
        fname_prefix: String that is added as a prefix to the filename of the saved checkpoint.
    """
    directory = cfg.checkpointing.directory
    filename = (
        fname_prefix + os.path.splitext(cfg.checkpointing.checkpoint_fname_save)[0]
    )
    path_name = os.path.join(directory, filename)
    overwrite_checkpoints = cfg.checkpointing.overwrite_checkpoints

    if not overwrite_checkpoints and Path(path_name + ".pkl").is_file():
        raise ValueError(
            "Checkpoint already exists in {}, to overwrite existing checkpoint set "
            "'overwrite_checkpoints' in config to True.".format(
                cfg.checkpointing.directory
            )
        )

    with Path(path_name + "_tmp.pkl").open("wb") as f:
        pickle.dump(training_state, f)

    Path(path_name + ".pkl").unlink(missing_ok=True)

    if Path(path_name + "_tmp.pkl").is_file():
        Path(path_name + "_tmp.pkl").rename(path_name + ".pkl")

    if logger:
        logger.write_artifact({filename: Path(path_name + ".pkl").as_posix()})
