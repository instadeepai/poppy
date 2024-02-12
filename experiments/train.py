import hydra
import jax
import omegaconf

from poppy.trainers.trainer import Trainer
from poppy.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [f"{cfg.algo_name}", "training", f"{cfg.env_name}"]
        neptune_config["parameters"] = cfg

        loggers.append(NeptuneLogger(**neptune_config))

    # return the loggers
    return EnsembleLogger(loggers)


@hydra.main(
    config_path="config",
    version_base=None,
    config_name="config_exp",
)
def run(cfg: omegaconf.DictConfig) -> None:

    # create the name of the run's directory - used for logging and checkpoints
    run_subdirectory = (
        str(cfg.env_name)
        + "/"
        + str(cfg.algo_name)
        + "/"
        + f"bs{cfg.batch_size}_ps{cfg.pop_size}"
        + f"_ga{cfg.optimizer.num_gradient_accumulation_steps}"
        + f"_seed{cfg.seed}/"
    )

    # update base name with complete name
    cfg.checkpointing.directory = cfg.checkpointing.directory + run_subdirectory
    cfg.validation.checkpointing.restore_path = (
        cfg.validation.checkpointing.restore_path + run_subdirectory
    )

    # Check and configure the available devices.
    available_devices = len(jax.local_devices())
    if cfg.num_devices < 0:
        cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= cfg.num_devices
        ), f"{cfg.num_devices} devices requested but only {available_devices} available."

    # Create the logger and save the config.
    logger = create_logger(cfg)
    # logger.write_config(cfg)

    # Train!
    trainer = Trainer(cfg, logger)
    trainer.train()

    # Tidy.
    logger.close()


if __name__ == "__main__":
    run()
