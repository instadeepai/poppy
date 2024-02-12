import hydra
import jax
import omegaconf

from poppy.trainers.validation import validate
from poppy.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [f"{cfg.algo_name}", "fastrl", f"{cfg.env_name}"]
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

    # Check and configure the available devices.
    validation_cfg = cfg.validation

    available_devices = len(jax.local_devices())
    if validation_cfg.num_devices < 0:
        validation_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= validation_cfg.num_devices
        ), f"{validation_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    cfg.logger.neptune.name = "fastrl-" + cfg.logger.neptune.name
    logger = create_logger(cfg)

    # init the random key
    metrics = validate(
        cfg=validation_cfg,
    )

    metrics = {f"validate/{k}": v for (k, v) in metrics.items()}
    logger.write(metrics)


if __name__ == "__main__":
    run()
