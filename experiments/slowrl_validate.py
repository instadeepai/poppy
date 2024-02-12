import hydra
import jax
import omegaconf

from poppy.trainers.slowrl_validation import slowrl_validate
from poppy.utils.logger import EnsembleLogger, NeptuneLogger, TerminalLogger


def create_logger(cfg) -> EnsembleLogger:
    loggers = []

    if "terminal" in cfg.logger:
        loggers.append(TerminalLogger(**cfg.logger.terminal))

    if "neptune" in cfg.logger:
        neptune_config = {}
        neptune_config["name"] = cfg.logger.neptune.name
        neptune_config["project"] = cfg.logger.neptune.project
        neptune_config["tags"] = [
            f"{cfg.algo_name}",
            "slowrl",
            f"{cfg.env_name}",
            # "final-exp",
        ]
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
    slowrl_cfg = cfg.slowrl

    available_devices = len(jax.local_devices())
    if slowrl_cfg.num_devices < 0:
        slowrl_cfg.num_devices = available_devices
        print(f"Using {available_devices} available device(s).")
    else:
        assert (
            available_devices >= slowrl_cfg.num_devices
        ), f"{slowrl_cfg.num_devices} devices requested but only {available_devices} available."

    # create a logger
    cfg.logger.neptune.name = "slowrl-" + cfg.logger.neptune.name
    logger = create_logger(cfg)

    # init the random key
    key = jax.random.PRNGKey(slowrl_cfg.problem_seed)
    metrics = slowrl_validate(
        random_key=key, cfg=slowrl_cfg, params=None, logger=logger
    )

    print(metrics)


if __name__ == "__main__":
    run()
