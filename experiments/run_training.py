import argparse
import dataclasses
import jax
from typing import Dict
import yaml

from poppy.trainers import TrainingConfig
from poppy.utils.logger import TerminalLogger
from poppy.utils.config_utils import make_env_trainer, EnvironmentConfig


def read_config_file(filename: str) -> Dict:
    with open(filename, 'r') as f:
        return yaml.load(f, yaml.Loader)


def make_env_config(config: Dict) -> EnvironmentConfig:
    return EnvironmentConfig(
        name=config["environment"]["name"],
        params=config["environment"]["params"]
    )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a YAML file containing the configuration parameters")
    args = parser.parse_args()

    # Determine the number of devices
    n = len(jax.local_devices())
    print(f"Running on {n} devices")

    # Initialize the logger
    logger = TerminalLogger(label="", time_delta=10)

    # Read the configuration
    config = read_config_file(args.config)

    # Parse the environment configuration
    env_config = make_env_config(config)
    logger.write_config(dataclasses.asdict(env_config))
    environment, trainer_class = make_env_trainer(env_config)

    training_config = TrainingConfig(
        learning_rate_encoder=1e-4 if config["train_encoder"] else 0.0,
        learning_rate_decoder=1e-4,
        batch_size=config["num_problems_training"] // n,  # batch size *per device* (repeated for each element of the population)
        minibatch_train=config["minibatch_size_training"] // n,
        pop_size=config["pop_size"],
        pomo_size=config["num_starting_points"],
        train_best=config["train_best"],
        l2_regularization=1e-6,
        seed=0,
        validation_freq=config["validation_freq"],
        num_validation_problems=config["num_problems_validation"] // n,  # num_validation_problems *per device*
        minibatch_validation=config["minibatch_size_validation"] // n,
        num_devices=-1,  # -1 --> auto-detect
        save_checkpoint=config["save_checkpoint"],
        save_best=config["save_best"],
        load_checkpoint=config["load_checkpoint"],  # '' --> no checkpoint
        load_decoder=config["load_decoder"],
        load_optimizer=config["load_optimizer"],
        compute_expensive_metrics=False,
        use_augmentation_validation=True,
        save_matrix_freq=-1,
    )

    trainer = trainer_class(
        environment=environment,
        config=training_config,
        logger=logger,
    )
    trainer.train(num_steps=config["num_steps"])
