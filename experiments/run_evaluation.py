import argparse
import pickle as pkl
import time

import jax
import jax.numpy as jnp
import numpy as np

from poppy.trainers import TrainingConfig
from poppy.utils.config_utils import EnvironmentConfig, make_env_trainer
from poppy.utils.utils import (
    fetch_from_devices,
    reduce_from_devices,
    spread_over_devices,
)


def create_trainer(
    num_devices: int,
    env_config: EnvironmentConfig,
    pop_size: int,
    model_path: str,
    num_problems: int,
):
    config = TrainingConfig(
        pop_size=pop_size,
        seed=0,
        num_validation_problems=num_problems
        // num_devices,  # num_validation_problems *per device*
        minibatch_validation=1000 // (num_devices * 25),
        num_devices=-1,  # -1 --> auto-detect
        save_checkpoint=False,
        load_checkpoint=model_path,
        load_decoder=True,
        use_augmentation_validation=False,
    )

    environment, trainer_class = make_env_trainer(env_config)
    trainer = trainer_class(
        environment=environment,
        config=config,
        logger=None,
    )
    trainer.init()
    return trainer


def get_scores(trainer, problems):
    t = time.time()
    metrics, logging = trainer.run_validation_epoch(trainer.training_state, problems)
    jax.tree_map(lambda x: x.block_until_ready(), metrics)  # For accurate timings.
    metrics["total_time"] = time.time() - t
    if trainer.config.num_devices > 1:
        metrics = reduce_from_devices(metrics, axis=0)
        logging = fetch_from_devices(logging, as_numpy=True)
        logging["score_matrix"] = np.concatenate(logging["score_matrix"], axis=-1)
        return metrics, logging
    return metrics, logging


def get_scores_augmentation(num_devices, trainer, problems):
    if num_devices > 1:
        problems_aug = jax.pmap(jax.vmap(trainer.get_augmentations))(problems)  # devices, batch, augmentations, N, 2
        problems_aug = jnp.swapaxes(
            problems_aug, 0, 2
        )  # augmentations, batch, devices, N, 2
        problems_aug = jnp.swapaxes(
            problems_aug, 1, 2
        )  # augmentations, devices, batch, N, 2
    else:
        problems_aug = jax.vmap(trainer.get_augmentations)(problems)
        problems_aug = jnp.swapaxes(problems_aug, 0, 1)  # augmentations, batch, N, 2

    loggings = []
    for problems in problems_aug:
        metrics, logging = get_scores(trainer, problems)
        loggings.append(logging["score_matrix"])

    # stack(loggings): (augmentations, population, n_instances)
    return np.stack(loggings).max(0)


def load_cvrp_test_data(num_nodes: int):
    if num_nodes == 100:
        dataset_filename = "experiments/evaluation_data/vrp100_test_seed1234.pkl"
    elif num_nodes == 125:
        dataset_filename = "experiments/evaluation_data/vrp125_test_small_seed1235.pkl"
    elif num_nodes == 150:
        dataset_filename = "experiments/evaluation_data/vrp150_test_small_seed1235.pkl"
    else:
        raise RuntimeError(f"Error: There is no {num_nodes}-node CVRP dataset.")

    with open(dataset_filename, "rb") as f:
        dataset = pkl.load(f)

    coordinates = [[instance[0]] + instance[1] for instance in dataset]
    demands = np.expand_dims([[0.0] + instance[2] for instance in dataset], axis=2)
    data = np.concatenate([coordinates, demands], axis=-1)
    return np.stack(data)


def load_knapsack_test_data(num_items: int):
    if num_items == 100:
        dataset_filename = "experiments/evaluation_data/knapsack_100.pkl"
    elif num_items == 200:
        dataset_filename = "experiments/evaluation_data/knapsack_200.pkl"
    elif num_items == 250:
        dataset_filename = "experiments/evaluation_data/knapsack_250.pkl"
    else:
        raise RuntimeError(f"Error: There is no {num_items}-item TSP dataset.")

    with open(dataset_filename, "rb") as f:
        dataset = pkl.load(f)
    return dataset


def load_tsp_test_data(num_cities: int):
    if num_cities == 100:
        dataset_filename = "experiments/evaluation_data/tsp100_test_seed1234.pkl"
    elif num_cities == 125:
        dataset_filename = "experiments/evaluation_data/tsp125_test_small_seed1235.pkl"
    elif num_cities == 150:
        dataset_filename = "experiments/evaluation_data/tsp150_test_small_seed1235.pkl"
    elif num_cities == 200:
        dataset_filename = "experiments/evaluation_data/tsp200_test_small_seed1235.pkl"
    else:
        raise RuntimeError(f"Error: There is no {num_cities}-item Knapsack dataset.")

    with open(dataset_filename, "rb") as f:
        dataset = pkl.load(f)
    return np.stack(dataset)


def load_test_data(env_config: EnvironmentConfig):
    if env_config.name == "cvrp":
        return load_cvrp_test_data(env_config.params["num_nodes"])
    elif env_config.name == "knapsack":
        return load_knapsack_test_data(env_config.params["num_items"])
    elif env_config.name == "tsp":
        return load_tsp_test_data(env_config.params["num_cities"])
    raise RuntimeError(f"Error: Unknown environment name '{env_config.name}'.")


def make_env_config(env_name: str, problem_size: int) -> EnvironmentConfig:
    if env_name == "cvrp":
        if problem_size == 100:
            norm_factor = 50
        elif problem_size == 125:
            norm_factor = 55
        elif problem_size == 150:
            norm_factor = 60
        else:
            raise RuntimeError(
                f"Error: A problem size of {problem_size} is not supported for CVRP."
            )
        params = {"num_nodes": problem_size, "norm_factor": norm_factor}
    elif env_name == "knapsack":
        params = {"num_items": problem_size, "total_budget": 25}
    elif env_name == "tsp":
        params = {"num_cities": problem_size}
    else:
        raise RuntimeError(f"Error: Unknown environment name '{env_name}'.")
    return EnvironmentConfig(name=env_name, params=params)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="name of the environment (cvrp, knapsack, tsp)")
    parser.add_argument(
        "problem_size",
        type=int,
        help="size of the problem (number of nodes in CVRP, number of items in Knapsack, number of cities in TSP)",
    )
    parser.add_argument(
        "pop_size", type=int, help="size of the population used in the model"
    )
    parser.add_argument(
        "model_path", help="path to the folder containing the model parameters"
    )
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = get_arg_parser().parse_args()

    # Determine the number of devices
    n = len(jax.local_devices())
    print(f"Running on {n} devices")

    # Parse the environment configuration
    env_config = make_env_config(args.env, args.problem_size)
    environment, trainer_class = make_env_trainer(env_config)

    # Fetch the test data
    test_data = load_test_data(env_config)

    # Create the trainer
    trainer = create_trainer(
        num_devices=n,
        env_config=env_config,
        pop_size=args.pop_size,
        model_path=args.model_path,
        num_problems=test_data.shape[0],
    )

    # Compute the score and print it
    if n > 1:
        test_data = spread_over_devices(test_data)
    score_matrix = get_scores_augmentation(n, trainer, test_data)
    print(f"Evaluation Score: {score_matrix.max(0).mean()}")
