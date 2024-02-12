import time
import jax
import jax.numpy as jnp
import numpy as np
from run_evaluation import create_trainer, load_test_data, make_env_config, get_arg_parser
from poppy.utils.config_utils import make_env_trainer
from poppy.sampler import Sampler


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
    num_problems = test_data.shape[0]

    # Create the trainer
    trainer = create_trainer(
        num_devices=n,
        env_config=env_config,
        pop_size=args.pop_size,
        model_path=args.model_path,
        num_problems=num_problems,
    )

    # Use augmentations
    problems_aug = jax.vmap(trainer.get_augmentations)(test_data)
    problems_aug = jnp.swapaxes(
        problems_aug, 0, 1
    )  # augmentations, batch, N, problem_dim

    # Evaluate the model
    t0 = time.time()
    rewards_aug = []
    for problems in problems_aug:
        sampler = Sampler(trainer, problems)
        rewards = sampler.run_sampling()
        rewards = rewards.reshape(rewards.shape[0], -1)

        rewards_scores = sampler.scores.max(1).max(-1)
        rewards_full = jnp.stack([rewards, rewards_scores]).max(0)
        rewards_full = np.array(rewards_full)

        rewards_aug.append(rewards_full)

    rewards_aug = np.stack(rewards_aug).max(0)
    t1 = time.time()
    assert rewards_aug.size == num_problems

    # Print the results
    print(f"Mean reward: {np.mean(rewards_aug)}")
    print(f"Time: {t1 - t0}")
