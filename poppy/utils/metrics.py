import jax
import jax.numpy as jnp
from jax import random


def compute_expensive_metrics(episode_reward):
    """
    Compute expensive diversity metrics.
    """
    batch_size = episode_reward.shape[1]
    n_agents = episode_reward.shape[0]
    score_matrix = episode_reward.max(-1)
    metrics = {}

    def next_agents_score(agents):
        """
        :param agents: one hot binary vector of length pop_size. Ones for agents in the subpopulation,
         zeros for others.
        :return: a new one hot binary vector of length pop_size with the best next agent added, and the
         performance of the subpopulation on the validation problems.
        """

        selected_agents = agents.tile((batch_size, 1)).transpose()
        best_perf = (score_matrix + jnp.where(selected_agents == 0, -1000, 0)).max(0)
        additional_agent_gains = jnp.minimum(
            score_matrix - best_perf.tile((n_agents, 1)), 0
        )
        best_next_agent = additional_agent_gains.mean(-1).argmax()

        next_agent = agents.at[best_next_agent].set(1)
        return next_agent, best_perf.mean()

    final_agents, scores = jax.lax.scan(
        lambda a, _: next_agents_score(a),
        jnp.zeros(n_agents, dtype=jnp.int8),
        xs=None,
        length=n_agents,
    )
    prog_if_equal = jnp.linspace(score_matrix.mean(), scores[-1], n_agents)
    prog_if_unequal = jnp.ones(n_agents) * scores[-1]
    prog_if_unequal.at[0].set(score_matrix.mean())

    metrics["prog_per_agent_auc"] = (prog_if_equal[1:] - scores[1:]).mean() / (
        prog_if_equal[1:] - prog_if_unequal[1:]
    ).mean()
    metrics["best_1_agents"] = scores[1]
    metrics["best_2_agents"] = scores[2]
    metrics["best_3_agents"] = scores[3]
    metrics["best_0.25_agents"] = scores[n_agents // 4]
    metrics["best_0.5_agents"] = scores[n_agents // 2]
    metrics["best_0.75_agents"] = scores[3 * n_agents // 4]

    def get_agent_quantile(p):
        """
        :param p: fraction of the performance of the full population
        :return: minimum number of agents to get the performance fraction
        """
        return jnp.argmax(scores < ((1 - p) * score_matrix.mean() + p * scores[-1]))

    metrics["n_agents_0.99"] = get_agent_quantile(0.99)
    metrics["n_agents_0.9"] = get_agent_quantile(0.9)
    metrics["n_agents_0.75"] = get_agent_quantile(0.75)
    metrics["n_agents_0.5"] = get_agent_quantile(0.5)

    return metrics


def compute_cheap_metrics(episode_reward, key):
    """
    Compute cheap diversity metrics.
    """
    batch_size = episode_reward.shape[1]
    n_agents = episode_reward.shape[0]
    score_matrix = episode_reward.max(-1)
    metrics = {}

    def random_agents_score(agents_subpop, n_seeds=5):
        """
        :param agents_subpop: number of agents in the random subpopulation
        :param n_seed: number of random trial for averaging
        :return: mean performance of n_seed subpopulation of n_agents agents
        """
        random_scores = score_matrix[
            random.randint(key, (n_seeds * agents_subpop,), 0, n_agents),
            :,
        ]
        return jnp.reshape(random_scores, (n_seeds, agents_subpop, -1)).max(1).mean()

    metrics["random_2_agents"] = random_agents_score(2)
    metrics["random_3_agents"] = random_agents_score(3)
    metrics["random_0.25_agents"] = random_agents_score(n_agents // 4)
    metrics["random_0.5_agents"] = random_agents_score(n_agents // 2)
    metrics["random_0.75_agents"] = random_agents_score(3 * n_agents // 4)

    def get_best_instances_repartition():
        """
        :return: distribution of the best instances of each agent
        """
        best_instances_per_agent = jnp.bincount(
            jnp.argmax(score_matrix, axis=0),
            minlength=n_agents,
            length=n_agents,
        )
        return jnp.cumsum(jnp.sort(best_instances_per_agent)[::-1]) / batch_size

    best_instance_repartition = get_best_instances_repartition()
    metrics["num_best_instances_1"] = best_instance_repartition[0]
    metrics["num_best_instances_2"] = best_instance_repartition[1]
    metrics["num_best_instances_3"] = best_instance_repartition[2]
    metrics["num_best_instances_0.25"] = best_instance_repartition[n_agents // 4]
    metrics["num_best_instances_0.5"] = best_instance_repartition[n_agents // 2]
    metrics["num_best_instances_0.75"] = best_instance_repartition[3 * n_agents // 4]

    def get_number_ties():
        """
        :return: number of ties in the score matrix
        """
        return (
            score_matrix == score_matrix.max(axis=0, keepdims=True)
        ).sum() - batch_size

    metrics["number_ties"] = get_number_ties()

    return metrics


def get_contribution_agent(score_matrix):
    """
    return the difference of the population score with or without the agent
    """

    def perf_without_i(i):
        return score_matrix.at[i, :].set(-1e3).max(0).mean()

    return score_matrix.max(0).mean() - jax.vmap(perf_without_i, in_axes=0)(
        jnp.arange(score_matrix.shape[0])
    )


def get_contribution_agent_startpos(full_score_matrix):
    """
    return the difference of the population score with or without the agent considering each
    (instance, start_position) pair as a different problem
    """

    def perf_without_i(i):
        return full_score_matrix.at[i, :, :].set(-100).max(0).mean()

    return full_score_matrix.max(0).mean() - jax.vmap(perf_without_i, in_axes=0)(
        jnp.arange(full_score_matrix.shape[0])
    )
