import jax
import jax.numpy as jnp
from jax import random


def get_metrics(metrics, performance_matrix, key=None, compute_expensive_metrics=False):
    """Get the performance metrics of the population of agents.

    Args:
        metrics: A dictionary with the current metrics.
        performance_matrix: The cumulative episode return of size [B, N, S], where B is the batch
        size, N is the number of agents and S is the number of starting positions.
        key: A PRNGKey (which may be None).
        compute_expensive_metrics: A boolean that determines whether the computationally expensive
        metrics are calculated or not.

    Returns:
        metrics: An updated dictionary with additional metrics added.
    """
    performance_matrix = performance_matrix.max(-1)  # [BATCH_SIZE, NUM_AGENTS]

    if not key:
        key = random.PRNGKey(0)

    metrics.update(get_agent_contributions(performance_matrix))
    metrics.update(get_pop_performance_with_random_agents(performance_matrix, key))
    metrics.update(get_best_instances_with_n_agents(performance_matrix))
    metrics.update(get_number_ties(performance_matrix))

    if compute_expensive_metrics:
        additional_metrics, scores = get_pop_performance_with_best_agents(
            performance_matrix
        )
        metrics.update(additional_metrics)
        metrics.update(
            get_minimum_agents_to_reach_performance(performance_matrix, scores)
        )

    return metrics


def get_agent_contributions(performance_matrix):
    """Compute the difference of the population score in scenarios with and without each agent.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.

    Return:
        metrics: A dictionary containing the contribution of agent to the population score.
    """

    def performance_without_agent(x):
        """
        Returns: the mean performance without agent 'x' in the population.
        """
        return performance_matrix.at[:, x].set(-1e3).max(1).mean()

    agent_contributions = performance_matrix.max(1).mean() - jax.vmap(
        performance_without_agent, in_axes=0
    )(jnp.arange(performance_matrix.shape[1]))

    metrics = {}
    for i in range(len(agent_contributions)):
        metrics[f"contribution_agent_{i}"] = agent_contributions[i]

    return metrics


def get_pop_performance_with_random_agents(performance_matrix, key):
    """Compute the population performance with a certain number of random agents.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.

    Return:
        metrics: A dictionary containing the population performance with a certain number of random
        agents in the population.
    """
    total_agents = performance_matrix.shape[1]

    def pop_score_with_n_random_agents(agents_subpop, n_seeds=5):
        """
        Args:
            agents_subpop: number of agents in the random subpopulation.
            n_seeds: number of random trial for averaging.
        Returns: the mean performance of n_seed subpopulation of n_agents agents.
        """
        random_scores = performance_matrix[
            :, random.randint(key, (n_seeds * agents_subpop,), 0, total_agents)
        ]
        return (
            jnp.reshape(
                random_scores, (performance_matrix.shape[0], n_seeds, agents_subpop)
            )
            .max(2)
            .mean()
        )

    metrics = {}
    if total_agents >= 3:
        metrics.update(
            {"performance_with_random_agents_2": pop_score_with_n_random_agents(2)}
        )
    if total_agents >= 4:
        metrics.update(
            {"performance_with_random_agents_3": pop_score_with_n_random_agents(3)}
        )
    if total_agents > 4:
        metrics.update(
            {
                "performance_with_random_agents_25%_pop": pop_score_with_n_random_agents(
                    total_agents // 4
                ),
                "performance_with_random_agents_50%_pop": pop_score_with_n_random_agents(
                    total_agents // 2
                ),
                "performance_with_random_agents_75%_pop": pop_score_with_n_random_agents(
                    3 * total_agents // 4
                ),
            }
        )

    return metrics


def get_best_instances_with_n_agents(performance_matrix):
    """Compute the percentage of problem instances that 'n' best agents achieve the max performance.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.

    Return:
        metrics: A dictionary containing the percentage of problem instances that a certain number
        of agents perform the best in.
    """
    batch_size, total_agents = performance_matrix.shape[0], performance_matrix.shape[1]

    def get_best_instances_repartition():
        """
        Returns: the distribution of the best instances of each agent.
        """
        best_instances_per_agent = jnp.bincount(
            jnp.argmax(performance_matrix, axis=1),
            minlength=total_agents,
            length=total_agents,
        )
        return jnp.cumsum(jnp.sort(best_instances_per_agent)[::-1]) / batch_size

    best_instance_repartition = get_best_instances_repartition()

    metrics = {}

    if total_agents >= 3:
        metrics.update(
            {
                "num_instances_solved_with_best_1_agent": best_instance_repartition[0],
                "num_instances_solved_with_best_2_agent": best_instance_repartition[1],
                "num_instances_solved_with_best_3_agent": best_instance_repartition[2],
            }
        )
    if total_agents >= 4:
        metrics.update(
            {
                "num_instances_solved_with_best_25%_pop": best_instance_repartition[
                    total_agents // 4
                ],
                "num_instances_solved_with_best_50%_pop": best_instance_repartition[
                    total_agents // 2
                ],
                "num_instances_solved_with_best_75%_pop": best_instance_repartition[
                    3 * total_agents // 4
                ],
            }
        )

    return metrics


def get_number_ties(performance_matrix):
    """Compute the number of ties (i.e., equal performance) between the agents.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.

    Return:
        metrics: A dictionary containing the number of ties in the score matrix.
    """
    batch_size = performance_matrix.shape[0]
    metrics = {
        "number_ties": (
            performance_matrix == performance_matrix.max(axis=1, keepdims=True)
        ).sum()
        - batch_size
    }

    return metrics


def get_pop_performance_with_best_agents(performance_matrix):
    """Compute the population performance with a specific number of best agents.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.

    Returns:
        metrics: A dictionary containing the performance of the population with a certain number
        of the best agents.
        scores: An array of size equal to population size, where the indices represent the number
        of best agents, and the value represents the mean performance of the population with those
        best agents.
    """
    batch_size, total_agents = performance_matrix.shape[0], performance_matrix.shape[1]

    def next_agents_score(agents):
        """
        Args:
            agents: one hot binary vector of length pop_size. Ones for agents in the subpopulation,
            zeros for others.
        Returns:
            next_agent: the updated one hot binary vector of length pop_size with the next best
            agent added.
            best_perf.mean: the performance of the subpopulation on the validation problems.
        """
        selected_agents = agents.tile((batch_size, 1))
        best_perf = (
            performance_matrix + jnp.where(selected_agents == 0, -1000, 0)
        ).max(1)

        additional_agent_gains = jnp.minimum(
            performance_matrix - best_perf.tile((total_agents, 1)).transpose(), 0
        )
        best_next_agent = additional_agent_gains.mean(0).argmax()
        next_agent = agents.at[best_next_agent].set(1)

        return next_agent, best_perf.mean()

    final_agents, scores = jax.lax.scan(
        lambda a, _: next_agents_score(a),
        jnp.zeros(total_agents, dtype=jnp.int8),
        xs=None,
        length=total_agents,
    )

    metrics = {}
    if total_agents >= 3:
        metrics.update(
            {
                "performance_of_best_1_agent": scores[1],
                "performance_of_best_2_agent": scores[2],
                "performance_of_best_3_agent": scores[3],
            }
        )
    if total_agents >= 4:
        metrics.update(
            {
                "performance_of_best_25%_pop": scores[total_agents // 4],
                "performance_of_best_50%_pop": scores[total_agents // 2],
                "performance_of_best_75%_pop": scores[3 * total_agents // 4],
            }
        )

    return metrics, scores


def get_minimum_agents_to_reach_performance(performance_matrix, scores):
    """Compute the minimum number of agents needed to reach certain percentage of the performance.

    Args:
        performance_matrix: The cumulative episode return of size [B, N], where B is the batch size
        and N is the number of agents.
        scores: An array of size equal to population size, where the indices represent the number
        of best agents, and the value represents the mean performance of the population with those
        best agents.

    Returns:
        metrics: A dictionary containing the minimum number of agents needed to reach a certain
        performance fraction.
    """

    def get_agent_quantile(p):
        """
        Args:
            p: fraction of the performance of the full population.
        Return: minimum number of agents to get the performance fraction.
        """
        return jnp.argmax(
            scores < ((1 - p) * performance_matrix.mean() + p * scores[-1])
        )

    metrics = {
        "n_agents_to_achieve_99%_performance": get_agent_quantile(0.99),
        "n_agents_to_achieve_90%_performance": get_agent_quantile(0.9),
        "n_agents_to_achieve_75%_performance": get_agent_quantile(0.75),
        "n_agents_to_achieve_50%_performance": get_agent_quantile(0.5),
    }

    return metrics
