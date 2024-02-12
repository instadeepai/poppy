import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def save_matrix_img(
    order_agent,  # (num_agents)
    score_matrix,  # (num_agents, num_instances)
    filename,  # str
    agent_score,  # (num_agents) agent's contributions will be appended to the matrix as the last column
):
    def flip_after_first(t):
        first_pos = t.argmax()
        minus_sign = jnp.ones(t.shape[0]) - 2 * (
            (jnp.arange(t.shape[0]) > first_pos).astype(int)
        )
        return t * minus_sign

    n_agents = score_matrix.shape[0]

    if order_agent is None:
        order_agent = (
            (score_matrix == score_matrix.max(axis=0, keepdims=True))
            .sum(axis=1)
            .argsort()[::-1]
        )

    new_trajs = (score_matrix == score_matrix.max(axis=0, keepdims=True))[
        order_agent, :
    ].astype(int)
    instance_num = (
        jnp.concatenate(
            (
                -jnp.power(jnp.array([2]), jnp.arange(n_agents // 2, 0, -1)),
                jnp.power(jnp.array([2]), jnp.arange(n_agents - n_agents // 2)),
            )
        )[::-1]
        * jax.vmap(flip_after_first, in_axes=1)(new_trajs)
    ).sum(1)

    order_instance = instance_num.argsort()[::-1]
    plot_matrix = (score_matrix == score_matrix.max(axis=0, keepdims=True))[order_agent]
    plot_matrix = plot_matrix[:, order_instance]
    ratio = plot_matrix.shape[1] // plot_matrix.shape[0]
    plot_matrix = jnp.repeat(plot_matrix, repeats=ratio, axis=0)

    if agent_score is not None:
        agent_score = (
            (agent_score - agent_score.min()) / (agent_score.max() - agent_score.min())
        )[order_agent]
        # add a column at the end of the matrix with the agent score
        agent_score = jnp.repeat(agent_score, repeats=ratio, axis=0)
        agent_score = jnp.repeat(agent_score[:, None], repeats=ratio, axis=1)
        plot_matrix = jnp.concatenate((plot_matrix, agent_score), axis=1)

    plt.figure(figsize=(10, 10))
    plt.imshow(plot_matrix)
    plt.savefig(filename)
    plt.close()

    return order_agent
