# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax.numpy as jnp
from jax import random
from chex import Array, PRNGKey


def compute_tour_length(problem: Array, order: Array) -> jnp.float32:
    """Calculate the length of a tour."""
    problem = problem[order]
    return jnp.linalg.norm((problem - jnp.roll(problem, -1, axis=0)), axis=1).sum()


def generate_problem(problem_key: PRNGKey, num_cities: jnp.int32) -> Array:
    return random.uniform(problem_key, (num_cities, 2), minval=0, maxval=1)


def generate_start_position(start_key: PRNGKey, num_cities: jnp.int32) -> jnp.int32:
    return random.randint(start_key, (), minval=0, maxval=num_cities)


def get_coordinates_augmentations(coordinates: Array) -> Array:
    """
    Returns the 8 augmentations of the coordinates of a given instance problem described in [1].
    [1] https://arxiv.org/abs/2010.16011
    Usages: TSP and CVRP.
    Args:
        coordinates: array of coordinates for all cities [problem_size, 2]
    Returns:
        Array with 8 augmentations [8, problem_size, 2]
    """

    # Coordinates -> (1 - coordinates) for each city
    rotated_coordinates = jnp.array(
        [
            coordinates,
            jnp.transpose(jnp.array([1 - coordinates[:, 0], coordinates[:, 1]])),
            jnp.transpose(jnp.array([coordinates[:, 0], 1 - coordinates[:, 1]])),
            jnp.transpose(jnp.array([1 - coordinates[:, 0], 1 - coordinates[:, 1]])),
        ]
    )

    # Coordinates are also flipped
    flipped_coordinates = jnp.einsum(
        "ijk -> jki",
        jnp.array([rotated_coordinates[:, :, 1], rotated_coordinates[:, :, 0]]),
    )

    return jnp.concatenate([rotated_coordinates, flipped_coordinates], axis=0)
