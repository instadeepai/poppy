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

from typing import Tuple

from chex import Array, PRNGKey
from jax import numpy as jnp
from jax import random


def compute_value_items(values: Array, used_mask: Array) -> Array:
    """
    Compute the value of the items in the knapsack.
    """
    return jnp.dot(used_mask, values)


def generate_problem(problem_key: PRNGKey, problem_size: jnp.int32) -> Array:
    return random.uniform(problem_key, (problem_size, 2), minval=0, maxval=1)


def generate_first_item(start_key: PRNGKey, problem_size: jnp.int32) -> jnp.int32:
    return random.randint(start_key, (), minval=0, maxval=problem_size)
