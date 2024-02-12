import random
import string
import time
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np


def fetch_from_devices(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        if as_numpy and isinstance(x, jax.pxla.ShardedDeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def reduce_from_devices(x, axis=0, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        if isinstance(x, jax.pxla.ShardedDeviceArray):
            x = x.mean(axis=axis)
            if as_numpy:
                x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def fetch_from_first_device(x, as_numpy: bool = True):
    """Converts a distributed TrainingState to a single-device TrainingState."""

    def fetch_fn(x):
        x = x[0]
        if as_numpy and isinstance(x, jax.xla.DeviceArray):
            x = np.asarray(x)
        return x

    return jax.tree_util.tree_map(fetch_fn, x)


def spread_over_devices(x, devices=None, as_sharded_array=True):
    """Converts a single-device jnp array to a distributed jnp array."""
    devices = devices or jax.local_devices()

    def distribute_fn(x):
        x = x.reshape(len(devices), -1, *(x.shape[1:]))
        x = [x_i for x_i in x]
        if as_sharded_array:
            x = jax.device_put_sharded(x, devices)
        return x

    return jax.tree_util.tree_map(distribute_fn, x)


def make_log_name(random_size=5):
    """Return a directory name using the experiment launch time and a random
    end string to avoid collisions.

    Args:
        random_size: int
        The length of the random postfix.
    Returns:
         A string incorporating the current date and a random postfix.
    """

    def _random_str():
        a = list(string.ascii_letters)
        random.shuffle(a)
        return "".join(a[:random_size])

    return time.strftime("%Y_%m_%d_%H_%M_%S_") + _random_str()


def dataclass_to_dict(obj):
    """Convert a dataclass object to a dict, whether imported from chex or dataclasses package.

    Args:
        obj: The dataclass object to convert.

    Returns:
        A dict representation of the dataclass object.
    """
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


def generate_zeros_from_spec(spec: jnp.ndarray) -> jnp.ndarray:
    zeros: jnp.ndarray = jnp.zeros(spec.shape, spec.dtype)
    return zeros
