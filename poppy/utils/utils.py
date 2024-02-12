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
            if as_numpy:
                x = np.asarray(x)
            x = x.mean(axis=axis)
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
        x = list(x)
        if as_sharded_array:
            x = jax.device_put_sharded(x, devices)
        return x

    return jax.tree_util.tree_map(distribute_fn, x)


def generate_zeros_from_spec(spec: jnp.ndarray) -> jnp.ndarray:
    zeros: jnp.ndarray = jnp.zeros(spec.shape, spec.dtype)
    return zeros
