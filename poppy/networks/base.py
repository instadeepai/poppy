from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array

from poppy.environments.cvrp.types import Observation as CVRPObservation
from poppy.environments.knapsack.types import Observation as KnapsackObservation
from poppy.environments.tsp.types import Observation as TSPObservation

if TYPE_CHECKING:
    from dataclasses import dataclass

else:
    from chex import dataclass


@dataclass
class Networks:  # type: ignore

    encoder_fn: hk.Transformed
    decoder_fn: hk.Transformed


class EncoderBase(ABC, hk.Module):
    """Transformer-based encoder.

    By default, this is the encoder used by Kool et al. (arXiv:1803.08475) and
    Kwon et al. (arXiv:2010.16011).
    """

    def __init__(
        self,
        num_layers,
        num_heads,
        key_size,
        model_size=None,
        expand_factor=4,
        name="encoder",
    ):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.expand_factor = expand_factor

    def __call__(self, problem: Array) -> Array:
        x = self.get_problem_projection(problem)

        for i in range(self.num_layers):
            mha = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=self.model_size,
                w_init_scale=1 / self.num_layers,
                name=f"mha_{i}",
            )
            norm1 = hk.LayerNorm(
                axis=-1,  # should be batch norm according to Kool et al.
                create_scale=True,
                create_offset=True,
                name=f"norm_{i}a",
            )

            x = norm1(x + mha(query=x, key=x, value=x))

            mlp = hk.nets.MLP(
                output_sizes=[self.expand_factor * self.model_size, self.model_size],
                activation=jax.nn.relu,
                activate_final=False,
                name=f"mlp_{i}",
            )
            # should be batch norm according to POMO
            norm2 = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name=f"norm_{i}b"
            )
            x = norm2(x + mlp(x))

        return x

    @abstractmethod
    def get_problem_projection(self, problem: Array) -> Array:
        pass


class DecoderBase(ABC, hk.Module):
    """
    Decoder module.
    By default, this is the decoder used by Kool et al. (arXiv:1803.08475) and Kwon et al. (arXiv:2010.16011).
    """

    def __init__(self, num_heads, key_size, model_size=128, name="decoder"):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size

    def __call__(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        context = self.get_context(observation, embeddings)
        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.key_size,
            model_size=self.model_size,
            w_init_scale=1,
            name="mha_dec",
        )

        attention_mask = jnp.expand_dims(observation.action_mask, (0, 1))
        context = mha(
            query=context,
            key=embeddings,
            value=embeddings,
            mask=self.get_transformed_attention_mask(attention_mask),
        )  # --> [128]

        attn_logits = (
            embeddings @ context.squeeze() / jnp.sqrt(self.model_size)
        )  # --> [num_cities/items]
        attn_logits = 10 * jnp.tanh(attn_logits)  # clip to [-10,10]

        return attn_logits

    @abstractmethod
    def get_context(
        self,
        observation: Union[TSPObservation, KnapsackObservation, CVRPObservation],
        embeddings: Array,
    ) -> Array:
        pass

    @abstractmethod
    def get_transformed_attention_mask(self, attention_mask: Array) -> Array:
        pass
