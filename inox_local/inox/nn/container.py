r"""Container modules"""

__all__ = [
    "Sequential",
]


from textwrap import indent
from typing import Any
import jax

import inox.tree

from .module import Module


class Sequential(Module):
    r"""Creates a composition of layers.

    .. math:: y = f_n \circ \dots \circ f_2 \circ f_1(x)

    Arguments:
        layers: A sequence of layers :math:`f_1, f_2, \dots, f_n`.
    """

    def __init__(self, *layers: Module):
        self.layers = layers

    def __call__(self, x, key=None, **kwargs):
        r"""
        Arguments:
            x: The input :math:`x`.

        Returns:
            The output :math:`y`.
        """
        if key is not None:
            keys = jax.random.split(key, len(self.layers))
            key_iter = iter(keys)
        else:
            key_iter = iter([None] * len(self.layers))
            
        for layer in self.layers:
            layer_key = next(key_iter)
            if hasattr(layer, '__call__'):
                # Check if layer accepts key parameter
                import inspect
                sig = inspect.signature(layer.__call__)
                if 'key' in sig.parameters:
                    x = layer(x, key=layer_key, **kwargs)
                else:
                    x = layer(x, **kwargs)
            else:
                x = layer(x, **kwargs)
        return x

    def tree_repr(self, **kwargs) -> str:
        lines = (inox.tree.prepr(layer, **kwargs) for layer in self.layers)
        lines = ",\n".join(lines)

        if lines:
            lines = "\n" + indent(lines, "  ") + "\n"

        return f"{self.__class__.__name__}({lines})"
