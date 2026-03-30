# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Vision-specific mutable NAS building blocks.

This module contains higher-level search-space helpers that are commonly
needed when applying NAS to modern vision architectures such as Vision
Transformers (ViT), flavours of EfficientNet, or hierarchical ConvNets
(e.g. ConvNeXt).  Each class is built on top of the primitives already
provided in :mod:`nni.nas.nn.pytorch` so that all existing one-shot and
evolutionary strategies work out-of-the-box.
"""

from __future__ import annotations

from typing import List, Tuple, Union, cast

import torch
import torch.nn as nn
from nni.mutable import Categorical, MutableExpression, ensure_frozen

from .base import MutableModule, ParametrizedModule
from .layers import MutableConv2d, MutableLinear, MutableLayerNorm

__all__ = [
    'MutablePatchEmbedding',
]


class MutablePatchEmbedding(ParametrizedModule):
    """Patch embedding layer with searchable ``patch_size`` and ``embed_dim``.

    In Vision Transformer (ViT) architectures an image of shape
    ``(B, C, H, W)`` is first split into non-overlapping square patches of
    size ``patch_size × patch_size`` and then linearly projected into an
    ``embed_dim``-dimensional space.  Concretely the projection is implemented
    as a single strided :class:`~torch.nn.Conv2d` with
    ``kernel_size == stride == patch_size``.

    This class wraps that convolution with mutable ``patch_size`` and/or
    ``embed_dim`` parameters so that NAS strategies can search over them
    jointly.  Because changing ``patch_size`` also changes the sequence length
    emitted by the layer (``(H // patch_size) * (W // patch_size)`` tokens),
    users may call :meth:`seq_length` with the image resolution to obtain the
    symbolic expression for the number of patches under the current sample.

    Parameters
    ----------
    img_size : int
        Spatial resolution of the square input image (height == width).
        Used only to pre-compute the *maximum* sequence length.
    patch_size : int or Categorical[int]
        Side-length of each square patch.  Pass a
        :class:`~nni.mutable.Categorical` to make it searchable, e.g.::

            patch_size = nni.choice('patch_size', [8, 16, 32])
    in_channels : int
        Number of input image channels (e.g. 3 for RGB).
    embed_dim : int or Categorical[int]
        Projection (embedding) dimension.  Pass a
        :class:`~nni.mutable.Categorical` for a searchable value.
    bias : bool
        Whether the projection convolution uses a bias term.

    Examples
    --------
    Creating a ViT search space with searchable patch size and embedding
    dimension::

        import nni
        from nni.nas.nn.pytorch.layers_vision import MutablePatchEmbedding

        patch_embed = MutablePatchEmbedding(
            img_size=224,
            patch_size=nni.choice('patch_size', [8, 16, 32]),
            in_channels=3,
            embed_dim=nni.choice('embed_dim', [192, 384, 768]),
        )

    Notes
    -----
    * All candidates of ``patch_size`` must evenly divide ``img_size`` so that
      there is no partial patch at the image boundary.
    * The class inherits from :class:`~nni.nas.nn.pytorch.base.ParametrizedModule`
      which means it integrates seamlessly with the graph-based mutation engine
      and all one-shot weight-sharing strategies.
    * Position embeddings (absolute or relative) are **not** included here;
      they depend on the resulting sequence length which may itself be mutable
      and are better placed as a separate module in the model definition.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: Union[int, Categorical],
        in_channels: int,
        embed_dim: Union[int, Categorical],
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels

        # Validate that every candidate patch_size divides img_size.
        if isinstance(patch_size, Categorical):
            invalid = [p for p in patch_size.values if img_size % p != 0]
            if invalid:
                raise ValueError(
                    f'MutablePatchEmbedding: img_size={img_size} is not divisible by '
                    f'patch_size candidates {invalid}.'
                )

        # Store the mutable (or fixed) arguments for shape-inference helpers.
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # The projection: a strided convolution that simultaneously splits the
        # image into patches and projects each patch into embed_dim-dimensional
        # space.  Using MutableConv2d propagates any Categorical arguments into
        # the NNI mutable graph so that strategies can read and sample them.
        self.projection = MutableConv2d(
            in_channels,
            cast(int, embed_dim),
            kernel_size=cast(int, patch_size),
            stride=cast(int, patch_size),
            bias=bias,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def seq_length(self, img_size: int | None = None) -> MutableExpression:
        """Return the symbolic number of tokens produced for a given image size.

        Parameters
        ----------
        img_size : int, optional
            Image resolution to use for the computation.  Defaults to
            ``self.img_size`` set at construction time.

        Returns
        -------
        MutableExpression
            A (possibly mutable) integer expression representing
            ``(img_size // patch_size) ** 2``.
        """
        if img_size is None:
            img_size = self.img_size
        patches_per_side = MutableExpression.to_int(img_size // self.patch_size)
        return MutableExpression.to_int(patches_per_side * patches_per_side)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches into embedding space.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Patch token sequence of shape
            ``(B, (H // patch_size) * (W // patch_size), embed_dim)``.
        """
        # Shape: (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten spatial dims → token sequence: (B, embed_dim, N) → (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x
