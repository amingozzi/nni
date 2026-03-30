Vision Model Building Blocks
=============================

NNI provides a set of mutable building blocks designed specifically for **vision models**, including
convolutional networks (CNNs) and Vision Transformers (ViTs). These modules extend the standard
NNI mutable primitives to cover patterns that are common in modern vision architectures but were
previously not represented in the search space API.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Class
     - Brief Description
   * - :class:`~nni.nas.nn.pytorch.MutablePatchEmbedding`
     - Mutable patch projection for Vision Transformers; ``patch_size`` and ``embed_dim`` are searchable
   * - :class:`~nni.nas.oneshot.pytorch.supermodule.operation.MixedDepthwiseConv2d`
     - Weight-sharing depthwise convolution with a searchable ``kernel_size``

-----

MutablePatchEmbedding
---------------------

:class:`~nni.nas.nn.pytorch.MutablePatchEmbedding` replaces the fixed patch projection stem of a Vision
Transformer with a mutable equivalent. The patch size and embedding dimension are exposed as
:class:`~nni.mutable.Categorical` choices so that any one-shot or multi-trial strategy can search
over them.

.. code-block:: python

    from nni.nas.nn.pytorch import MutablePatchEmbedding, ModelSpace
    import nni

    class MyViTSpace(ModelSpace):
        def __init__(self):
            super().__init__()
            self.patch_embed = MutablePatchEmbedding(
                img_size=224,
                patch_size=nni.choice('patch_size', [8, 16, 32]),
                in_channels=3,
                embed_dim=nni.choice('embed_dim', [384, 512, 768]),
            )
            # ... rest of the transformer ...

        def forward(self, x):
            # patch_embed returns (B, num_patches, embed_dim)
            tokens = self.patch_embed(x)
            # continue with transformer encoder ...
            return tokens

The number of resulting tokens (``seq_length``) can be queried programmatically:

.. code-block:: python

    # seq_length adapts to the chosen patch_size
    n_tokens = model.patch_embed.seq_length()  # returns (img_size // patch_size) ** 2

**Constraints checked at construction time**

* Every candidate ``patch_size`` value must exactly divide ``img_size``.  An error is raised
  immediately if this invariant is violated, preventing silent incorrect behaviour at runtime.

.. note::

   ``MutablePatchEmbedding`` requires ``img_size`` to be square (same height and width).
   Rectangular inputs are currently not supported.

-----

MixedDepthwiseConv2d
--------------------

:class:`~nni.nas.oneshot.pytorch.supermodule.operation.MixedDepthwiseConv2d` is a weight-sharing
supermodule that wraps a depthwise separable convolution and makes its ``kernel_size`` a
:class:`~nni.mutable.Categorical` choice.  It extends the existing
:class:`~nni.nas.oneshot.pytorch.supermodule.operation.MixedConv2d` by **enforcing
``groups == in_channels``** at construction time, which is the canonical definition of a depthwise
convolution.  This is the same constraint enforced by :class:`torch.nn.Conv2d` when used in
depthwise mode.

.. code-block:: python

    from nni.nas.nn.pytorch import ModelSpace, LayerChoice
    from nni.nas.oneshot.pytorch.supermodule.operation import MixedDepthwiseConv2d
    import nni

    class MobileBlock(ModelSpace):
        def __init__(self, channels):
            super().__init__()
            # Depthwise conv with searchable kernel size
            self.dw = MixedDepthwiseConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=nni.choice('dw_kernel', [3, 5, 7]),
            )
            self.pw = torch.nn.Conv2d(channels, channels * 4, kernel_size=1)

        def forward(self, x):
            return self.pw(self.dw(x))

The ``padding`` is automatically set to ``kernel_size // 2`` for each candidate so that the spatial
resolution is preserved, mirroring the behaviour of :class:`MixedConv2d`.

.. tip::

   ``MixedDepthwiseConv2d`` is registered in ``NATIVE_MIXED_OPERATIONS`` and will therefore be
   automatically recognised by all built-in one-shot strategies (DARTS, ENAS, ProxylessNAS, etc.)
   without any extra configuration.

-----

Using Vision Building Blocks with One-shot Strategies
------------------------------------------------------

The vision building blocks integrate seamlessly with existing one-shot strategies:

.. code-block:: python

    from nni.nas.experiment import NasExperiment, NasExperimentConfig
    from nni.nas.evaluator.pytorch import Classification
    from nni.nas.strategy import DARTS
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T

    model_space = MyViTSpace()

    evaluator = Classification(
        train_dataloaders=DataLoader(
            CIFAR10('.', train=True, transform=T.ToTensor(), download=True),
            batch_size=64,
        ),
        val_dataloaders=DataLoader(
            CIFAR10('.', train=False, transform=T.ToTensor()),
            batch_size=64,
        ),
    )

    config = NasExperimentConfig('local', 'DARTS', evaluator)
    experiment = NasExperiment(model_space, evaluator, config)
    experiment.start(8080)

-----

See Also
--------

* :doc:`construct_space` — full list of mutation primitives
* :doc:`hardware_aware_nas` — adding FLOPs / latency constraints to the search
* :ref:`Compression of Transformer modules <compression-transformer-modules>` — pruning and
  quantization support for LayerNorm, MultiheadAttention, GELU/SiLU and related activation layers
