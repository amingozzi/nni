Construct Model Space
=====================

NNI provides powerful (and multi-level) APIs for users to easily express model space (or search space).

* *Mutation Primitives*: high-level APIs (e.g., LayerChoice) that are utilities to build blocks in search space. In most cases, mutation pritimives should be straightforward yet expressive enough. **We strongly recommend users to try them first,** and report issues if those APIs are not satisfying.
* *Hyper-module Library*: plug-and-play modules that are proved useful. They are usually well studied in research, and comes with pre-searched results. (For example, the optimal activation function in `AutoActivation <https://arxiv.org/abs/1710.05941>`__ is reported to be `Swish <https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html>`__).
* *Mutator*: for advanced users only. NNI provides interface to customize new mutators for expressing more complicated model spaces.

The following table summarizes all the APIs we have provided for constructing search space.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Name
     - Category
     - Brief Description
   * - :class:`~nni.nas.nn.pytorch.ModelSpace`
     - Mutation Primitives
     - All model spaces should inherit this class
   * - :class:`~nni.nas.nn.pytorch.ParametrizedModule`
     - Mutation Primitives
     - Modules with mutable parameters should inherit this class
   * - :class:`LayerChoice <nni.nas.nn.pytorch.LayerChoice>`
     - Mutation Primitives
     - Select from some PyTorch modules
   * - :class:`InputChoice <nni.nas.nn.pytorch.InputChoice>`
     - Mutation Primitives
     - Select from some inputs (tensors)
   * - :class:`Repeat <nni.nas.nn.pytorch.Repeat>`
     - Mutation Primitives
     - Repeat a block by a variable number of times
   * - :class:`Cell <nni.nas.nn.pytorch.Cell>`
     - Mutation Primitives
     - Cell structure popularly used in literature
   * - :class:`NasBench101Cell <nni.nas.hub.pytorch.modules.NasBench101Cell>`
     - Mutation Primitives
     - Cell structure (variant) proposed by NAS-Bench-101
   * - :class:`NasBench201Cell <nni.nas.hub.pytorch.modules.NasBench201Cell>`
     - Mutation Primitives
     - Cell structure (variant) proposed by NAS-Bench-201
   * - :class:`AutoActivation <nni.nas.hub.pytorch.modules.AutoActivation>`
     - Hyper-modules library
     - Searching for activation functions
   * - :class:`MutablePatchEmbedding <nni.nas.nn.pytorch.MutablePatchEmbedding>`
     - Vision Building Blocks
     - Mutable patch projection stem for Vision Transformers; ``patch_size`` and ``embed_dim`` are searchable
   * - :class:`MixedDepthwiseConv2d <nni.nas.oneshot.pytorch.supermodule.operation.MixedDepthwiseConv2d>`
     - Vision Building Blocks
     - Weight-sharing depthwise convolution with searchable ``kernel_size`` for one-shot strategies
   * - :class:`Mutator <nni.nas.space.Mutator>`
     - :doc:`Mutator <mutator>`
     - Flexible mutations on graphs. :doc:`See tutorial here <mutator>`

Vision Building Blocks
----------------------

NNI includes a library of mutable primitives targeted at **vision models** — particularly
convolutional backbones and Vision Transformers (ViTs). They can be used directly in
:class:`~nni.nas.nn.pytorch.ModelSpace` definitions and are compatible with all NNI strategies.

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Class
     - Brief Description
   * - :class:`~nni.nas.nn.pytorch.MutablePatchEmbedding`
     - Patch projection stem (``patch_size`` and ``embed_dim`` are searchable choices)
   * - :class:`~nni.nas.oneshot.pytorch.supermodule.operation.MixedDepthwiseConv2d`
     - Depthwise convolution with a searchable ``kernel_size``; requires ``groups == in_channels``

For detailed usage examples and integration with hardware-aware search, see
:doc:`vision_building_blocks`.
