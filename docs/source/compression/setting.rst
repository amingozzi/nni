Compression Setting
===================

To enhance the compression compatibility of any type of module, NNI 3.0 introduces the notion of ``setting``,
which is a pre-established template utilized to depict compression information. 
The primary objective of this pre-established template is to support shorthand writing in the ``config_list``.

The list of NNI default supported module types can be accessed via this `link <https://github.com/microsoft/nni/tree/master/nni/compression/base/setting.py>`__.
Please review the ``registry`` in ``PruningSetting``, ``QuantizationSetting`` and ``DistillationSetting`` to see the supported module type and its default setting.
It should be noted that ``DistillationSetting`` will automatically register a default output setting for all module types,
which implies that distilling any module output is available by design.

.. _compression-transformer-modules:

Built-in Transformer and Modern Vision Module Support
-----------------------------------------------------

In addition to the classic CNN modules (``Conv2d``, ``BatchNorm2d``, ``Linear``, …), NNI now ships
with pre-registered settings for the Transformer-era and modern vision modules listed below.
No additional registration is needed to compress these layer types.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 20

   * - Module Type
     - Pruning
     - Quantization
     - Notes
   * - ``torch.nn.LayerNorm``
     - ✓
     - ✓
     - Weight and bias; used in every Transformer block
   * - ``torch.nn.GroupNorm``
     - ✓
     - ✓
     - Common in diffusion and segmentation models
   * - ``torch.nn.InstanceNorm1d``
     - ✓
     - ✓
     - Style-transfer and sequence normalisation
   * - ``torch.nn.InstanceNorm2d``
     - ✓
     - ✓
     - Image style-transfer normalisation
   * - ``torch.nn.MultiheadAttention``
     - ✓
     - ✓
     - ``in_proj_weight``, ``out_proj.weight``; head-level pruning granularity recommended
   * - ``torch.nn.GELU``
     - —
     - ✓
     - Output quantization; default activation in BERT / ViT / GPT-2
   * - ``torch.nn.SiLU``
     - —
     - ✓
     - Output quantization; used in EfficientNet, YOLOv8, and ViT variants
   * - ``torch.nn.Hardswish``
     - —
     - ✓
     - Output quantization; used in MobileNetV3

Example — pruning a ViT encoder block including its LayerNorm layers:

.. code-block:: python

    from nni.compression.pytorch.pruning import L1NormPruner

    config_list = [
        # prune all Linear layers (Q, K, V projections, FFN) to 50 % sparsity
        {'op_types': ['Linear'], 'sparse_ratio': 0.5},
        # also prune LayerNorm weight to align with pruned feature dimensions
        {'op_types': ['LayerNorm'], 'sparse_ratio': 0.5},
    ]

    pruner = L1NormPruner(model, config_list)
    _, masks = pruner.compress()

Example — quantizing a vision model that uses SiLU activations:

.. code-block:: python

    from nni.compression.pytorch.quantization import QATQuantizer

    config_list = [
        {'op_types': ['Conv2d', 'Linear'], 'quant_dtypes': ['int8']},
        # quantize SiLU output to capture downstream activation statistics
        {'op_types': ['SiLU'], 'target_names': ['_output_'], 'quant_dtypes': ['int8']},
    ]

    quantizer = QATQuantizer(model, config_list, training_step=training_step)

Register Setting
----------------

If you discover that the module type you intend to compress is unavailable, you may register it into the appropriate compression setting.
However, if the registered module type is already present in the registry, the new setting will directly replace the old one.
In the subsequent sections, you will learn how to register settings in various compression scenarios. 

You can find all the supported compression setting keys and their explanations :doc:`here <config_list>`.

If you only want to make temporary settings for a specific layer without affecting other default templates of the same module type,
please refer to section.

Assume a customized module type and we want to compress it:

.. code-block:: python

    class CustomizedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = torch.nn.Parameter(200, 100)
            self.p2 = torch.nn.Parameter(200, 50)

        def forward(self, x, y):
            return torch.matmul(self.p1, x) + torch.matmul(self.p2, y)

Pruning Setting
^^^^^^^^^^^^^^^

Here is a default setting for pruning module weight and bias.

.. code-block:: python

    default_setting = {
        'weight': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'granularity': 'default',
            'internal_metric_block': None,
            'apply_method': 'mul',
        },
        'bias': {
            'align': {
                'module_name': None,
                'target_name': 'weight',
                'dims': [0],
            },
            'apply_method': 'mul',
        }
    }

We can create a setting for ``CustomizedModule`` by following the above default setting.

.. code-block:: python

    customized_setting = {
        'p1': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'granularity': [1, -1],
            'internal_metric_block': None,
            'apply_method': 'mul',
        },
        'p2': {
            'sparse_ratio': None,
            'max_sparse_ratio': None,
            'min_sparse_ratio': None,
            'sparse_threshold': None,
            'global_group_id': None,
            'dependency_group_id': None,
            'granularity': [1, -1],
            'internal_metric_block': None,
            'apply_method': 'mul',
        },
        '_output_': {
            'align': {
                'module_name': None,
                'target_name': 'p1',
                'dims': [0],
            },
            'apply_method': 'mul',
            'granularity': [-1, 1]
        }
    }

    PruningSetting.register('CustomizedModule', customized_setting)

The customized setting means that ``p1`` and ``p2`` will be applied channel-wise masks on the first dim of parameter,
``_output_`` will be applied channel-wise masks on the second dim of output.
Instead of generating masks by pruning algorithms, the output masks is generated by align with ``p1`` masks on the first dim.

Quantization Setting
^^^^^^^^^^^^^^^^^^^^

Here is a default setting for quantizing module inputs, outputs and weight.

.. code-block:: python

    default_setting = {
        '_input_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        'weight': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        },
        '_output_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': 'default',
            'apply_method': 'clamp_round',
        }
    }

Just modified the keys and registered it to ``QuantizationSetting`` are all you need for quantizing this module.

.. code-block:: python

    customized_setting = {
        '_input_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': [-1, 1],
            'apply_method': 'clamp_round',
        },
        'p1': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': [1, -1],
            'apply_method': 'clamp_round',
        },
        'p2': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': [1, -1],
            'apply_method': 'clamp_round',
        },
        '_output_': {
            'quant_dtype': None,
            'quant_scheme': None,
            'granularity': [-1, 1],
            'apply_method': 'clamp_round',
        }
    }

    QuantizationSetting.register('CustomizedModule', customized_setting)

Temporarily Setting Update
--------------------------

Sometimes we just want to temporarily modify the setting template and don't want to make a global change.
Then we could directly write full setting in ``config_list`` to achieve this.

For example, if the compressed model has conv-bn-relu pattern, and for a better pruning simulation and performance,
we want to mask the batchnorm on the channels convolution masked.
Then we could temporarily make batchnorm masks align with convolution layer weight masks.

.. code-block:: python

    config_list = [{
        'op_names': ['conv1', 'conv2'],
        'granularity': 'out_channel',
    }, {
        'op_names': ['bn1'],
        'target_settings': {
            'weight': {
                'align': {
                    'module_name': 'conv1',
                    'target_name': 'weight',
                    'dims': [0],
                },
                'granularity': 'out_channel',
            }
        }
    }, {
        'op_names': ['bn2'],
        'target_settings': {
            'weight': {
                'align': {
                    'module_name': 'conv2',
                    'target_name': 'weight',
                    'dims': [0],
                },
                'granularity': 'out_channel',
            }
        }
    }]
