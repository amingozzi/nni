Install NNI
===========

This source tree requires Python >= 3.10.
It is tested and supported on Ubuntu >= 18.04,
Windows 10 >= 21H2, and macOS >= 11.

There are 3 ways to install NNI:

* :ref:`Using pip <installation-pip>`
* :ref:`Build source code <installation-source>`
* :ref:`Using Docker <installation-docker>`

.. _installation-pip:

Using pip
---------

NNI provides official packages for x86-64 CPUs. They can be installed with pip:

.. code-block:: text

    pip install nni

Or to upgrade to latest version:

.. code-block:: text

    pip install --upgrade nni

You can check installation with:

.. code-block:: text

    nnictl --version

On Linux systems without Conda, you may encounter ``bash: nnictl: command not found`` error.
In this case you need to add pip script directory to ``PATH``:

.. code-block:: bash

    echo 'export PATH=${PATH}:${HOME}/.local/bin' >> ~/.bashrc
    source ~/.bashrc

.. _installation-source:

Installing from Source Code
---------------------------

This fork hosts source code on `GitHub <https://github.com/amingozzi/nni>`__.

NNI has experimental support for ARM64 CPUs, including Apple M1.
It requires to install from source code.

See :doc:`/notes/build_from_source`.

.. _installation-docker:

Using Docker
------------

NNI provides official Docker image on `Docker Hub <https://hub.docker.com/r/msranni/nni>`__.

.. code-block:: text

    docker pull msranni/nni

Installing Extra Dependencies
-----------------------------

For this source checkout, install the framework dependencies with an editable install:

.. code-block:: text

    python -m pip install -e ".[nas]"

The ``nas`` extra includes PyTorch, torchvision, PyTorch Lightning, torchmetrics, and TensorBoard.
Use ``.[compression]`` for native PyTorch compression without Lightning, or ``.[pytorch]``
for PyTorch and torchvision only. These extras require PyTorch >= 2.6 and Lightning >= 2.6
where applicable; the current compatibility targets are PyTorch 2.14.0 and Lightning 2.6.6.
NNI's Lightning integrations use the ``pytorch_lightning`` namespace.
See the `Lightning 2.6 training API
<https://lightning.ai/docs/pytorch/2.6.6/common/trainer.html>`__.

For reproducible framework versions:

.. code-block:: text

    python -m pip install -r dependencies/pytorch.txt

Choose the CPU or CUDA wheels using the `official PyTorch installation selector
<https://pytorch.org/get-started/locally/>`__ before installing NNI.
The optional OpenMMLab and TensorRT extensions must match the chosen PyTorch/CUDA build;
they are not covered by the CPU compatibility workflow.

``TorchSerializer`` loads trusted strategy checkpoints with ``weights_only=False`` so that
NumPy random states and other Python objects can be restored. For tensor-only checkpoints,
use ``TorchSerializer(weights_only=True)``. Never load an untrusted pickle checkpoint.
See `PyTorch serialization semantics
<https://docs.pytorch.org/docs/2.14/notes/serialization.html>`__.

NNI's existing ONNX visualization, latency profiling, and TensorRT export paths explicitly
use ``dynamo=False`` to retain their TorchScript exporter behavior. The new exporter has
different graph and dependency requirements; see `torch.onnx
<https://docs.pytorch.org/docs/2.14/onnx.html>`__.

Some built-in algorithms of NNI requires extra packages.
Use ``nni[<algorithm-name>]`` to install their dependencies.

For example, to install dependencies of :class:`DNGO tuner<nni.algorithms.hpo.dngo_tuner.DNGOTuner>` :

.. code-block:: text

    pip install nni[DNGO]

This command will not reinstall NNI itself, even if it was installed in development mode.

Alternatively, you may install all extra dependencies at once:

.. code-block:: text

    pip install nni[all]

**NOTE**: SMAC tuner depends on swig3, which requires a manual downgrade on Ubuntu:

.. code-block:: bash

    sudo apt install swig3.0
    sudo rm /usr/bin/swig
    sudo ln -s swig3.0 /usr/bin/swig
