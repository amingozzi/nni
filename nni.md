# NNI — Complete API Reference for Coding Agents

**Version:** 3.0  
**Package:** `nni`  
**Repository:** https://github.com/microsoft/nni  
**Purpose:** Automated Machine Learning toolkit covering Hyperparameter Optimization (HPO),
Neural Architecture Search (NAS), and Model Compression for PyTorch (primary) and other frameworks.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Top-Level API (`import nni`)](#2-top-level-api-import-nni)
3. [HPO — Trial API](#3-hpo--trial-api)
4. [HPO — Experiment API](#4-hpo--experiment-api)
5. [HPO — Search Space Format](#5-hpo--search-space-format)
6. [HPO — Built-in Tuners & Assessors](#6-hpo--built-in-tuners--assessors)
7. [NAS — Mutable Primitives](#7-nas--mutable-primitives)
8. [NAS — Model Space Construction](#8-nas--model-space-construction)
9. [NAS — Vision Building Blocks](#9-nas--vision-building-blocks)
10. [NAS — Evaluators](#10-nas--evaluators)
11. [NAS — Exploration Strategies](#11-nas--exploration-strategies)
12. [NAS — Strategy Middleware](#12-nas--strategy-middleware)
13. [NAS — One-Shot Strategies (Weight Sharing)](#13-nas--one-shot-strategies-weight-sharing)
14. [NAS — Hardware-Aware NAS & Profilers](#14-nas--hardware-aware-nas--profilers)
15. [NAS — NasExperiment](#15-nas--nasexperiment)
16. [NAS — Model Space Hub](#16-nas--model-space-hub)
17. [Compression — Overview & config_list Format](#17-compression--overview--config_list-format)
18. [Compression — Pruning](#18-compression--pruning)
19. [Compression — Quantization](#19-compression--quantization)
20. [Compression — Distillation](#20-compression--distillation)
21. [Compression — Speedup (Mask Application)](#21-compression--speedup-mask-application)
22. [Compression — Evaluators](#22-compression--evaluators)
23. [Compression — Supported Module Types](#23-compression--supported-module-types)
24. [Serialization Utilities](#24-serialization-utilities)
25. [Common Patterns & End-to-End Examples](#25-common-patterns--end-to-end-examples)

---

## 1. Installation & Setup

```bash
# From PyPI (CPU)
pip install nni

# With recommended extras
pip install nni[BOHB]     # BOHB advisor
pip install nni[SMAC]     # SMAC tuner
pip install nni[NAS]      # NAS dependencies (pytorch-lightning, etc.)

# From source (skipping TypeScript build)
python setup.py develop --skip-ts
```

**Framework default** — PyTorch is the default. Change it before any other NNI import:

```python
import nni
nni.set_default_framework('pytorch')   # default; also 'tensorflow', 'mxnet', 'none'
```

Environment variable alternative: `NNI_FRAMEWORK=pytorch` (set before process starts).

---

## 2. Top-Level API (`import nni`)

```python
import nni
```

### Search space shortcut functions

These create `Mutable` objects used in both HPO search spaces and NAS model spaces.

| Function | Returns | Description |
|---|---|---|
| `nni.choice(label, choices)` | `Categorical` or `LayerChoice` | Pick one item from a list. If items are `nn.Module`, creates a `LayerChoice`. |
| `nni.uniform(label, low, high)` | `Numerical` | Uniform float in `[low, high]`. |
| `nni.quniform(label, low, high, q)` | `Numerical` | Quantised uniform: values are multiples of `q`. |
| `nni.loguniform(label, low, high)` | `Numerical` | Log-uniform float; search in log space. |
| `nni.qloguniform(label, low, high, q)` | `Numerical` | Quantised log-uniform. |
| `nni.normal(label, mu, sigma)` | `Numerical` | Normal distribution. |
| `nni.qnormal(label, mu, sigma, q)` | `Numerical` | Quantised normal. |

```python
lr   = nni.loguniform('lr', 1e-4, 1e-1)
drop = nni.uniform('dropout', 0.0, 0.5)
act  = nni.choice('activation', ['relu', 'tanh', 'sigmoid'])
```

### Serialization helpers

| Symbol | Description |
|---|---|
| `nni.trace(cls_or_fn)` | Decorator that records construction args so the object can be re-created from JSON. |
| `nni.dump(obj)` | Serialize a `@trace`-decorated object to a JSON-compatible dict. |
| `nni.load(data)` | Deserialize a dict produced by `nni.dump`. |

```python
@nni.trace
class MyBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

block = MyBlock(64)
data  = nni.dump(block)    # {'__type__': '...MyBlock', 'args': [64]}
block2 = nni.load(data)    # reconstructed MyBlock(64)
```

### Other top-level names

| Symbol | Description |
|---|---|
| `nni.Experiment` | Class to launch and manage an HPO experiment (see §4). |
| `nni.NoMoreTrialError` | Exception raised when the search space is fully explored. |
| `nni.training_update` | Notify NNI of weight-sharing model updates (used internally by one-shot NAS). |
| `nni.enable_global_logging(level)` | Enable NNI's Python logging at the given level. |
| `nni.silence_stdout()` | Redirect trial stdout to NNI log. |
| `nni.ClassArgsValidator` | Utility for validating algorithm `class_args` in config. |

---

## 3. HPO — Trial API

Import: `import nni` (these are exported from `nni.trial` but re-exported at the top level).

These functions are called **inside a trial script** (the script that trains a model).

```python
import nni

# 1. Receive hyperparameters from the tuner
params = nni.get_next_parameter()
# params is a dict, e.g. {'lr': 0.01, 'dropout': 0.3, 'activation': 'relu'}

lr       = params['lr']
dropout  = params['dropout']

# 2. Train the model ...
for epoch in range(max_epochs):
    train(...)
    val_acc = evaluate(...)
    # 3. Report intermediate metrics (optional, used by assessors for early stopping)
    nni.report_intermediate_result(val_acc)

# 4. Report final metric (required)
nni.report_final_result(best_val_acc)
```

### Full Trial API

| Function | Description |
|---|---|
| `nni.get_next_parameter() -> dict` | Receive one hyperparameter set from the tuner. Call **once** per trial. |
| `nni.get_next_parameters() -> dict` | Alias of `get_next_parameter`. |
| `nni.get_current_parameter(tag=None)` | Re-fetch the parameter dict already received, or a single key. |
| `nni.report_intermediate_result(metric)` | Report a float (or `{'default': float, ...}`) during training. |
| `nni.report_final_result(metric)` | Report final metric. **Must be called** before the script exits. |
| `nni.get_experiment_id() -> str` | Return the experiment ID. |
| `nni.get_trial_id() -> str` | Return the trial's unique ID. |
| `nni.get_sequence_id() -> int` | Return the trial sequence number (Trial No. in web UI). |

**Intermediate result dict format:**

```python
nni.report_intermediate_result({'default': 0.91, 'loss': 0.33, 'accuracy': 0.91})
# 'default' is used by the tuner/assessor; other keys are shown on the web portal.
```

---

## 4. HPO — Experiment API

```python
from nni.experiment import Experiment, ExperimentConfig
from nni.experiment.config import AlgorithmConfig, LocalConfig
```

### ExperimentConfig

| Field | Type | Description |
|---|---|---|
| `experiment_name` | `str` | Optional human-readable name. |
| `search_space` | `dict` | Search space dict (see §5). |
| `search_space_file` | `Path` | Path to a JSON file with search space (alternative to `search_space`). |
| `trial_command` | `str` | Shell command to run each trial (e.g. `"python trial.py"`). |
| `trial_code_directory` | `Path` | Directory containing trial code. Default `'.'`. |
| `trial_concurrency` | `int` | Max trials running simultaneously. **Required.** |
| `trial_gpu_number` | `int` | GPUs per trial. |
| `max_trial_number` | `int` | Stop after this many trials. |
| `max_experiment_duration` | `str` or `int` | E.g. `'2h'`, `'30m'`, or seconds as int. |
| `max_trial_duration` | `str` or `int` | Kill trials running longer than this. |
| `tuner` | `AlgorithmConfig` | Tuner configuration (see §6). |
| `assessor` | `AlgorithmConfig` | Assessor configuration (optional; for early stopping). |
| `advisor` | `AlgorithmConfig` | Advisor configuration (replaces tuner+assessor for BOHB, Hyperband). |
| `training_service` | `TrainingServiceConfig` | Where to run trials (see below). |
| `debug` | `bool` | Enable debug logging. |
| `log_level` | `str` | `'trace'`, `'debug'`, `'info'`, `'warning'`, `'error'`, `'fatal'`. |
| `experiment_working_directory` | `Path` | Where to store experiment data. Default `~/nni-experiments`. |
| `use_annotation` | `bool` | Use NNI annotation syntax (legacy). |

### AlgorithmConfig

```python
AlgorithmConfig(name='TPE', class_args={'optimize_mode': 'maximize'})
```

| Field | Description |
|---|---|
| `name` | Built-in algorithm name (e.g. `'TPE'`, `'Random'`, `'BOHB'`). |
| `class_name` | Fully-qualified class name for custom algorithms. |
| `code_directory` | Directory containing custom algorithm code. |
| `class_args` | Dict of keyword arguments passed to the algorithm constructor. |

### Training Service Configs

**LocalConfig** (`nni.experiment.config.LocalConfig`)

```python
from nni.experiment.config import LocalConfig
ts = LocalConfig(use_active_gpu=True, max_trial_number_per_gpu=2)
```

| Field | Default | Description |
|---|---|---|
| `use_active_gpu` | `None` | Allow trials on GPUs already in use. |
| `max_trial_number_per_gpu` | `1` | Max concurrent trials per GPU. |
| `gpu_indices` | `None` | List of GPU indices to use (e.g. `[0, 1]`). |
| `reuse_mode` | `False` | Reuse trial processes (faster startup). |

**RemoteConfig** (`nni.experiment.config.RemoteConfig`)

```python
from nni.experiment.config import RemoteConfig, RemoteMachineConfig
ts = RemoteConfig(
    machine_list=[RemoteMachineConfig(host='gpu1.example.com', user='alice')]
)
```

`RemoteMachineConfig` fields: `host`, `user`, `port` (22), `password`, `ssh_key_file`, `gpu_indices`, `python_path`.

**Other training services:** `AmlConfig` (Azure ML), `OpenpaiConfig`, `KubeflowConfig`, `FrameworkControllerConfig`, `DlcConfig`.

### Experiment class

```python
exp = Experiment('local')               # shortcut: pass platform name string
exp = Experiment(config)                # or pass full ExperimentConfig

exp.config.search_space = {...}
exp.config.tuner = AlgorithmConfig(name='TPE')
exp.config.training_service.use_active_gpu = True
```

| Method | Description |
|---|---|
| `exp.start(port=8080, debug=False, run_mode=RunMode.Background)` | Start NNI manager. Non-blocking. Web UI at `http://localhost:{port}`. |
| `exp.run(port=8080, wait_completion=True, debug=False)` | Start and block until all trials finish. Returns `True` on success. |
| `exp.run_or_resume(...)` | Resume from checkpoint if one exists, otherwise run fresh. |
| `exp.stop()` | Stop the experiment. |
| `exp.resume(...)` | Resume a stopped experiment. |
| `exp.view(port)` | Reattach the web UI to an already-running experiment. |
| `exp.get_status() -> str` | `'INITIALIZED'`, `'RUNNING'`, `'DONE'`, `'ERROR'`, `'STOPPED'`, etc. |
| `exp.list_trial_jobs() -> list[TrialJob]` | List all trial jobs. |
| `exp.export_data() -> list[TrialResult]` | Export all trial results. |
| `exp.update_trial_concurrency(n)` | Live update concurrency. |
| `exp.update_max_trial_number(n)` | Live update max trial count. |
| `exp.update_search_space(space)` | Live update search space. |
| `exp.kill_trial_job(trial_id)` | Kill a specific trial. |
| `Experiment.connect(port)` | Class method — attach to a running experiment by port. |

### RunMode

```python
from nni.experiment import RunMode
RunMode.Background  # NNI manager stops when Python exits; logs suppressed (default)
RunMode.Foreground  # NNI manager stops when Python exits; logs printed to stdout
RunMode.Detach      # NNI manager keeps running after Python exits
```

### Minimal HPO example

```python
import nni
from nni.experiment import Experiment

search_space = {
    'lr':      {'_type': 'loguniform', '_value': [1e-4, 1e-1]},
    'dropout': {'_type': 'uniform',    '_value': [0.0, 0.5]},
    'hidden':  {'_type': 'choice',     '_value': [64, 128, 256]},
}

exp = Experiment('local')
exp.config.experiment_name   = 'my_hpo'
exp.config.search_space       = search_space
exp.config.trial_command      = 'python trial.py'
exp.config.trial_concurrency  = 4
exp.config.max_trial_number   = 50
exp.config.tuner              = nni.experiment.config.AlgorithmConfig(
    name='TPE', class_args={'optimize_mode': 'maximize'}
)
exp.run(port=8080)
```

---

## 5. HPO — Search Space Format

Search spaces are JSON dicts. Each key is a hyperparameter name.

```json
{
  "param_name": {"_type": "<type>", "_value": <value>}
}
```

| `_type` | `_value` | Description |
|---|---|---|
| `"choice"` | `[v1, v2, ...]` | Pick one value from the list. |
| `"randint"` | `[lower, upper]` | Random integer in `[lower, upper)`. |
| `"uniform"` | `[low, high]` | Uniform float. |
| `"quniform"` | `[low, high, q]` | Quantised uniform; value is `round(U(low,high)/q)*q`. |
| `"loguniform"` | `[low, high]` | Log-uniform float. |
| `"qloguniform"` | `[low, high, q]` | Quantised log-uniform. |
| `"normal"` | `[mu, sigma]` | Normal distribution. |
| `"qnormal"` | `[mu, sigma, q]` | Quantised normal. |

---

## 6. HPO — Built-in Tuners & Assessors

### Tuners

| Name (for `AlgorithmConfig.name`) | Description | Key `class_args` |
|---|---|---|
| `TPE` | Tree-structured Parzen Estimator. Generally a good default. | `optimize_mode` (`'maximize'`/`'minimize'`), `seed` |
| `Random` | Random search. | `optimize_mode`, `seed` |
| `GridSearch` | Exhaustive grid search. | — |
| `Evolution` | Naive evolutionary algorithm. | `optimize_mode`, `population_size` |
| `SMAC` | Bayesian with SMAC3 (needs `pip install nni[SMAC]`). | `optimize_mode` |
| `Anneal` | Simulated annealing. | `optimize_mode` |
| `Metis` | Gaussian Process with ensemble; tuned for latency-sensitive tasks. | `optimize_mode` |
| `DNGO` | Deep Network for Global Optimization. | `optimize_mode` |
| `GP` | Gaussian Process Tuner. | `optimize_mode` |
| `PBT` | Population Based Training (requires `reuse_mode=True`). | `optimize_mode`, `all_checkpoint_dir` |

### Advisors (replace both tuner and assessor)

| Name | Description |
|---|---|
| `BOHB` | Bayesian optimization + Hyperband. Set via `config.advisor`. |
| `Hyperband` | Classic Hyperband. |

```python
exp.config.advisor = AlgorithmConfig(
    name='BOHB',
    class_args={'optimize_mode': 'maximize', 'min_budget': 1, 'max_budget': 27, 'eta': 3}
)
```

### Assessors (early stopping)

| Name | Description |
|---|---|
| `Medianstop` | Stop underperforming trials vs. running median. |
| `Curvefitting` | Predict final performance; stop if below threshold. |

```python
exp.config.assessor = AlgorithmConfig(
    name='Medianstop', class_args={'optimize_mode': 'maximize', 'start_step': 5}
)
```

---

## 7. NAS — Mutable Primitives

```python
from nni.mutable import Mutable, LabeledMutable, Categorical, CategoricalMultiple, Numerical, MutableExpression
```

### Core classes

| Class | Description |
|---|---|
| `Categorical(values, label)` | Choose one value from a finite list. Equivalent to `nni.choice`. |
| `CategoricalMultiple(values, label, n_chosen)` | Choose `n_chosen` values from the list. |
| `Numerical(low, high, label, ...)` | A continuous or integer numerical range. |
| `MutableExpression` | Symbolic arithmetic on `Mutable` objects (e.g. `a * b + c`). |

```python
from nni.mutable import Categorical
kernel = Categorical([3, 5, 7], label='kernel_size')
```

### Freezing (materialising a sample)

```python
sample = {'kernel_size': 5, 'depth': 3}
frozen_model = model_space.freeze(sample)   # returns a plain nn.Module
```

```python
from nni.mutable import frozen_context
with frozen_context(sample):
    model = MyModelSpace()   # all mutables are immediately resolved
```

---

## 8. NAS — Model Space Construction

```python
import nni.nas.nn.pytorch as nn
# OR the standard import pattern:
from nni.nas.nn.pytorch import (
    ModelSpace, ParametrizedModule, MutableModule,
    LayerChoice, InputChoice, Repeat, Cell,
)
```

`nni.nas.nn.pytorch` **re-exports the entire `torch.nn` namespace as mutable-aware wrappers.**
Every `torch.nn` module (Conv2d, Linear, BatchNorm2d, …) is available through this import and behaves
identically but cooperates with NNI's mutation graph.

### ModelSpace

All model search spaces must inherit `ModelSpace`.

```python
class MySpace(ModelSpace):
    def __init__(self):
        super().__init__()
        self.conv = nn.LayerChoice([
            nn.Conv2d(3, 16, 3, padding=1),
            nn.Conv2d(3, 16, 5, padding=2),
        ], label='conv_choice')

    def forward(self, x):
        return self.conv(x)
```

### LayerChoice

Select one `nn.Module` from a list of candidates.

```python
layer = nn.LayerChoice(
    candidates=[nn.ReLU(), nn.Tanh(), nn.GELU()],
    label='activation',       # string identifier; auto-generated if omitted
    weights=[0.5, 0.3, 0.2]  # optional prior for random sampling
)
```

| Key attribute/method | Description |
|---|---|
| `layer.names` | List of candidate names. |
| `len(layer)` | Number of candidates. |
| `list(layer)` | All candidate modules. |
| `layer.freeze(sample)` | Return the selected candidate as a plain `nn.Module`. |

### InputChoice

Choose a subset of input tensors to combine.

```python
self.skip = nn.InputChoice(
    n_candidates=3,       # total tensors offered
    n_chosen=1,           # how many to pick
    label='skip_choice'
)

# In forward():
out = self.skip([tensor_a, tensor_b, tensor_c])
```

### Repeat

Repeat a block a variable number of times.

```python
# Fixed repeat
self.blocks = nn.Repeat(ResBlock(64), 3)

# Variable depth (1–4 times)
self.blocks = nn.Repeat(ResBlock(64), (1, 4))

# Independent LayerChoices per depth
self.blocks = nn.Repeat(
    lambda i: nn.LayerChoice([ConvA(64), ConvB(64)], label=f'block{i}'),
    depth=(1, 4)
)
```

### Cell

A directed acyclic graph cell, common in classic NAS (NASNet-style).

```python
cell = nn.Cell(
    op_candidates=[nn.Conv2d(...), nn.MaxPool2d(...)],
    num_nodes=4,
    num_ops_per_node=2,
    num_predecessors=2,
    merge_op='all',   # 'all' or 'loose_end'
    label='cell'
)
```

### ParametrizedModule

Modules whose *hyperparameters* (not architecture) are searchable. Value choices inside
`__init__` automatically bind to the model's search space.

```python
class FlexConv(ParametrizedModule):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        # kernel_size can be a Categorical
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(x)

# Usage inside ModelSpace:
self.conv = FlexConv(3, 32, nni.choice('ks', [3, 5, 7]))
```

### freeze() and simplify()

```python
sample = model_space.simplify()      # → dict of all label→candidates
frozen = model_space.freeze(sample)  # → plain nn.Module
```

---

## 9. NAS — Vision Building Blocks

```python
from nni.nas.nn.pytorch import MutablePatchEmbedding
from nni.nas.oneshot.pytorch.supermodule.operation import MixedDepthwiseConv2d
```

### MutablePatchEmbedding

Mutable patch projection stem for Vision Transformers (ViTs). Both `patch_size` and `embed_dim`
are searchable `Categorical` choices.

```python
import nni
from nni.nas.nn.pytorch import MutablePatchEmbedding, ModelSpace

class MyViT(ModelSpace):
    def __init__(self):
        super().__init__()
        self.patch_embed = MutablePatchEmbedding(
            img_size=224,
            patch_size=nni.choice('patch_size', [8, 16, 32]),
            in_channels=3,
            embed_dim=nni.choice('embed_dim', [384, 512, 768]),
        )
        # ... transformer encoder blocks ...

    def forward(self, x):
        tokens = self.patch_embed(x)   # (B, N, embed_dim)
        return tokens
```

| Constructor arg | Description |
|---|---|
| `img_size` | Square input image size (e.g. `224`). Every candidate `patch_size` must divide it evenly. |
| `patch_size` | `Categorical` or plain `int`. |
| `in_channels` | Input channels (typically `3`). |
| `embed_dim` | `Categorical` or plain `int`. |

| Method | Description |
|---|---|
| `seq_length()` | Returns `(img_size // current_patch_size) ** 2` — the number of token positions. |

**Constraint:** All candidate `patch_size` values must exactly divide `img_size`. Violated candidates
raise `ValueError` at construction time.

### MixedDepthwiseConv2d

Weight-sharing depthwise convolution with searchable `kernel_size`. Automatically sets
`groups == in_channels` (the depthwise constraint) and computes `padding = kernel_size // 2`.

```python
import nni
from nni.nas.oneshot.pytorch.supermodule.operation import MixedDepthwiseConv2d

class MobileBlock(ModelSpace):
    def __init__(self, c):
        super().__init__()
        self.dw = MixedDepthwiseConv2d(
            in_channels=c, out_channels=c,
            kernel_size=nni.choice('dw_ks', [3, 5, 7])
        )
        self.pw = nn.Conv2d(c, c * 4, 1)

    def forward(self, x):
        return self.pw(self.dw(x))
```

`MixedDepthwiseConv2d` is registered in `NATIVE_MIXED_OPERATIONS` and recognized automatically
by all one-shot strategies.

---

## 10. NAS — Evaluators

Evaluators define **how a candidate model is trained and scored**.

### Lightning (primary evaluator)

```python
from nni.nas.evaluator.pytorch import Classification, Regression, Lightning
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

evaluator = Classification(
    train_dataloaders=DataLoader(CIFAR10('.', train=True,  transform=T.ToTensor(), download=True), batch_size=64),
    val_dataloaders  =DataLoader(CIFAR10('.', train=False, transform=T.ToTensor()),                 batch_size=64),
    max_epochs=10,
    num_classes=10,
    export_onnx=False,
)
```

| Class | Description |
|---|---|
| `Classification(train_dataloaders, val_dataloaders, max_epochs, num_classes, ...)` | Image/text classification. Uses cross-entropy loss and top-k accuracy. |
| `Regression(train_dataloaders, val_dataloaders, max_epochs, ...)` | Regression tasks. Uses MSE loss. |
| `Lightning(lightning_module, trainer, ...)` | Full control — provide your own `pl.LightningModule` and `pl.Trainer`. |

### FunctionalEvaluator

For arbitrary Python training functions (no PyTorch Lightning required).

```python
from nni.nas.evaluator import FunctionalEvaluator

def train_and_eval(model):
    # train model, return scalar metric
    return accuracy

evaluator = FunctionalEvaluator(train_and_eval)
```

---

## 11. NAS — Exploration Strategies

Strategies decide which architectures to explore.

```python
from nni.nas.strategy import (
    Random, GridSearch,
    RegularizedEvolution,
    TPE, PolicyBasedRL,
)
```

### Multi-trial strategies

| Class | Description | Key args |
|---|---|---|
| `Random(seed=None)` | Random sampling. | `seed` |
| `GridSearch()` | Exhaustive grid search over discrete choices. | — |
| `RegularizedEvolution(population_size, sample_size, cycles, ...)` | Aging-evolution algorithm. | `optimize_mode` |
| `TPE(optimize_mode, seed)` | Tree-structured Parzen Estimator adapted for NAS. | `optimize_mode` |
| `PolicyBasedRL(...)` | REINFORCE-based RL strategy. | `optimize_mode` |

```python
strategy = RegularizedEvolution(
    population_size=50, sample_size=25, cycles=200,
    optimize_mode='maximize'
)
```

---

## 12. NAS — Strategy Middleware

Wrap any strategy in middleware to add filtering, deduplication, failure handling, etc.

```python
from nni.nas.strategy.middleware import (
    Chain, Filter, Deduplication, FailureHandler,
    MultipleEvaluation, MedianStop
)
```

| Class | Description |
|---|---|
| `Chain(strategy, *middlewares)` | Compose a strategy with one or more middlewares. |
| `Filter(filter_fn, on_failure='skip')` | Discard or penalise architectures failing `filter_fn(sample) -> bool`. |
| `Deduplication()` | Skip architectures already evaluated. |
| `FailureHandler(max_failures)` | Retry on trial failure. |
| `MultipleEvaluation(n)` | Evaluate each architecture `n` times and average. |
| `MedianStop(start_step, mode)` | Early-stop multi-trial architectures below running median. |

```python
from nni.nas.strategy import Random
from nni.nas.strategy.middleware import Chain, Filter, Deduplication
from nni.nas.profiler.pytorch.flops import FlopsProfiler

profiler = FlopsProfiler(model_space, torch.randn(1, 3, 224, 224))

strategy = Chain(
    Random(),
    Deduplication(),
    Filter(lambda sample: profiler.profile(sample) < 300e6)
)
```

---

## 13. NAS — One-Shot Strategies (Weight Sharing)

One-shot strategies train a **supernet** (all architectures share weights) and extract
the best sub-architecture without retraining.

```python
from nni.nas.strategy import (
    DARTS, GumbelDARTS,
    Proxyless,
    ENAS,
    RandomOneShot,
)
```

| Class | Type | Description |
|---|---|---|
| `DARTS(gradient_clip=5.0)` | Differentiable | Classic DARTS via continuous architecture parameters. |
| `GumbelDARTS(...)` | Differentiable | DARTS with Gumbel-Softmax relaxation. |
| `Proxyless(...)` | Mixed | ProxylessNAS: binary gates reduce memory. |
| `ENAS(ctrl_lr=3.5e-4, ctrl_steps=50)` | Sampling | Efficient NAS with controller. |
| `RandomOneShot()` | Sampling | Single-path sampling (SNAS / Few-Shot NAS baseline). |

### One-shot profiler penalty

Add a hardware-constraint loss to one-shot training:

```python
from nni.nas.oneshot.pytorch.profiler import ExpectationProfilerPenalty, SampleProfilerPenalty, RangeProfilerFilter

# Penalise FLOPs > 300M during training
from nni.nas.profiler.pytorch.flops import FlopsProfiler
profiler = FlopsProfiler(model_space, dummy_input)
penalty  = ExpectationProfilerPenalty(profiler, baseline=300e6)  # baseline MUST be non-zero
strategy = DARTS(penalty=penalty)

# Or filter (sampling-based):
filter_  = RangeProfilerFilter(profiler, low=200e6, high=300e6)
strategy = ENAS(filter=filter_)
```

| Class | Description |
|---|---|
| `ExpectationProfilerPenalty(profiler, baseline)` | For differentiable strategies. `baseline` must be > 0. |
| `SampleProfilerPenalty(profiler, baseline)` | For sampling-based strategies. `baseline` must be > 0. |
| `RangeProfilerFilter(profiler, low, high)` | Reject any architecture outside `[low, high]`. |

---

## 14. NAS — Hardware-Aware NAS & Profilers

```python
from nni.nas.profiler.pytorch.flops import FlopsProfiler, NumParamsProfiler, count_flops_params
from nni.nas.profiler.pytorch.nn_meter import NnMeterProfiler
```

| Class | Description |
|---|---|
| `FlopsProfiler(model_space, dummy_input)` | Computes FLOPs for a sampled architecture. |
| `NumParamsProfiler(model_space, dummy_input)` | Computes parameter count. |
| `NnMeterProfiler(model_space, dummy_input, backend)` | Latency prediction via nn-Meter (requires `nn-meter` package). |

```python
profiler = FlopsProfiler(model_space, torch.randn(1, 3, 224, 224))
flops    = profiler.profile({'conv_choice': 0, 'depth': 2})  # returns float (FLOPs)
```

---

## 15. NAS — NasExperiment

```python
from nni.nas.experiment import NasExperiment, NasExperimentConfig
```

`NasExperiment` extends `Experiment` with NAS-specific configuration.

```python
from nni.nas.experiment import NasExperiment, NasExperimentConfig
from nni.nas.strategy import DARTS
from nni.nas.evaluator.pytorch import Classification

model_space = MyModelSpace()
evaluator   = Classification(train_dl, val_dl, max_epochs=50)
strategy    = DARTS()

config = NasExperimentConfig('local')
config.max_trial_number = 1           # one-shot: only 1 trial (the supernet)
config.trial_concurrency = 1

exp = NasExperiment(model_space, evaluator, config)
exp.run(port=8080)
```

### NasExperimentConfig — extra fields (over ExperimentConfig)

| Field | Description |
|---|---|
| `execution_engine` | Engine config: `TrainingServiceEngineConfig`, `SequentialEngineConfig`, `CgoEngineConfig`. |
| `model_format` | `GraphModelFormatConfig`, `SimplifiedModelFormatConfig`, `RawModelFormatConfig`. |

### Accessing results

```python
# After experiment ends
exported = exp.export_top_models(top_k=5, optimize_mode='maximize')
for arch, metric in exported:
    frozen_model = model_space.freeze(arch)
```

---

## 16. NAS — Model Space Hub

Pre-defined, benchmarked search spaces under `nni.nas.hub.pytorch`.

```python
from nni.nas.hub.pytorch import (
    MobileNetV3Space, ShuffleNetSpace, ProxylessNAS,
    AutoFormer, DARTS, NASNet, ENAS, AmoebaNet, PNAS,
    NasBench101, NasBench201,
)
```

| Class | Task | Notes |
|---|---|---|
| `MobileNetV3Space` | ImageNet classification | Largest space in TuNAS. |
| `ShuffleNetSpace` | ImageNet classification | Single-Path One-Shot search space. |
| `ProxylessNAS` | ImageNet classification | Based on MobileNetV2. |
| `AutoFormer` | ImageNet classification | ViT-based; elastic depth, heads, embed_dim. |
| `DARTS` | CIFAR-10 / ImageNet | Classic DARTS cell search space. |
| `NASNet` | CIFAR-10 / ImageNet | Original NASNet cell search space. |
| `ENAS` | CIFAR-10 | Efficient NAS cell space. |
| `NasBench101` | CIFAR-10 | Benchmarked space from NAS-Bench-101. |
| `NasBench201` | CIFAR-10 / ImageNet-16 | Benchmarked space from NAS-Bench-201. |

### Loading pre-searched models

```python
model = MobileNetV3Space.load_searched_model(
    'mobilenetv3-small-100',
    pretrained=True, download=True
)
```

Available aliases include: `mobilenetv3-{large,small}-{100,075,050}`, `cream-{014,043,...}`,
`proxyless-{cpu,gpu,mobile}`, `acenas-{m1,m2,m3}`, `autoformer-{tiny,small,base}`.

---

## 17. Compression — Overview & config_list Format

NNI compression wraps trained PyTorch models with pruning, quantization, or distillation.
The primary API is a `config_list` — a list of dicts describing which modules to compress and how.

### config_list structure

```python
config_list = [
    {
        # --- Target selection (at least one required) ---
        'op_types': ['Conv2d', 'Linear'],          # by module type
        'op_names': ['conv1', 'layer1.0.conv1'],   # by module name (overrides op_types)
        'exclude_op_names': ['classifier'],        # exclude specific names

        # --- Pruning keys ---
        'sparse_ratio': 0.5,          # target sparsity (0–1)
        'max_sparse_ratio': 0.9,      # optional cap
        'min_sparse_ratio': 0.1,      # optional floor
        'global_group_id': 1,         # share sparsity budget across this group
        'dependency_group_id': 2,     # enforce correlated masks across this group
        'granularity': 'out_channel', # 'default', 'fine-grained', 'out_channel', 'in_channel', [dims...]

        # --- Quantization keys ---
        'quant_dtypes': ['int8'],      # target dtype(s)
        'target_names': ['weight', '_input_', '_output_'],

        # --- Common ---
        'apply_method': 'mul',         # 'mul' (mask), 'fill' (fill zeros)
    }
]
```

Key `target_names` tokens: `'weight'`, `'bias'`, `'_input_'`, `'_output_'`.

Key `granularity` values: `'default'`, `'fine-grained'` (element-wise), `'out_channel'`, `'in_channel'`, or a list of integers specifying which dimensions to group.

---

## 18. Compression — Pruning

```python
from nni.compression.pruning import (
    LevelPruner,
    L1NormPruner, L2NormPruner, FPGMPruner,
    SlimPruner,
    TaylorPruner,
    MovementPruner,
    LinearPruner, AGPPruner,
)
```

### Pruner workflow

```python
pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()       # generate masks (model weights unchanged)
pruner.unwrap_model()              # remove wrapper modules

# Optional: actually zero/remove weights
from nni.compression.speedup import ModelSpeedup
ms = ModelSpeedup(model, dummy_input, masks)
ms.speedup_model()                 # modifies model in-place
```

### Pruner classes

| Class | Method | Key constructor args |
|---|---|---|
| `LevelPruner` | Element-wise magnitude | `model, config_list` |
| `L1NormPruner` | L1-norm channel/filter pruning | `model, config_list` |
| `L2NormPruner` | L2-norm channel/filter pruning | `model, config_list` |
| `FPGMPruner` | Geometric median filter pruning | `model, config_list` |
| `SlimPruner` | BN-scaling-factor pruning (requires pre-training with sparsity regularizer) | `model, config_list, evaluator` |
| `TaylorPruner` | Taylor expansion importance (first-order gradient) | `model, config_list, evaluator, training_steps` |
| `MovementPruner` | Soft movement pruning (fine-tuning-based) | `model, config_list, evaluator, training_steps` |
| `LinearPruner` | Iterative pruner with linear sparsity schedule | `model, config_list, pruning_algorithm, total_rounds, ...` |
| `AGPPruner` | Automated Gradual Pruning schedule | `model, config_list, pruning_algorithm, total_rounds, ...` |

### Compressor base methods

| Method | Description |
|---|---|
| `compress(max_steps, max_epochs)` | Run the compression algorithm. Returns `(model, masks)`. |
| `unwrap_model()` | Remove wrapper and restore original module references. |
| `get_masks()` | Return current masks as `{module_name: {target_name: tensor}}`. |
| `update_masks(masks)` | Manually set masks. |
| `from_compressor(compressor, new_config_list, ...)` | Class method to chain compressors. |

---

## 19. Compression — Quantization

```python
from nni.compression.quantization import (
    QATQuantizer,
    LsqQuantizer,
    LsqPlusQuantizer,
    DoReFaQuantizer,
    BNNQuantizer,
    PtqQuantizer,
)
```

### Quantizer workflow

```python
config_list = [{'op_types': ['Conv2d', 'Linear'], 'quant_dtypes': ['int8']}]

quantizer = QATQuantizer(model, config_list, evaluator=evaluator, quant_start_step=1000)
quantizer.compress(max_steps=5000, max_epochs=None)
quantizer.unwrap_model()  # remove wrappers; model now has quantization ops
```

### Quantizer classes

| Class | Method | Notes |
|---|---|---|
| `QATQuantizer` | Quantization-Aware Training | Most accurate; requires fine-tuning. |
| `LsqQuantizer` | Learned Step-Size Quantization | Learns per-channel scale factors. |
| `LsqPlusQuantizer` | LSQ+ | Extended LSQ with better initialisation. |
| `DoReFaQuantizer` | DoReFa-Net | Low-bit weights and activations. |
| `BNNQuantizer` | Binary Neural Network | 1-bit weights and activations. |
| `PtqQuantizer` | Post-Training Quantization | No fine-tuning; fast but less accurate. |

---

## 20. Compression — Distillation

```python
from nni.compression.distillation import DynamicLayerwiseDistiller, Adaptive1dLayerwiseDistiller
```

Distillation transfers knowledge from a teacher to a student during compression.

```python
config_list = [{'op_types': ['Linear'], 'lambda': 1.0, 'link': 'teacher_layer_name'}]
distiller = DynamicLayerwiseDistiller(
    model=student_model,
    config_list=config_list,
    evaluator=evaluator,
    teacher_model=teacher_model,
    teacher_predict=lambda batch, model: model(batch[0]),
)
distiller.compress(max_steps=3000)
```

---

## 21. Compression — Speedup (Mask Application)

After obtaining masks, `ModelSpeedup` rewires the network so masked channels/filters are truly removed
(reduces FLOPS and memory, not just zeroed values).

```python
from nni.compression.speedup import ModelSpeedup, auto_set_denpendency_group_ids

# Optionally auto-detect dependency groups
auto_set_denpendency_group_ids(model, dummy_input, config_list)

pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()
pruner.unwrap_model()

ms = ModelSpeedup(model, dummy_input=torch.randn(1, 3, 224, 224), masks_file=masks)
ms.speedup_model()   # model is now structurally smaller
```

---

## 22. Compression — Evaluators

Evaluators are required by gradient-based pruners (TaylorPruner, MovementPruner, SlimPruner).

```python
from nni.compression import LightningEvaluator, TorchEvaluator, TransformersEvaluator
```

| Class | When to use |
|---|---|
| `LightningEvaluator(trainer, pl_module)` | You already use PyTorch Lightning. |
| `TorchEvaluator(training_func)` | Plain PyTorch training loop. |
| `TransformersEvaluator(trainer)` | HuggingFace Transformers `Trainer`. |
| `DeepspeedTorchEvaluator(...)` | DeepSpeed training. |

```python
from nni.compression import TorchEvaluator

def train(model, optimizer, criterion, train_loader, **kwargs):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

evaluator = TorchEvaluator(
    training_func=train,
    optimizers=torch.optim.SGD(model.parameters(), lr=0.01),
    criterion=torch.nn.CrossEntropyLoss(),
    train_loader=train_loader,
)
```

---

## 23. Compression — Supported Module Types

The following `torch.nn` module types have **pre-registered default settings** in NNI 3.0.
No manual registration is needed.

### Pruning registry

| Module | Notes |
|---|---|
| `Conv1d`, `Conv2d`, `Conv3d` | Standard conv layers. |
| `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d` | Transposed convolutions. |
| `Linear` | Fully connected. |
| `Embedding` | Embedding tables. |
| `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d` | Batch normalisation. |
| `LayerNorm` | Transformer normalisation. |
| `GroupNorm` | Group normalisation (segmentation / style transfer). |
| `InstanceNorm1d`, `InstanceNorm2d` | Instance normalisation. |
| `MultiheadAttention` | `in_proj_weight`, `out_proj.weight`. Head-level granularity recommended. |

### Quantization registry (all pruning types, plus)

| Module | Notes |
|---|---|
| `GELU` | Output quantization. Default activation in BERT, ViT, GPT-2. |
| `SiLU` | Output quantization. Used in EfficientNet, YOLOv8, ViT variants. |
| `Hardswish` | Output quantization. Used in MobileNetV3. |

### Registering custom module types

```python
from nni.compression.base.setting import PruningSetting, QuantizationSetting

PruningSetting.register('MyCustomLayer', {
    'weight': {
        'sparse_ratio': None, 'granularity': 'default', 'apply_method': 'mul',
        'global_group_id': None, 'dependency_group_id': None,
        'max_sparse_ratio': None, 'min_sparse_ratio': None,
        'sparse_threshold': None, 'internal_metric_block': None,
    }
})
```

---

## 24. Serialization Utilities

NNI provides a serialization layer on top of `cloudpickle` / JSON that records object construction
arguments, enabling reproducibility and experiment configuration sharing.

```python
from nni.common.serializer import trace, dump, load, serialize, deserialize

@trace
class Optimizer:
    def __init__(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum

opt   = Optimizer(lr=0.01, momentum=0.9)
saved = dump(opt)     # JSON-serializable dict
opt2  = load(saved)   # new Optimizer(lr=0.01, momentum=0.9)
```

> **Security note:** `load()` / `deserialize()` ultimately calls `cloudpickle.loads`. Only load data
> from trusted sources.

---

## 25. Common Patterns & End-to-End Examples

### Pattern A — Simple HPO with TPE

```python
# trial.py
import nni, torch

params = nni.get_next_parameter()
model  = build_model(params['hidden_size'], params['dropout'])
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

for epoch in range(50):
    train_loss = train_one_epoch(model, optimizer, train_loader)
    val_acc    = evaluate(model, val_loader)
    nni.report_intermediate_result(val_acc)

nni.report_final_result(val_acc)
```

```python
# run_experiment.py
from nni.experiment import Experiment
from nni.experiment.config import AlgorithmConfig, LocalConfig

exp = Experiment('local')
exp.config.trial_command     = 'python trial.py'
exp.config.trial_concurrency = 4
exp.config.max_trial_number  = 100
exp.config.search_space      = {
    'lr':          {'_type': 'loguniform', '_value': [1e-4, 1e-1]},
    'hidden_size': {'_type': 'choice',     '_value': [64, 128, 256, 512]},
    'dropout':     {'_type': 'uniform',    '_value': [0.0, 0.5]},
}
exp.config.tuner = AlgorithmConfig(name='TPE', class_args={'optimize_mode': 'maximize'})
exp.run(port=8080)
best = sorted(exp.export_data(), key=lambda t: t.value, reverse=True)[0]
print('Best params:', best.parameter, 'Best acc:', best.value)
```

---

### Pattern B — Multi-trial NAS (CIFAR-10)

```python
import torch.nn as nn
import nni
import nni.nas.nn.pytorch as nas_nn
from nni.nas.nn.pytorch import ModelSpace, LayerChoice, Repeat
from nni.nas.evaluator.pytorch import Classification
from nni.nas.strategy import RegularizedEvolution
from nni.nas.experiment import NasExperiment, NasExperimentConfig
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

class MySpace(ModelSpace):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.blocks = Repeat(
            lambda i: LayerChoice([
                nn.Conv2d(16, 16, 3, padding=1),
                nn.Conv2d(16, 16, 5, padding=2),
            ], label=f'block{i}'),
            depth=(2, 5)
        )
        self.head = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x.mean([2, 3]))

transform = T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)])
evaluator = Classification(
    train_dataloaders=DataLoader(CIFAR10('.', True,  transform=transform, download=True), batch_size=128),
    val_dataloaders  =DataLoader(CIFAR10('.', False, transform=transform),                batch_size=256),
    max_epochs=20,
)
config = NasExperimentConfig('local')
config.trial_concurrency = 2
config.max_trial_number  = 30
exp = NasExperiment(MySpace(), evaluator, config, strategy=RegularizedEvolution(optimize_mode='maximize'))
exp.run(port=8080)
```

---

### Pattern C — One-shot NAS (DARTS)

```python
from nni.nas.strategy import DARTS
from nni.nas.experiment import NasExperiment, NasExperimentConfig
from nni.nas.hub.pytorch import DARTS as DARTSSpace   # pre-built cell space

space     = DARTSSpace()
evaluator = Classification(train_dl, val_dl, max_epochs=50)
strategy  = DARTS(gradient_clip=5.0)

config = NasExperimentConfig('local')
config.trial_concurrency = 1
config.max_trial_number  = 1

exp = NasExperiment(space, evaluator, config, strategy=strategy)
exp.run(port=8080)
```

---

### Pattern D — L1 Pruning + Speedup

```python
import torch
from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup

# Assume `model` is a pretrained nn.Module
config_list = [
    {'op_types': ['Conv2d'],  'sparse_ratio': 0.5},
    {'op_types': ['Linear'],  'sparse_ratio': 0.5, 'exclude_op_names': ['head']},
]

pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()
pruner.unwrap_model()

dummy = torch.randn(1, 3, 224, 224)
ms = ModelSpeedup(model, dummy, masks)
ms.speedup_model()

print(model)   # structurally smaller model
```

---

### Pattern E — QAT Quantization

```python
from nni.compression.quantization import QATQuantizer
from nni.compression import TorchEvaluator
import torch

def fine_tune(model, optimizer, criterion, train_loader, **kw):
    for x, y in train_loader:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()

evaluator = TorchEvaluator(
    training_func=fine_tune,
    optimizers=torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9),
    criterion=torch.nn.CrossEntropyLoss(),
    train_loader=train_loader,
)

config_list = [{'op_types': ['Conv2d', 'Linear'], 'quant_dtypes': ['int8']}]
quantizer   = QATQuantizer(model, config_list, evaluator, quant_start_step=500)
quantizer.compress(max_steps=3000)
quantizer.unwrap_model()
```

---

### Pattern F — ViT Search with MutablePatchEmbedding + Hardware Constraint

```python
import nni
import torch
import nni.nas.nn.pytorch as nasnn
from nni.nas.nn.pytorch import ModelSpace, MutablePatchEmbedding
from nni.nas.strategy import ENAS
from nni.nas.oneshot.pytorch.profiler import SampleProfilerPenalty
from nni.nas.profiler.pytorch.flops import FlopsProfiler
from nni.nas.experiment import NasExperiment, NasExperimentConfig
from nni.nas.evaluator.pytorch import Classification

class ViTSpace(ModelSpace):
    def __init__(self):
        super().__init__()
        self.patch_embed = MutablePatchEmbedding(
            img_size=224,
            patch_size=nni.choice('ps', [16, 32]),
            in_channels=3,
            embed_dim=nni.choice('ed', [384, 768]),
        )
        self.head = nasnn.Linear(nni.choice('ed', [384, 768]), 1000)

    def forward(self, x):
        tokens = self.patch_embed(x)         # (B, N, embed_dim)
        return self.head(tokens.mean(1))

model_space = ViTSpace()
dummy       = torch.randn(1, 3, 224, 224)
profiler    = FlopsProfiler(model_space, dummy)
penalty     = SampleProfilerPenalty(profiler, baseline=4e9)   # 4 GFLOPs target

config = NasExperimentConfig('local')
config.trial_concurrency = 1
config.max_trial_number  = 1

exp = NasExperiment(
    model_space,
    Classification(train_dl, val_dl, max_epochs=30),
    config,
    strategy=ENAS(penalty=penalty)
)
exp.run(port=8080)
```

---

## Quick-Reference Import Map

```python
# Top-level HPO & utilities
import nni
from nni.experiment import Experiment, RunMode
from nni.experiment.config import (
    ExperimentConfig, AlgorithmConfig, CustomAlgorithmConfig,
    LocalConfig, RemoteConfig, RemoteMachineConfig, AmlConfig,
)

# NAS model space
import nni.nas.nn.pytorch as nn          # drop-in for torch.nn, adds mutables
from nni.nas.nn.pytorch import (
    ModelSpace, ParametrizedModule, MutableModule,
    LayerChoice, InputChoice, Repeat, Cell,
    MutablePatchEmbedding,               # vision ViT patch embedding
)

# NAS one-shot operations
from nni.nas.oneshot.pytorch.supermodule.operation import MixedDepthwiseConv2d

# NAS evaluators
from nni.nas.evaluator.pytorch import Classification, Regression, Lightning
from nni.nas.evaluator import FunctionalEvaluator

# NAS strategies
from nni.nas.strategy import (
    Random, GridSearch, RegularizedEvolution, TPE, PolicyBasedRL,   # multi-trial
    DARTS, GumbelDARTS, Proxyless, ENAS, RandomOneShot,             # one-shot
)
from nni.nas.strategy.middleware import Chain, Filter, Deduplication

# NAS profilers
from nni.nas.profiler.pytorch.flops import FlopsProfiler, NumParamsProfiler
from nni.nas.oneshot.pytorch.profiler import (
    ExpectationProfilerPenalty, SampleProfilerPenalty, RangeProfilerFilter
)

# NAS experiment
from nni.nas.experiment import NasExperiment, NasExperimentConfig

# NAS hub
from nni.nas.hub.pytorch import (
    MobileNetV3Space, ShuffleNetSpace, ProxylessNAS,
    AutoFormer, DARTS as DARTSSpace, NasBench101, NasBench201,
)

# Compression
from nni.compression.pruning import (
    L1NormPruner, L2NormPruner, LevelPruner, FPGMPruner,
    SlimPruner, TaylorPruner, MovementPruner, LinearPruner, AGPPruner,
)
from nni.compression.quantization import (
    QATQuantizer, LsqQuantizer, LsqPlusQuantizer,
    DoReFaQuantizer, BNNQuantizer, PtqQuantizer,
)
from nni.compression.distillation import DynamicLayerwiseDistiller, Adaptive1dLayerwiseDistiller
from nni.compression.speedup import ModelSpeedup, auto_set_denpendency_group_ids
from nni.compression import LightningEvaluator, TorchEvaluator, TransformersEvaluator
from nni.compression.base.setting import PruningSetting, QuantizationSetting

# Mutables (low-level)
from nni.mutable import Categorical, CategoricalMultiple, Numerical
```
