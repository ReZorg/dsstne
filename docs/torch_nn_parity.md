# PyTorch `torch.nn` Parity Matrix

This document tracks parity between DSSTNE and [PyTorch `torch.nn`](https://pytorch.org/docs/stable/nn.html).

## Optimizers

All 7 training modes are now parseable via the JSON CDL config (`"optimizer"` key).

| CDL string    | DSSTNE `TrainingMode` | `torch.optim` equivalent |
|---------------|------------------------|--------------------------|
| `"sgd"`       | `SGD`                  | `torch.optim.SGD`        |
| `"momentum"`  | `Momentum`             | `torch.optim.SGD(momentum=…)` |
| `"adagrad"`   | `AdaGrad`              | `torch.optim.Adagrad`    |
| `"nesterov"`  | `Nesterov`             | `torch.optim.SGD(nesterov=True)` |
| `"rmsprop"`   | `RMSProp`              | `torch.optim.RMSprop`    |
| `"adadelta"`  | `AdaDelta`             | `torch.optim.Adadelta`   |
| `"adam"`      | `Adam`                 | `torch.optim.Adam`       |

## Activation Functions

| DSSTNE `Activation`          | `torch.nn` equivalent             | Status  |
|------------------------------|-----------------------------------|---------|
| `Sigmoid`                    | `torch.nn.Sigmoid`                | ✅ Full |
| `Tanh`                       | `torch.nn.Tanh`                   | ✅ Full |
| `RectifiedLinear`            | `torch.nn.ReLU`                   | ✅ Full |
| `LeakyRectifiedLinear`       | `torch.nn.LeakyReLU`              | ✅ Full |
| `ParametricRectifiedLinear`  | `torch.nn.PReLU`                  | ✅ Full (forward + backward) |
| `ExponentialLinear`          | `torch.nn.ELU`                    | ✅ Full |
| `ScaledExponentialLinear`    | `torch.nn.SELU`                   | ✅ Full |
| `SoftMax`                    | `torch.nn.Softmax`                | ✅ Full |
| `SoftPlus`                   | `torch.nn.Softplus`               | ✅ Full (forward + backward) |
| `SoftSign`                   | `torch.nn.Softsign`               | ✅ Full (forward + backward) |
| `RELUMax`                    | `torch.nn.ReLU` (Maxout context)  | ✅ Full (forward + backward) |
| `LinearMax`                  | `torch.nn.Identity` (Maxout context) | ✅ Full (forward + backward) |
| `Linear`                     | Identity / no activation          | ✅ Full |

## Pooling Functions

| CDL string                     | DSSTNE `PoolingFunction` | `torch.nn` equivalent          |
|--------------------------------|--------------------------|--------------------------------|
| `"max"`                        | `Max`                    | `torch.nn.MaxPool*`            |
| `"average"`                    | `Average`                | `torch.nn.AvgPool*`            |
| `"lrn"` / `"localresponsenormalization"` | `LRN`       | `torch.nn.LocalResponseNorm`   |
| `"maxout"`                     | `Maxout`                 | Maxout network                 |
| `"dotproduct"`                 | `DotProduct`             | Attention / dot product        |
| `"cosine"`                     | `Cosine`                 | `torch.nn.CosineSimilarity`    |
| `"stochastic"`                 | `Stochastic`             | Stochastic depth / pooling     |
| `"localcontrastnormalization"` / `"lcn"` | `LCN`       | Local contrast normalization   |
| `"globaltemporal"`             | `GlobalTemporal`         | Global temporal pooling        |

## Backward Pass Coverage

For all new activation functions (SoftPlus, SoftSign, RELUMax, LinearMax, ParametricRectifiedLinear),
backward pass kernels have been implemented for:

- Dense output delta (`kCalculateOutputDelta`, `kCalculateIndexedOutputDelta`)
- Sparse output delta (`kCalculateSparseOutputDelta`, `kCalculateIndexedSparseOutputDelta`)
- Sparse analog output delta (`kCalculateSparseAnalogOutputDelta`, `kCalculateIndexedSparseAnalogOutputDelta`)
- L2 Hinge variants (dense and sparse)
- L1 variants (dense and sparse)
- Hidden-layer Hadamard product (`kCalculateHadamardProduct`)

## Changes Made

| Phase | File | Description |
|-------|------|-------------|
| 1 | `src/amazon/dsstne/utils/cdl.cpp` | Added all 7 optimizers to `sOptimizationMap` |
| 2 | `src/amazon/dsstne/engine/NNNetwork.cpp` | Added `stochastic`, `lcn`, `globaltemporal` pooling parsers |
| 3 | `src/amazon/dsstne/engine/kActivation.cu` | Added SoftPlus, SoftSign, RELUMax forward kernels |
| 4 | `src/amazon/dsstne/engine/kernels.h` | Declared new forward activation functions |
| 5 | `src/amazon/dsstne/engine/NNLayer.cpp` | Wired new activations into `CalculateActivation` |
| 6 | `src/amazon/dsstne/engine/kDelta.cu` | Added SoftPlus, SoftSign, RELUMax, LinearMax, PReLU Hadamard kernels |
| 7 | `src/amazon/dsstne/engine/kDelta.cu` | Added output delta kernels and dispatch for all new activations |
| 8 | `tst/unittests/TestCDLParser.cpp` | Unit tests for optimizer map and enum completeness |
