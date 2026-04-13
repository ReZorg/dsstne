# Amazon DSSTNE - Technical Architecture Documentation

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Core Components](#2-core-components)
3. [Data Flow Architecture](#3-data-flow-architecture)
4. [GPU Computing Model](#4-gpu-computing-model)
5. [Language Bindings](#5-language-bindings)
6. [Configuration System](#6-configuration-system)
7. [Testing Infrastructure](#7-testing-infrastructure)
8. [Build System](#8-build-system)

---

## 1. System Overview

### 1.1 Purpose

DSSTNE (Deep Scalable Sparse Tensor Network Engine) is a high-performance deep learning library optimized for:
- **Sparse input data** common in recommendation systems
- **Multi-GPU model parallelism** for large-scale models
- **Production deployment** with emphasis on speed and scale

### 1.2 Key Differentiators

| Feature | DSSTNE | Traditional DL Libraries |
|---------|--------|-------------------------|
| Sparse Data Handling | Native GPU-optimized kernels | Often CPU fallback |
| Multi-GPU Strategy | Model-parallel by default | Data-parallel typical |
| Primary Use Case | Recommendations at scale | General purpose |
| Weight Matrix Size | Supports trillion+ parameters | Limited by single GPU |

### 1.3 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Applications                          │
├─────────────┬─────────────┬─────────────┬──────────────────────┤
│   Python    │    Java     │   CLI Tools │    C++ Direct API    │
│  Bindings   │  Bindings   │  (train,    │                      │
│             │             │   predict)  │                      │
├─────────────┴─────────────┴─────────────┴──────────────────────┤
│                    Runtime Context Layer                        │
│                    (DsstneContext.cpp)                          │
├─────────────────────────────────────────────────────────────────┤
│                      Core Engine                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  NNNetwork  │  │   NNLayer   │  │       NNWeight          │ │
│  │             │◄─┤             │◄─┤  (weight matrices)      │ │
│  │ (training,  │  │ (forward/   │  │                         │ │
│  │  predict)   │  │  backprop)  │  │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     GPU Kernel Layer                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  kernels.cu | kActivation.cu | kDelta.cu | kLoss.cu      │  │
│  │  (CUDA kernels for sparse/dense operations)              │  │
│  └──────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Dependencies                                  │
│     CUDA/cuDNN  |  cuBLAS  |  OpenMPI  |  NetCDF  |  jsoncpp   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Engine Module (`src/amazon/dsstne/engine/`)

#### NNNetwork (NNNetwork.h/cpp)

The central class managing neural network operations.

**Key Responsibilities:**
- Network topology management
- Training orchestration (Train method)
- Prediction execution (PredictBatch method)
- Weight management and checkpointing
- Multi-GPU coordination

**Public Interface:**
```cpp
class NNNetwork {
public:
    void LoadDataSets(vector<NNDataSetBase*>& vData);
    float Train(uint32_t epochs, NNFloat alpha, NNFloat lambda, ...);
    void PredictBatch(uint32_t layers = 0);
    void CalculateTopK(const string& layer, uint32_t k, ...);
    bool SaveNetCDF(const string& fname);
    NNLayer* GetLayer(const string& layer) const;
    // ... more methods
};
```

#### NNLayer (NNLayer.h/cpp)

Represents a single layer in the network.

**Layer Kinds:**
- `Input` - Data input layer
- `Hidden` - Intermediate processing layer
- `Output` - Result layer
- `Target` - Training target layer

**Layer Types:**
- `FullyConnected` - Dense connections
- `Convolutional` - (Limited support)
- `Pooling` - Pooling operations

#### NNWeight (NNWeight.h/cpp)

Manages weight matrices between layers.

**Features:**
- Weight initialization schemes (Xavier, Gaussian, Uniform, SELU)
- Gradient computation and updates
- Multi-GPU weight synchronization
- L1/L2 regularization support

#### NNDataSet (NNTypes.h)

Template class for dataset management.

**Supported Data Types:**
- `NNFloat` (float)
- `double`
- `uint32_t`, `int32_t`
- `uint64_t`, `int64_t`
- `unsigned char`, `char`

**Dataset Attributes:**
- `Sparse` - Sparse representation
- `Indexed` - Indexed data access
- `Weighted` - Weighted samples

### 2.2 Utilities Module (`src/amazon/dsstne/utils/`)

| File | Purpose |
|------|---------|
| `Train.cpp` | Training CLI tool implementation |
| `Predict.cpp` | Prediction CLI tool implementation |
| `generateNetCDF.cpp` | Data format conversion |
| `NetCDFhelper.cpp` | NetCDF file operations |
| `NNRecsGenerator.cpp` | Recommendation generation |
| `Filters.cpp` | Prediction filtering |
| `Utils.cpp` | Common utility functions |
| `cdl.cpp` | Configuration parsing |

### 2.3 KNN Module (`src/amazon/dsstne/knn/`)

GPU-accelerated K-Nearest Neighbors implementation.

| File | Purpose |
|------|---------|
| `KNearestNeighbors.h` | Main KNN interface |
| `KnnExactGpu.cu` | GPU exact KNN kernels |
| `topk.cu` | Top-K selection algorithms |
| `MathUtil.cu` | Math utility kernels |
| `KnnData.cpp` | KNN data structures |

### 2.4 Runtime Module (`src/amazon/dsstne/runtime/`)

| File | Purpose |
|------|---------|
| `DsstneContext.h/cpp` | Runtime context management |

---

## 3. Data Flow Architecture

### 3.1 Training Pipeline

```
Raw Data (TSV/CSV)
       │
       ▼
┌──────────────────┐
│  generateNetCDF  │  Convert to NetCDF format
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   NNDataSet      │  Load sparse/dense data
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   NNNetwork      │  Initialize network from JSON config
│   LoadDataSets   │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Train Loop     │  For each epoch:
│                  │    - Forward propagation
│                  │    - Calculate error
│                  │    - Backpropagation
│                  │    - Update weights
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   SaveNetCDF     │  Save trained model
└──────────────────┘
```

### 3.2 Prediction Pipeline

```
Trained Model (.nc)
       │
       ▼
┌──────────────────┐
│   LoadNetwork    │  Load model from NetCDF
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Input Data     │  Prepare input batch
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   PredictBatch   │  Forward propagation only
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   CalculateTopK  │  Get top-K predictions
└──────────────────┘
       │
       ▼
┌──────────────────┐
│   Apply Filters  │  Filter results
└──────────────────┘
       │
       ▼
Output Recommendations
```

---

## 4. GPU Computing Model

### 4.1 Memory Hierarchy

```
┌─────────────────────────────────────┐
│           Host Memory               │
│  (CPU RAM - Dataset loading)        │
└────────────────┬────────────────────┘
                 │ PCIe Transfer
                 ▼
┌─────────────────────────────────────┐
│           GPU Global Memory         │
│  - Weight matrices                  │
│  - Activations                      │
│  - Gradients                        │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│           GPU Shared Memory         │
│  (Fast on-chip cache for kernels)   │
└─────────────────────────────────────┘
```

### 4.2 Multi-GPU Model Parallelism

DSSTNE distributes layers across GPUs:

```
GPU 0                GPU 1                GPU 2
┌─────────┐         ┌─────────┐         ┌─────────┐
│Layer 0  │         │Layer 0  │         │Layer 0  │
│Slice 0  │←───────▶│Slice 1  │←───────▶│Slice 2  │
└─────────┘         └─────────┘         └─────────┘
     │                   │                   │
     ▼                   ▼                   ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│Layer 1  │         │Layer 1  │         │Layer 1  │
│Slice 0  │←───────▶│Slice 1  │←───────▶│Slice 2  │
└─────────┘         └─────────┘         └─────────┘
```

### 4.3 Key CUDA Kernels

| Kernel | File | Purpose |
|--------|------|---------|
| `kScaleAndBias` | kernels.cu | Weight scaling |
| `kClearUnit` | kernels.cu | Unit initialization |
| `kCalculateSigmoid` | kActivation.cu | Sigmoid activation |
| `kCalculateRELU` | kActivation.cu | ReLU activation |
| `kCalculateSoftMax` | kActivation.cu | SoftMax activation |
| `kCalculateL2Error` | kLoss.cu | L2 loss computation |
| `kCalculateCrossEntropy` | kLoss.cu | Cross-entropy loss |
| `kCalculateSigmoidDelta` | kDelta.cu | Sigmoid gradient |

---

## 5. Language Bindings

### 5.1 Python Bindings (`python/`)

**Module Structure:**
```
python/
├── dsstnemodule.cc      # C extension main
├── dsstnecalculate.h    # Calculation functions
├── NNNetworkAccessors.h # Network property access
├── NNLayerAccessors.h   # Layer property access
├── NNWeightAccessors.h  # Weight property access
├── CDLAccessors.h       # Configuration access
├── setup.py             # Build configuration
└── encoder/             # Data encoding utilities
```

**Key Functions Exposed:**
- Network loading and prediction
- Configuration management
- Dataset operations
- Top-K calculations

### 5.2 Java Bindings (`java/`)

**Package Structure:**
```
java/src/main/java/com/amazon/dsstne/
├── Dsstne.java          # Main API
├── NNNetwork.java       # Network wrapper
├── NNLayer.java         # Layer wrapper
├── NetworkConfig.java   # Configuration
├── TopKOutput.java      # Prediction results
├── data/                # Dataset classes
└── knn/                 # KNN interface
```

**JNI Implementation:**
- `java/src/main/native/` - Native C++ JNI code
- `java/src/main/native_knn/` - KNN JNI code

---

## 6. Configuration System

### 6.1 Network Definition Language (JSON)

```json
{
    "Version": 0.8,
    "Name": "Example Network",
    "Kind": "FeedForward",
    "ShuffleIndices": true,
    
    "ScaledMarginalCrossEntropy": {
        "oneTarget": 1.0,
        "zeroTarget": 0.0,
        "oneScale": 1.0,
        "zeroScale": 1.0
    },
    
    "Layers": [
        {
            "Name": "Input",
            "Kind": "Input",
            "N": "auto",
            "DataSet": "input",
            "Sparse": true
        },
        {
            "Name": "Hidden1",
            "Kind": "Hidden",
            "Type": "FullyConnected",
            "Source": "Input",
            "N": 1024,
            "Activation": "Relu",
            "pDropout": 0.5,
            "WeightInit": {"Scheme": "Gaussian", "Scale": 0.01}
        },
        {
            "Name": "Output",
            "Kind": "Output",
            "Type": "FullyConnected",
            "DataSet": "output",
            "N": "auto",
            "Activation": "Sigmoid",
            "Sparse": true
        }
    ],
    
    "ErrorFunction": "ScaledMarginalCrossEntropy"
}
```

### 6.2 Configuration Parameters

**Network-Level:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `Version` | float | Config version |
| `Name` | string | Network name |
| `Kind` | enum | FeedForward, AutoEncoder |
| `ErrorFunction` | enum | Loss function type |
| `ShuffleIndices` | bool | Shuffle training data |

**Layer-Level:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `Name` | string | Layer identifier |
| `Kind` | enum | Input, Hidden, Output |
| `Type` | enum | FullyConnected, etc. |
| `N` | int/"auto" | Layer size |
| `Activation` | enum | Activation function |
| `pDropout` | float | Dropout probability |
| `WeightInit` | object | Weight initialization |

---

## 7. Testing Infrastructure

### 7.1 Test Organization

```
tst/
├── unittests/              # CPU-only unit tests
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── TestNetCDFhelper.cpp
│   ├── TestUtils.cpp
│   ├── TestUtilsComprehensive.cpp
│   ├── TestNetCDFhelperExtended.cpp
│   └── TestDataTypes.cpp
├── gputests/               # GPU-required tests
│   ├── CMakeLists.txt
│   ├── TestActivationFunctions.cpp
│   ├── TestCostFunctions.cpp
│   └── TestSort.cpp
├── amazon/dsstne/engine/   # Engine-specific tests
│   ├── TestNNDataSet.cpp
│   ├── TestGpuBuffer.cpp
│   └── TestNNDataSetDimensions.cpp
└── test_data/              # Test fixtures
```

### 7.2 Test Framework

- **CppUnit** for C++ tests
- **JUnit** for Java tests
- **pytest** (recommended) for Python tests

### 7.3 Running Tests

```bash
# CPU unit tests
cd tst/unittests && mkdir build && cd build
cmake .. && make && ./unittests

# GPU tests (requires CUDA)
cd tst && make && make run-tests
```

---

## 8. Build System

### 8.1 Build Structure

```
Makefile (root)
├── src/amazon/dsstne/Makefile
│   ├── engine/Makefile
│   ├── utils/Makefile
│   ├── runtime/Makefile
│   └── knn/Makefile
├── tst/Makefile
├── python/Makefile
└── java/Makefile
```

### 8.2 Build Targets

| Target | Command | Description |
|--------|---------|-------------|
| All | `make` | Build everything |
| Engine | `make -C src/amazon/dsstne/engine` | Core library |
| Utils | `make -C src/amazon/dsstne/utils` | CLI tools |
| Tests | `make -C tst` | Build tests |
| Install | `make install` | Install to PREFIX |
| Clean | `make clean` | Remove build artifacts |

### 8.3 Build Output

```
build/
├── lib/
│   ├── libdsstne.a         # Static engine library
│   └── libdsstne_utils.so  # Shared utils library
├── bin/
│   ├── train               # Training CLI
│   ├── predict             # Prediction CLI
│   └── generateNetCDF      # Data conversion tool
├── include/                # Header files
└── tst/bin/                # Test binaries
```

### 8.4 Dependencies Matrix

| Component | CUDA | cuDNN | OpenMPI | NetCDF | jsoncpp | CppUnit |
|-----------|------|-------|---------|--------|---------|---------|
| Engine | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| Utils | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| KNN | ✓ | - | - | - | - | - |
| Tests | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Python | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| Java | ✓ | ✓ | ✓ | - | - | - |

---

## Appendix A: Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.9 | 2016 | Initial release |
| Current | 2024 | Multi-GPU, Python/Java bindings |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Sparse Data | Data where most values are zero |
| Model Parallelism | Distributing model layers across GPUs |
| Data Parallelism | Distributing data batches across GPUs |
| NetCDF | Network Common Data Format |
| cuDNN | NVIDIA Deep Neural Network library |
| MPI | Message Passing Interface |

---

*Document Version: 1.0*
*Last Updated: April 2026*
