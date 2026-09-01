# Amazon DSSTNE - Deep Integration Implementation Plan

> **Status Update (see "Implementation Status" below):** The majority of the
> features described in this plan and in issue #16 have been implemented and are
> covered by the C++ CPU unit-test suite and the Python test suite. A small
> number of items that require GPU / multi-GPU hardware to implement and
> validate remain open.

## Implementation Status

| Area | Status | Notes |
|------|--------|-------|
| Centered derivative (`NNNetwork.cpp`) | ✅ Done | Centered difference formula, error `O(delta^2)` |
| Explicit bias gradient (`_pbBiasGradient`) | ✅ Done | Buffer in `NNWeight.h`, used by `NNNetwork.cpp` |
| Multi-dimensional topK (`DsstneContext`) | ✅ Done | N-D output flattened to 1-D buffer |
| Filter >10.0 hack + vector pre-allocation | ✅ Done | See `Filters.cpp` |
| Streaming data loader | ✅ Done | `utils/StreamingDataLoader.*` + GPU test |
| Parallelism detector | ✅ Done | `engine/ParallelismDetector.*` |
| Embedding → KNN pipeline | ✅ Done | `pipeline/EmbeddingExtractor.*`, `NNKnnPipeline.*` |
| Shared GPU utilities | ✅ Done | `common/GpuCommon.h` |
| Unified configuration | ✅ Done | `config/DsstneConfig.{h,cpp}` — JSON/YAML load/save, env overrides, CLI parsing, merge, validation; covered by `TestDsstneConfig` (12 tests) |
| Centralized error / logging | ✅ Done | `common/DsstneError.h`, `common/Logger.h` |
| GitHub Actions CI (CPU) | ✅ Done | `.github/workflows/ci.yml` (no GPU required) |
| GPU/vGPU CI workflows | ✅ Done | `.github/workflows/gpu-tests.yml` (needs self-hosted GPU runner) |
| Python package + tests | ✅ Done | `python/dsstne/`, `python/tests/` |
| Java builders / async API | ✅ Done | `NetworkConfigBuilder.java`, `AsyncDsstne.java` |
| KNN reuses engine bitonic sorter | ✅ Done | `knn/topk.cu` includes `engine/bitonic.h` |
| Data-parallel weight distribution (`NNWeight.cpp`) | ⬜ Open | Requires multi-GPU hardware to validate |
| Multi-GPU Node Filter (`NNRecsGenerator.cpp`) | ⬜ Open | Requires multi-GPU hardware to validate |
| KNN/engine topK full dedup | ⬜ Open | KNN variant is intentionally padding-aware; not drop-in |

The sections below are the original planning document and are retained for
historical context and for the remaining open items.

## Executive Summary

This document outlines the implementation plan for the next phase of development focusing on deep integration of standalone features in the Amazon DSSTNE (Deep Scalable Sparse Tensor Network Engine) library. The plan addresses identified gaps, technical debt, and opportunities for enhancement.

### Key Decisions

| Decision | Status | Notes |
|----------|--------|-------|
| **GPU Environment** | Virtual GPUs | Testing will use vGPU infrastructure |
| **Priority Focus** | Core Inference & Training | Perfect these first, then extensions |
| **Breaking Changes** | Acceptable | Original external connections likely defunct |
| **Timeline** | 24 weeks | Approved |
| **API Docs** | At discretion | Doxygen/Sphinx/Javadoc as needed |

---

## 1. Current State Analysis

### 1.1 Architecture Overview

DSSTNE consists of the following major standalone components:

| Component | Location | Description | Integration Status |
|-----------|----------|-------------|-------------------|
| **Engine Core** | `src/amazon/dsstne/engine/` | Neural network core (NNNetwork, NNLayer, NNWeight) | Integrated |
| **Utilities** | `src/amazon/dsstne/utils/` | CLI tools (train, predict, generateNetCDF) | Standalone |
| **Runtime Context** | `src/amazon/dsstne/runtime/` | Runtime management (DsstneContext) | Partial |
| **KNN Module** | `src/amazon/dsstne/knn/` | K-Nearest Neighbors GPU implementation | Standalone |
| **Python Bindings** | `python/` | Python C extension and utilities | Standalone |
| **Java Bindings** | `java/` | JNI-based Java API | Standalone |

### 1.2 Identified Technical Debt

Based on codebase analysis, the following TODO/FIXME items need attention:

1. **NNNetwork.cpp:2473** - Non-centered derivative formula implementation
2. **NNNetwork.cpp:2563** - Explicit bias gradient hack
3. **NNWeight.cpp:841,1064** - Data-parallel detection and weights
4. **DsstneContext.cpp:48** - topK only supports 1-D outputs
5. **Filters.cpp:162,166** - Hack for values >10.0 and vector resizing
6. **NNRecsGenerator.cpp:136,148** - Time wrapper and Node Filter multi-GPU support
7. **Predict.cpp:198-199** - NetCDF regeneration inefficiency

---

## 2. Integration Phases (Reprioritized)

> **PRIORITY**: Core inference & training functionality must be perfected first.
> Parallel execution and multi-dimensional extensions come after core is stable.

### Phase 1: Core Inference & Training Tests (Weeks 1-6) ⭐ HIGHEST PRIORITY

#### 2.1.1 NNNetwork Core Test Suite

**Objective:** Create comprehensive tests for core neural network inference and training.

**Tasks:**
- [ ] Test network loading from NetCDF files
- [ ] Test forward propagation (PredictBatch)
- [ ] Test training loop (Train method)
- [ ] Test weight serialization/deserialization
- [ ] Test checkpoint save/restore
- [ ] Test all supported activation functions
- [ ] Test all supported loss functions
- [ ] Test all supported optimizers (SGD, Adam, etc.)

**Files to Create:**
- `tst/gputests/TestNNNetworkInference.cpp`
- `tst/gputests/TestNNNetworkTraining.cpp`
- `tst/gputests/TestNNNetworkCheckpoints.cpp`

#### 2.1.2 NNLayer Core Test Suite

**Objective:** Comprehensive testing of layer operations.

**Tasks:**
- [ ] Test layer initialization for all layer kinds (Input, Hidden, Output)
- [ ] Test forward propagation per layer type
- [ ] Test backpropagation per layer type
- [ ] Test dropout behavior
- [ ] Test batch normalization (if applicable)
- [ ] Test sparse vs dense layer operations

**Files to Create:**
- `tst/gputests/TestNNLayerForward.cpp`
- `tst/gputests/TestNNLayerBackward.cpp`

#### 2.1.3 NNWeight Core Test Suite

**Objective:** Test weight matrix operations thoroughly.

**Tasks:**
- [ ] Test all weight initialization schemes (Xavier, Gaussian, Uniform, SELU)
- [ ] Test weight update algorithms
- [ ] Test L1/L2 regularization
- [ ] Test gradient computation correctness
- [ ] Test weight sharing between layers

**Files to Create:**
- `tst/gputests/TestNNWeightInit.cpp`
- `tst/gputests/TestNNWeightUpdate.cpp`

#### 2.1.4 Virtual GPU Test Infrastructure

**Objective:** Set up vGPU testing environment.

**Tasks:**
- [ ] Document vGPU setup requirements
- [ ] Create test runner compatible with vGPU
- [ ] Add vGPU detection and configuration
- [ ] Create CI configuration for vGPU testing

**Files to Create:**
- `docs/VGPU_TESTING.md`
- `tst/vgpu_setup.sh` (helper script)

---

### Phase 2: Core Infrastructure Unification (Weeks 7-10)

#### 2.2.1 Unified Configuration Management

**Objective:** Create a single, consistent configuration system across all modules.

**Tasks:**
- [ ] Create unified `DsstneConfig` class encapsulating all configuration
- [ ] Migrate from JSON-only to support YAML and environment variables
- [ ] Implement configuration validation with clear error messages
- [ ] Add configuration versioning for backward compatibility

**Files to Modify:**
- `src/amazon/dsstne/utils/cdl.h/cpp`
- `src/amazon/dsstne/engine/NNTypes.h`
- New: `src/amazon/dsstne/config/DsstneConfig.h/cpp`

#### 2.2.2 Centralized Error Handling

**Objective:** Implement consistent error handling and logging across modules.

**Tasks:**
- [ ] Create `DsstneError` exception hierarchy
- [ ] Implement centralized logging with severity levels
- [ ] Add error codes and messages for all failure modes
- [ ] Create error recovery mechanisms where applicable

**Files to Create:**
- `src/amazon/dsstne/common/DsstneError.h/cpp`
- `src/amazon/dsstne/common/Logger.h/cpp`

---

### Phase 3: Gradient & Weight Optimization (Weeks 11-14)

> **Note:** These fixes are critical for training correctness.

#### 2.3.1 Centered Derivative Implementation

**Objective:** Replace non-centered derivative formula with centered version.

**Tasks:**
- [ ] Implement centered derivative formula in GPU kernels
- [ ] Add numerical stability improvements
- [ ] Benchmark accuracy improvements
- [ ] Update documentation with mathematical formulation

**Files to Modify:**
- `src/amazon/dsstne/engine/kernels.cu`
- `src/amazon/dsstne/engine/kDelta.cu`

#### 2.3.2 Explicit Bias Gradient

**Objective:** Replace bias gradient hack with explicit implementation.

**Tasks:**
- [ ] Implement dedicated bias gradient computation
- [ ] Add `_pbBiasGradient` buffer to `NNWeight`
- [ ] Optimize memory layout for gradient accumulation
- [ ] Verify correctness with gradient checking

**Files to Modify:**
- `src/amazon/dsstne/engine/NNWeight.h/cpp`
- `src/amazon/dsstne/engine/NNNetwork.cpp`

---

### Phase 4: Streaming & Efficiency (Weeks 15-18)

#### 2.4.1 Streaming Inference Pipeline

**Objective:** Eliminate NetCDF regeneration inefficiency in prediction.

**Tasks:**
- [ ] Create streaming data loader for inference
- [ ] Implement in-memory data conversion
- [ ] Add batch streaming support
- [ ] Optimize for low-latency inference

**Files to Modify:**
- `src/amazon/dsstne/utils/Predict.cpp`
- New: `src/amazon/dsstne/utils/StreamingDataLoader.h/cpp`

---

### Phase 5: Language Bindings (Weeks 19-22)

> **Note:** Breaking changes are acceptable - original external connections likely defunct.

#### 2.5.1 Python Binding Modernization

**Objective:** Modernize Python bindings with NumPy 2.x support.

**Tasks:**
- [ ] Update for NumPy 2.x compatibility
- [ ] Add type stubs for IDE support
- [ ] Implement Pythonic API wrapper
- [ ] Add async/await support for batch operations

**Files to Modify:**
- `python/dsstnemodule.cc`
- `python/setup.py`
- New: `python/dsstne/__init__.py` (Python wrapper)

#### 2.5.2 Java API Enhancement

**Objective:** Provide a more idiomatic Java API.

**Tasks:**
- [ ] Create Builder patterns for configuration
- [ ] Add try-with-resources support for context management
- [ ] Implement CompletableFuture for async operations
- [ ] Add proper JavaDoc documentation

**Files to Modify:**
- `java/src/main/java/com/amazon/dsstne/`
- `java/pom.xml` (update dependencies)

---

### Phase 6: Parallel & Multi-Dimensional Extensions (Weeks 23-24)

> **Note:** Only after core inference & training is perfected.

#### 2.6.1 Multi-Dimensional Output Support

**Objective:** Fix the limitation where topK only supports 1-D outputs.

**Tasks:**
- [ ] Extend `DsstneContext` to support N-D output layers
- [ ] Modify `CalculateTopK` to handle multi-dimensional tensors
- [ ] Update Python and Java bindings accordingly
- [ ] Add comprehensive test cases for N-D scenarios

**Files to Modify:**
- `src/amazon/dsstne/runtime/DsstneContext.cpp`
- `src/amazon/dsstne/engine/NNNetwork.cpp`
- `python/NNNetworkFunctions.h`
- `java/src/main/native/com_amazon_dsstne_Dsstne.cpp`

#### 2.6.2 Data-Parallel Weight Distribution

**Objective:** Implement proper data-parallel detection and weight distribution.

**Tasks:**
- [ ] Create `ParallelismDetector` utility class
- [ ] Implement automatic data-parallel detection in `NNWeight`
- [ ] Optimize weight synchronization for data-parallel mode
- [ ] Add benchmarks comparing model-parallel vs data-parallel

**Files to Modify:**
- `src/amazon/dsstne/engine/NNWeight.cpp`
- `src/amazon/dsstne/engine/NNNetwork.cpp`
- New: `src/amazon/dsstne/engine/ParallelismDetector.h/cpp`

#### 2.6.3 Multi-GPU Node Filter Support

**Objective:** Add Node Filter support for multi-GPU configurations.

**Tasks:**
- [ ] Extend filter infrastructure for distributed execution
- [ ] Implement filter synchronization across GPUs
- [ ] Add filter result aggregation
- [ ] Test with various filter types

**Files to Modify:**
- `src/amazon/dsstne/utils/Filters.cpp`
- `src/amazon/dsstne/utils/FilterHelper.cpp`
- `src/amazon/dsstne/utils/NNRecsGenerator.cpp`

#### 2.6.4 KNN Module Integration

**Objective:** Integrate KNN module with main neural network pipeline.

**Tasks:**
- [ ] Create `DataPipeline` abstraction for shared data handling
- [ ] Implement streaming support between NN inference and KNN
- [ ] Add support for embedding extraction → KNN lookup workflow
- [ ] Create Python/Java bindings for KNN functions

**Files to Create/Modify:**
- `src/amazon/dsstne/pipeline/DataPipeline.h/cpp`
- `src/amazon/dsstne/pipeline/EmbeddingKnnPipeline.h/cpp`
- `python/dsstnemodule.cc`
- `java/src/main/native_knn/com_amazon_dsstne_knn_KNearestNeighborsCuda.cpp`

---

## 3. Testing Strategy

### 3.1 Virtual GPU Testing Environment

**Environment Setup:**
- vGPU infrastructure for CI/CD
- Compatible with NVIDIA GRID or similar virtualization
- Memory and compute resource isolation

**Configuration:**
```bash
# vGPU detection
nvidia-smi -L  # List available virtual GPUs
# Test with specific vGPU
CUDA_VISIBLE_DEVICES=0 ./run_tests.sh
```

### 3.2 Unit Testing Expansion (Prioritized)

| Module | Current Tests | Target Tests | Priority |
|--------|---------------|--------------|----------|
| **NNNetwork (Inference)** | 0 | 30+ | ⭐ CRITICAL |
| **NNNetwork (Training)** | 0 | 30+ | ⭐ CRITICAL |
| **NNLayer** | 0 | 25+ | ⭐ CRITICAL |
| **NNWeight** | 0 | 20+ | ⭐ CRITICAL |
| Utils | 85+ | 85+ | ✅ Done |
| NetCDFhelper | 36+ | 40+ | High |
| KNN | 0 | 15+ | Low (Phase 6) |
| Bindings | 11 (Java) | 30+ | Medium |

### 3.3 Integration Testing

- [ ] Create end-to-end test pipeline
- [ ] Add benchmark regression tests
- [ ] Implement GPU-based test automation
- [ ] Create multi-GPU test environment

### 3.4 Performance Benchmarks

- [ ] Establish baseline benchmarks
- [ ] Create benchmark automation
- [ ] Add comparison against TensorFlow/PyTorch
- [ ] Track memory usage over time

---

## 4. Documentation Updates

### 4.1 Developer Documentation

- [ ] Create Architecture Decision Records (ADRs)
- [ ] Document internal APIs
- [ ] Add code contribution guidelines
- [ ] Create debugging guide

### 4.2 User Documentation

- [ ] Update getting started guide for new features
- [ ] Create API reference documentation
- [ ] Add migration guide for breaking changes
- [ ] Create troubleshooting guide

### 4.3 API Documentation

- [ ] Generate Doxygen documentation for C++
- [ ] Generate Sphinx documentation for Python
- [ ] Generate Javadoc for Java bindings

---

## 5. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GPU compatibility issues | High | Medium | Extensive multi-GPU testing |
| Performance regression | High | Low | Continuous benchmarking |
| Breaking API changes | Medium | Medium | Deprecation warnings, migration guides |
| CUDA version incompatibility | Medium | Low | Multi-CUDA version testing |
| Memory exhaustion | High | Medium | Memory profiling, streaming support |

---

## 6. Success Metrics

1. **Test Coverage**: Achieve >80% code coverage for core modules
2. **Performance**: No regression >5% in existing benchmarks
3. **Build Success**: CI passing on all supported platforms
4. **Documentation**: 100% of public APIs documented
5. **Integration**: All standalone features accessible via unified API

---

## 7. Timeline Summary (Reprioritized)

| Phase | Weeks | Focus Area | Priority |
|-------|-------|------------|----------|
| 1 | 1-6 | **Core Inference & Training Tests** | ⭐ HIGHEST |
| 2 | 7-10 | Core Infrastructure Unification | High |
| 3 | 11-14 | Gradient & Weight Optimization | High |
| 4 | 15-18 | Streaming & Efficiency | Medium |
| 5 | 19-22 | Language Bindings | Medium |
| 6 | 23-24 | Parallel & Multi-Dim Extensions | Low (after core) |

**Total Duration:** 24 weeks (6 months)

---

## 8. Appendices

### A. File Change Summary

```
New Files:
├── src/amazon/dsstne/config/DsstneConfig.h
├── src/amazon/dsstne/config/DsstneConfig.cpp
├── src/amazon/dsstne/common/DsstneError.h
├── src/amazon/dsstne/common/DsstneError.cpp
├── src/amazon/dsstne/common/Logger.h
├── src/amazon/dsstne/common/Logger.cpp
├── src/amazon/dsstne/engine/ParallelismDetector.h
├── src/amazon/dsstne/engine/ParallelismDetector.cpp
├── src/amazon/dsstne/pipeline/DataPipeline.h
├── src/amazon/dsstne/pipeline/DataPipeline.cpp
├── src/amazon/dsstne/pipeline/EmbeddingKnnPipeline.h
├── src/amazon/dsstne/pipeline/EmbeddingKnnPipeline.cpp
├── src/amazon/dsstne/utils/StreamingDataLoader.h
├── src/amazon/dsstne/utils/StreamingDataLoader.cpp
└── python/dsstne/__init__.py

Modified Files:
├── src/amazon/dsstne/runtime/DsstneContext.cpp
├── src/amazon/dsstne/engine/NNNetwork.cpp
├── src/amazon/dsstne/engine/NNWeight.cpp
├── src/amazon/dsstne/engine/NNWeight.h
├── src/amazon/dsstne/engine/kernels.cu
├── src/amazon/dsstne/engine/kDelta.cu
├── src/amazon/dsstne/utils/Filters.cpp
├── src/amazon/dsstne/utils/FilterHelper.cpp
├── src/amazon/dsstne/utils/NNRecsGenerator.cpp
├── src/amazon/dsstne/utils/Predict.cpp
├── src/amazon/dsstne/utils/cdl.h
├── src/amazon/dsstne/utils/cdl.cpp
├── python/dsstnemodule.cc
├── python/setup.py
├── java/src/main/java/com/amazon/dsstne/*
└── java/pom.xml
```

### B. Dependencies

- CUDA 9.1+ (recommend 11.x for newer GPUs/vGPU)
- cuDNN 7.0+
- OpenMPI 2.0+
- NetCDF 4.x
- jsoncpp
- CppUnit (testing)
- Python 3.7+ with NumPy
- Java 8+

### C. Breaking Changes Policy

> **Breaking changes are acceptable** for this project.
> 
> Original external connections to the repository are likely defunct.
> The focus is on creating a clean, well-tested codebase rather than
> maintaining backward compatibility with unmaintained integrations.

**Acceptable Breaking Changes:**
- API signature changes in Python/Java bindings
- Configuration file format updates
- CLI argument changes
- Removed deprecated functionality
- NetCDF format version updates (with migration tools)

**Mitigation:**
- Document all breaking changes in CHANGELOG.md
- Provide migration guides where feasible
- Version the API explicitly

---

*Document Version: 1.1*
*Last Updated: April 2026*
*Authors: GitHub Copilot Agent*

### Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | April 2026 | Initial plan |
| 1.1 | April 2026 | Reprioritized phases: core inference/training first, added vGPU testing, breaking changes acceptable |
