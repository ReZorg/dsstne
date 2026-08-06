# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with Amazon DSSTNE.

## Table of Contents

1. [Build Issues](#build-issues)
2. [CUDA Issues](#cuda-issues)
3. [Memory Issues](#memory-issues)
4. [Training Issues](#training-issues)
5. [Inference Issues](#inference-issues)
6. [Python Binding Issues](#python-binding-issues)
7. [Java Binding Issues](#java-binding-issues)

---

## Build Issues

### CMake Cannot Find CUDA

**Symptoms:**
```
CMake Error: Could not find CUDA toolkit
```

**Solutions:**
1. Ensure CUDA is installed:
   ```bash
   nvcc --version
   ```
2. Set CUDA path explicitly:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   cmake -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..
   ```
3. Check that `nvcc` is in your PATH.

### cuBLAS Not Found

**Symptoms:**
```
error: cublas_v2.h: No such file or directory
```

**Solutions:**
1. Install CUDA toolkit with cuBLAS:
   ```bash
   sudo apt-get install libcublas-dev
   ```
2. Add include path:
   ```bash
   export CPATH=/usr/local/cuda/include:$CPATH
   ```

### OpenMPI Compilation Errors

**Symptoms:**
```
mpi.h: No such file or directory
```

**Solutions:**
1. Install OpenMPI development headers:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libopenmpi-dev openmpi-bin
   
   # CentOS/RHEL
   sudo yum install openmpi-devel
   ```
2. Load MPI module (HPC environments):
   ```bash
   module load openmpi
   ```

### NetCDF Linking Errors

**Symptoms:**
```
undefined reference to `nc_open'
```

**Solutions:**
1. Install NetCDF C++ bindings:
   ```bash
   sudo apt-get install libnetcdf-dev libnetcdf-c++4-dev
   ```
2. Check library paths:
   ```bash
   ldconfig -p | grep netcdf
   ```

---

## CUDA Issues

### CUDA Device Not Found

**Symptoms:**
```
CUDA Error: no CUDA-capable device is detected
```

**Solutions:**
1. Check driver installation:
   ```bash
   nvidia-smi
   ```
2. Verify CUDA device visibility:
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```
3. Check permissions:
   ```bash
   ls -la /dev/nvidia*
   sudo usermod -aG video $USER
   ```

### Compute Capability Mismatch

**Symptoms:**
```
CUDA Error: no kernel image is available for execution on the device
```

**Solutions:**
1. Rebuild with correct compute capability:
   ```bash
   cmake -DCUDA_ARCH="sm_70;sm_75;sm_80" ..
   ```
2. Check your GPU's compute capability:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

### CUDA Out of Memory

**Symptoms:**
```
CUDA Error: out of memory
```

**Solutions:**
1. Reduce batch size:
   ```bash
   train --batchSize 32  # Try smaller values
   ```
2. Check memory usage:
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   ```
3. Use memory pool:
   ```cpp
   cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024);
   ```

---

## Memory Issues

### Host Memory Exhausted

**Symptoms:**
```
std::bad_alloc
terminate called after throwing an instance of 'std::bad_alloc'
```

**Solutions:**
1. Use streaming data loader instead of loading all data:
   ```cpp
   StreamingDataLoader loader(batchSize);
   loader.setPrefetchDepth(2);  // Limit prefetch
   ```
2. Reduce model size or use data sharding.
3. Monitor memory usage:
   ```bash
   watch -n 1 free -h
   ```

### Memory Leaks

**Symptoms:**
- Memory usage grows over time
- Eventually crashes with OOM

**Solutions:**
1. Run with memory sanitizer:
   ```bash
   export ASAN_OPTIONS=detect_leaks=1
   ./bin/train ...
   ```
2. Use CUDA memcheck:
   ```bash
   cuda-memcheck ./bin/train ...
   ```
3. Ensure proper cleanup:
   ```cpp
   // Always free CUDA memory
   cudaFree(d_ptr);
   
   // Use RAII wrappers
   GpuBuffer<float> buffer(size);  // Auto-freed
   ```

---

## Training Issues

### Loss Not Decreasing

**Symptoms:**
- Training loss stays flat or oscillates
- Validation metrics don't improve

**Solutions:**
1. Check learning rate:
   ```json
   {
     "LearningRate": 0.001,  // Try smaller values
     "LearningRateDecay": 0.9
   }
   ```
2. Verify data normalization:
   ```bash
   # Check input statistics
   generateNetCDF --info input.nc
   ```
3. Add gradient clipping:
   ```json
   {
     "GradientClip": 1.0
   }
   ```

### NaN Loss

**Symptoms:**
```
Epoch 1: loss = nan
```

**Solutions:**
1. Reduce learning rate significantly
2. Add weight regularization:
   ```json
   {
     "L2Regularization": 0.0001
   }
   ```
3. Check for data issues:
   ```python
   import numpy as np
   data = np.load('data.npy')
   print(f"Has NaN: {np.isnan(data).any()}")
   print(f"Has Inf: {np.isinf(data).any()}")
   ```

### Slow Training

**Symptoms:**
- Training takes much longer than expected
- Low GPU utilization

**Solutions:**
1. Increase batch size (if memory allows)
2. Enable multi-GPU:
   ```bash
   mpirun -n 4 train ...
   ```
3. Check data loading:
   ```cpp
   loader.setPrefetchDepth(4);  // Enable prefetching
   ```
4. Profile with nvprof:
   ```bash
   nvprof ./bin/train ...
   ```

---

## Inference Issues

### Prediction Mismatch

**Symptoms:**
- Predictions differ from expected values
- Results differ between runs

**Solutions:**
1. Ensure deterministic mode:
   ```cpp
   cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
   ```
2. Check input preprocessing matches training
3. Verify model loading:
   ```cpp
   // Load and verify
   network->SetCheckpoint(true);
   network->Load(modelFile);
   ```

### Slow Inference

**Symptoms:**
- High latency for predictions
- Low throughput

**Solutions:**
1. Batch predictions:
   ```cpp
   // Instead of single predictions
   network->PredictBatch(inputs, batchSize);
   ```
2. Use streaming data loader
3. Keep model loaded (don't reload per request)

---

## Python Binding Issues

### Import Error

**Symptoms:**
```python
ImportError: No module named 'dsstne._dsstne'
```

**Solutions:**
1. Rebuild Python extension:
   ```bash
   cd python
   python setup.py build_ext --inplace
   pip install -e .
   ```
2. Check shared library dependencies:
   ```bash
   ldd _dsstne*.so
   ```
3. Set library path:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### NumPy Compatibility

**Symptoms:**
```
RuntimeError: NumPy API mismatch
```

**Solutions:**
1. Rebuild with current NumPy:
   ```bash
   pip uninstall dsstne
   pip install numpy --upgrade
   cd python && pip install -e .
   ```
2. Check NumPy version:
   ```python
   import numpy as np
   print(np.__version__)  # Should match build-time version
   ```

---

## Java Binding Issues

### JNI Library Not Found

**Symptoms:**
```java
java.lang.UnsatisfiedLinkError: no dsstne_jni in java.library.path
```

**Solutions:**
1. Set library path:
   ```bash
   export LD_LIBRARY_PATH=/path/to/libdsstne_jni.so:$LD_LIBRARY_PATH
   java -Djava.library.path=/path/to/lib -jar app.jar
   ```
2. Rebuild JNI library:
   ```bash
   cd java
   mvn clean package -P native
   ```

### GPU Memory Leak in Java

**Symptoms:**
- Java process memory grows over time
- CUDA OOM after multiple predictions

**Solutions:**
1. Use try-with-resources:
   ```java
   try (Dsstne dsstne = new Dsstne(config)) {
       TopKOutput[] results = dsstne.predict(input);
   }  // Auto-closed
   ```
2. Explicit cleanup:
   ```java
   dsstne.close();
   System.gc();  // Suggest GC
   ```

---

## Getting Help

If you're still experiencing issues:

1. **Search existing issues:** https://github.com/amzn/amazon-dsstne/issues
2. **Check the FAQ:** docs/FAQ.md
3. **Open a new issue** with:
   - DSSTNE version
   - CUDA version (`nvcc --version`)
   - Driver version (`nvidia-smi`)
   - OS and version
   - Complete error message
   - Minimal reproduction steps

## Debug Mode

Enable verbose logging for debugging:

```bash
export DSSTNE_LOG_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1
./bin/train ...
```

## Performance Profiling

Profile GPU performance:

```bash
# NVIDIA Nsight Systems
nsys profile ./bin/train ...

# NVIDIA Nsight Compute
ncu --set full ./bin/train ...

# nvprof (legacy)
nvprof --print-gpu-trace ./bin/train ...
```
