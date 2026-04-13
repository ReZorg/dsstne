# Amazon DSSTNE - Testing Guide

## Overview

This document describes the testing infrastructure for Amazon DSSTNE, including how to run existing tests, add new tests, and understand test coverage.

## Test Categories

### 1. CPU Unit Tests (`tst/unittests/`)

Tests that run on CPU only, no GPU required. These tests cover:
- Utility functions (Utils.cpp)
- NetCDF helper functions (NetCDFhelper.cpp)
- Data type handling
- Configuration parsing

**Test Files:**
| File | Tests | Coverage |
|------|-------|----------|
| `TestNetCDFhelper.cpp` | 6 | loadIndex function |
| `TestUtils.cpp` | 1 | isNetCDFfile function |
| `TestUtilsComprehensive.cpp` | 35+ | Full Utils module |
| `TestNetCDFhelperExtended.cpp` | 30+ | Extended NetCDF helpers |
| `TestDataTypes.cpp` | 20+ | Data types and constants |

### 2. GPU Tests (`tst/gputests/`)

Tests requiring CUDA-capable GPU:
- Activation functions
- Cost/loss functions
- GPU sorting algorithms
- Neural network validation

**Test Files:**
| File | Tests | Coverage |
|------|-------|----------|
| `TestActivationFunctions.cpp` | Activation kernels |
| `TestCostFunctions.cpp` | Loss function kernels |
| `TestSort.cpp` | GPU sorting |
| `TestDune.cpp` | DUNE framework tests |
| `TestGpu.cpp` | General GPU tests |

### 3. Engine Tests (`tst/amazon/dsstne/engine/`)

Tests for core engine components:
- NNDataSet operations
- GpuBuffer management
- Data dimensions handling

**Test Files:**
| File | Tests | Coverage |
|------|-------|----------|
| `TestNNDataSet.cpp` | 15+ | Dataset creation, loading |
| `TestGpuBuffer.cpp` | 5+ | GPU buffer operations |
| `TestNNDataSetDimensions.cpp` | 5+ | Dimension handling |

### 4. Java Tests (`java/src/test/`)

JUnit tests for Java bindings.

---

## Running Tests

### Prerequisites

```bash
# Install test dependencies
sudo apt-get install libcppunit-dev cmake
```

### CPU Unit Tests

```bash
cd tst/unittests
mkdir build && cd build
cmake ..
make
./unittests
```

**Expected Output:**
```
TestNetCDFhelper::TestLoadIndexWithValidInput : OK
TestNetCDFhelper::TestLoadIndexWithDuplicateEntry : OK
...
OK (XX tests)
```

### GPU Tests (Requires CUDA)

```bash
cd tst
make
make run-tests
```

### Engine Tests (Requires CUDA and full build)

```bash
# First build the full DSSTNE library
cd /path/to/amazon-dsstne
make

# Then run engine tests
cd tst
make
LD_LIBRARY_PATH=$BUILD_DIR/lib ./bin/unittests
```

### Java Tests

```bash
cd java
mvn test
```

---

## Adding New Tests

### CPU Unit Tests

1. Create a new test file `tst/unittests/TestYourModule.cpp`:

```cpp
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestAssert.h>

// Include the header you're testing
#include "YourModule.h"

class TestYourModule : public CppUnit::TestFixture
{
public:
    void setUp() {
        // Setup code runs before each test
    }
    
    void tearDown() {
        // Cleanup code runs after each test
    }
    
    void testBasicFunctionality() {
        // Test implementation
        CPPUNIT_ASSERT_EQUAL(expected, actual);
    }
    
    void testEdgeCase() {
        CPPUNIT_ASSERT(condition);
    }
    
    void testExceptionThrown() {
        // This test expects an exception
    }
    
    // Register test suite
    CPPUNIT_TEST_SUITE(TestYourModule);
    CPPUNIT_TEST(testBasicFunctionality);
    CPPUNIT_TEST(testEdgeCase);
    CPPUNIT_TEST_EXCEPTION(testExceptionThrown, std::runtime_error);
    CPPUNIT_TEST_SUITE_END();
};
```

2. Include in `tst/unittests/main.cpp`:

```cpp
#include "TestYourModule.cpp"

int main()
{
    CppUnit::TextUi::TestRunner runner;
    // ... existing tests ...
    runner.addTest(TestYourModule::suite());
    return runner.run() ? EXIT_SUCCESS : EXIT_FAILURE;
}
```

### GPU Tests

1. Create `tst/gputests/TestYourGpuCode.cpp`:

```cpp
#include "cppunit/extensions/HelperMacros.h"
#include "GpuTypes.h"
#include "NNTypes.h"
#include "TestUtils.h"

class TestYourGpuCode : public CppUnit::TestFixture {
public:
    void testGpuKernel() {
        // Initialize GPU context if needed
        // Run GPU operations
        // Verify results
    }
    
    CPPUNIT_TEST_SUITE(TestYourGpuCode);
    CPPUNIT_TEST(testGpuKernel);
    CPPUNIT_TEST_SUITE_END();
};
```

2. Update `tst/gputests/CMakeLists.txt` to include your test.

### Engine Tests

Follow the pattern in `tst/amazon/dsstne/engine/` using CppUnit.

---

## Test Patterns

### Testing with Temporary Files

```cpp
class TestWithFiles : public CppUnit::TestFixture
{
private:
    string tempDir;
    
    void createTempDir() {
        char tmpl[] = "/tmp/dsstne_test_XXXXXX";
        char* result = mkdtemp(tmpl);
        if (result != nullptr) {
            tempDir = string(result);
        }
    }
    
    void removeTempDir() {
        if (!tempDir.empty()) {
            system(("rm -rf " + tempDir).c_str());
        }
    }

public:
    void setUp() {
        createTempDir();
    }
    
    void tearDown() {
        removeTempDir();
    }
    
    void testFileOperation() {
        string testFile = tempDir + "/test.txt";
        // Create and test files
    }
};
```

### Testing Expected Exceptions

```cpp
void testThrowsException() {
    // This test should throw std::runtime_error
}

CPPUNIT_TEST_EXCEPTION(testThrowsException, std::runtime_error);
```

### Testing Floating Point Values

```cpp
void testFloatCalculation() {
    float expected = 3.14159f;
    float actual = calculatePi();
    CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, 1e-5);
}
```

### Parameterized Testing

```cpp
void testMultipleInputs() {
    struct TestCase {
        int input;
        int expected;
    };
    
    TestCase cases[] = {
        {1, 1}, {2, 4}, {3, 9}, {4, 16}
    };
    
    for (const auto& tc : cases) {
        CPPUNIT_ASSERT_EQUAL(tc.expected, square(tc.input));
    }
}
```

---

## Test Assertions Reference

### CppUnit Assertions

| Assertion | Description |
|-----------|-------------|
| `CPPUNIT_ASSERT(condition)` | Assert condition is true |
| `CPPUNIT_ASSERT_MESSAGE(msg, cond)` | Assert with message |
| `CPPUNIT_ASSERT_EQUAL(expected, actual)` | Assert equality |
| `CPPUNIT_ASSERT_EQUAL_MESSAGE(msg, exp, act)` | Equality with message |
| `CPPUNIT_ASSERT_DOUBLES_EQUAL(exp, act, delta)` | Float equality |
| `CPPUNIT_FAIL(message)` | Force test failure |
| `CPPUNIT_ASSERT_THROW(expr, exception)` | Assert throws |
| `CPPUNIT_ASSERT_NO_THROW(expr)` | Assert no exception |

---

## Test Coverage

### Current Coverage Status

| Module | File Count | Test Coverage |
|--------|------------|---------------|
| `utils/Utils.cpp` | 36+ tests | ~90% |
| `utils/NetCDFhelper.cpp` | 36+ tests | ~85% |
| `engine/NNNetwork.cpp` | 0 tests | 0% |
| `engine/NNLayer.cpp` | 0 tests | 0% |
| `engine/NNWeight.cpp` | 0 tests | 0% |
| `engine/NNDataSet` | 15+ tests | ~70% |
| `knn/*` | 0 tests | 0% |

### Coverage Goals

- **Critical**: NNNetwork, NNLayer, NNWeight - target 80%
- **High**: Utils, NetCDFhelper - target 90%
- **Medium**: KNN module - target 60%
- **Low**: Language bindings - target 50%

---

## Continuous Integration

### Travis CI Configuration

The `.travis.yml` file configures automated testing:

```yaml
script:
  - cd tst/unittests
  - mkdir build && cd build
  - cmake ..
  - make
  - LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib ./unittests
```

### Local CI Simulation

```bash
# Simulate CI environment locally
docker build -t amazon-dsstne .
docker run amazon-dsstne /bin/bash -c "cd tst/unittests && mkdir -p build && cd build && cmake .. && make && ./unittests"
```

---

## Troubleshooting

### Common Issues

**1. CppUnit not found**
```bash
# Solution
sudo apt-get install libcppunit-dev
```

**2. NetCDF headers not found**
```bash
# Solution
sudo apt-get install libnetcdf-dev libnetcdf-c++4-dev
```

**3. CUDA tests fail to compile**
```bash
# Ensure CUDA is in path
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**4. GPU tests fail with "no CUDA-capable device"**
- Ensure you have a CUDA-capable GPU
- Run `nvidia-smi` to verify GPU visibility
- For Docker, use `nvidia-docker`

---

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Clean Up**: Use setUp/tearDown for resource management
3. **Descriptive Names**: Test names should describe what's being tested
4. **Test Edge Cases**: Include boundary conditions
5. **Fast Tests**: Keep unit tests fast (<100ms each)
6. **No GPU for Unit Tests**: Keep GPU tests separate
7. **Mock External Dependencies**: Use mocks where possible

---

*Document Version: 1.0*
*Last Updated: April 2026*
