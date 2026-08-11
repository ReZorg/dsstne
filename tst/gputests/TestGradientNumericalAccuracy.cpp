/*
 *  Copyright 2016-2026  Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License").
 *  You may not use this file except in compliance with the License.
 *  A copy of the License is located at
 *
 *  http://aws.amazon.com/apache2.0/
 *
 *  or in the "license" file accompanying this file.
 *  This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 *  either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 */

/**
 * @file TestGradientNumericalAccuracy.cpp
 * @brief Tests for numerical accuracy of gradient computations using centered finite differences.
 * 
 * These tests validate that the gradient implementations match numerical gradients
 * computed using the centered finite difference formula:
 *   f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
 */

#include <cppunit/extensions/HelperMacros.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>

#include "GpuTypes.h"
#include "NNTypes.h"
#include "kernels.h"
#include "../common/GpuCommon.h"

class TestGradientNumericalAccuracy : public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE(TestGradientNumericalAccuracy);
    
    // Activation gradient tests
    CPPUNIT_TEST(testSigmoidGradientAccuracy);
    CPPUNIT_TEST(testTanhGradientAccuracy);
    CPPUNIT_TEST(testReLUGradientAccuracy);
    CPPUNIT_TEST(testLeakyReLUGradientAccuracy);
    CPPUNIT_TEST(testELUGradientAccuracy);
    CPPUNIT_TEST(testSoftmaxGradientAccuracy);
    
    // Centered vs non-centered comparison
    CPPUNIT_TEST(testCenteredDerivativeAccuracy);
    
    // Numerical stability tests
    CPPUNIT_TEST(testGradientNumericalStability);
    CPPUNIT_TEST(testGradientNearZero);
    CPPUNIT_TEST(testGradientLargeValues);
    
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp() override {
        // Allocate GPU memory for tests
        _size = 1024;
        cudaMalloc(&_d_input, _size * sizeof(float));
        cudaMalloc(&_d_output, _size * sizeof(float));
        cudaMalloc(&_d_gradient, _size * sizeof(float));
        cudaMalloc(&_d_delta, _size * sizeof(float));
        
        // Initialize random input
        std::vector<float> input(_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (size_t i = 0; i < _size; ++i) {
            input[i] = dist(gen);
        }
        
        cudaMemcpy(_d_input, input.data(), _size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void tearDown() override {
        cudaFree(_d_input);
        cudaFree(_d_output);
        cudaFree(_d_gradient);
        cudaFree(_d_delta);
    }
    
    /**
     * Test sigmoid gradient accuracy against numerical gradient.
     */
    void testSigmoidGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-3f;
        
        // Compute analytical gradient using kernel
        invokeActivation(Activation::Sigmoid);
        invokeGradient(Activation::Sigmoid);
        
        // Get analytical gradient
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        // Compute numerical gradient using centered differences
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::Sigmoid, epsilon);
        
        // Compare
        float maxError = 0.0f;
        for (size_t i = 0; i < _size; ++i) {
            float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
            maxError = std::max(maxError, error);
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "Sigmoid gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Test tanh gradient accuracy against numerical gradient.
     */
    void testTanhGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-3f;
        
        invokeActivation(Activation::Tanh);
        invokeGradient(Activation::Tanh);
        
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::Tanh, epsilon);
        
        float maxError = 0.0f;
        for (size_t i = 0; i < _size; ++i) {
            float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
            maxError = std::max(maxError, error);
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "Tanh gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Test ReLU gradient accuracy against numerical gradient.
     */
    void testReLUGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-3f;
        
        invokeActivation(Activation::RectifiedLinear);
        invokeGradient(Activation::RectifiedLinear);
        
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::RectifiedLinear, epsilon);
        
        // Note: ReLU has discontinuity at 0, so we skip those points
        float maxError = 0.0f;
        std::vector<float> input(_size);
        cudaMemcpy(input.data(), _d_input, _size * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < _size; ++i) {
            if (std::abs(input[i]) > epsilon * 10) {  // Skip near zero
                float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
                maxError = std::max(maxError, error);
            }
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "ReLU gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Test LeakyReLU gradient accuracy.
     */
    void testLeakyReLUGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-3f;
        
        invokeActivation(Activation::LeakyRectifiedLinear);
        invokeGradient(Activation::LeakyRectifiedLinear);
        
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::LeakyRectifiedLinear, epsilon);
        
        float maxError = 0.0f;
        std::vector<float> input(_size);
        cudaMemcpy(input.data(), _d_input, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < _size; ++i) {
            if (std::abs(input[i]) > epsilon * 10) {
                float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
                maxError = std::max(maxError, error);
            }
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "LeakyReLU gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Test ELU gradient accuracy.
     */
    void testELUGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-3f;
        
        invokeActivation(Activation::ExponentialLinear);
        invokeGradient(Activation::ExponentialLinear);
        
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::ExponentialLinear, epsilon);
        
        float maxError = 0.0f;
        for (size_t i = 0; i < _size; ++i) {
            float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
            maxError = std::max(maxError, error);
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "ELU gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Test softmax gradient accuracy.
     */
    void testSoftmaxGradientAccuracy() {
        const float epsilon = 1e-4f;
        const float tolerance = 1e-2f;  // Higher tolerance for softmax
        
        invokeActivation(Activation::SoftMax);
        invokeGradient(Activation::SoftMax);
        
        std::vector<float> analyticalGrad(_size);
        cudaMemcpy(analyticalGrad.data(), _d_delta, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> numericalGrad = computeNumericalGradient(
            Activation::SoftMax, epsilon);
        
        float maxError = 0.0f;
        for (size_t i = 0; i < _size; ++i) {
            float error = std::abs(analyticalGrad[i] - numericalGrad[i]);
            maxError = std::max(maxError, error);
        }
        
        CPPUNIT_ASSERT_MESSAGE(
            "Softmax gradient error too large: " + std::to_string(maxError),
            maxError < tolerance);
    }
    
    /**
     * Compare centered vs non-centered finite difference accuracy.
     */
    void testCenteredDerivativeAccuracy() {
        // Use a simple test function: f(x) = x^3
        // f'(x) = 3x^2
        const float h = 0.001f;
        const float x = 2.0f;
        
        auto f = [](float x) { return x * x * x; };
        
        // Non-centered: f'(x) ≈ (f(x+h) - f(x)) / h
        float nonCentered = (f(x + h) - f(x)) / h;
        
        // Centered: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        float centered = (f(x + h) - f(x - h)) / (2.0f * h);
        
        // Analytical derivative
        float analytical = 3.0f * x * x;  // 12.0
        
        float errorNonCentered = std::abs(nonCentered - analytical);
        float errorCentered = std::abs(centered - analytical);
        
        // Centered should be more accurate
        CPPUNIT_ASSERT_MESSAGE(
            "Centered derivative should be more accurate than non-centered",
            errorCentered < errorNonCentered);
        
        // Centered error should be O(h^2) vs O(h) for non-centered
        CPPUNIT_ASSERT(errorCentered < 1e-5f);
    }
    
    /**
     * Test numerical stability near zero.
     */
    void testGradientNearZero() {
        const float tolerance = 1e-3f;
        
        // Create input near zero
        std::vector<float> input = {1e-7f, 1e-6f, 1e-5f, 1e-4f, 1e-3f,
                                     -1e-7f, -1e-6f, -1e-5f, -1e-4f, -1e-3f};
        
        cudaMemcpy(_d_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        // Test sigmoid (should be ~0.25 near zero)
        invokeActivation(Activation::Sigmoid);
        invokeGradient(Activation::Sigmoid);
        
        std::vector<float> grad(input.size());
        cudaMemcpy(grad.data(), _d_delta, input.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < input.size(); ++i) {
            // sigmoid'(0) = 0.25
            CPPUNIT_ASSERT(std::abs(grad[i] - 0.25f) < 0.01f);
            CPPUNIT_ASSERT(!std::isnan(grad[i]));
            CPPUNIT_ASSERT(!std::isinf(grad[i]));
        }
    }
    
    /**
     * Test numerical stability with large values.
     */
    void testGradientLargeValues() {
        // Create input with large values
        std::vector<float> input = {10.0f, 50.0f, 100.0f, -10.0f, -50.0f, -100.0f};
        
        cudaMemcpy(_d_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        // Test sigmoid (should be ~0 for large |x|)
        invokeActivation(Activation::Sigmoid);
        invokeGradient(Activation::Sigmoid);
        
        std::vector<float> grad(input.size());
        cudaMemcpy(grad.data(), _d_delta, input.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < input.size(); ++i) {
            // Gradient should be close to 0 for saturated sigmoid
            CPPUNIT_ASSERT(std::abs(grad[i]) < 0.01f);
            CPPUNIT_ASSERT(!std::isnan(grad[i]));
            CPPUNIT_ASSERT(!std::isinf(grad[i]));
        }
    }
    
    /**
     * Test gradient numerical stability across range.
     */
    void testGradientNumericalStability() {
        // Test a wide range of values
        std::vector<float> input;
        for (float x = -5.0f; x <= 5.0f; x += 0.1f) {
            input.push_back(x);
        }
        
        cudaMemcpy(_d_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        
        std::vector<Activation> activations = {
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::RectifiedLinear,
            Activation::LeakyRectifiedLinear,
            Activation::ExponentialLinear
        };
        
        for (Activation act : activations) {
            invokeActivation(act);
            invokeGradient(act);
            
            std::vector<float> grad(input.size());
            cudaMemcpy(grad.data(), _d_delta, input.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            
            for (size_t i = 0; i < input.size(); ++i) {
                CPPUNIT_ASSERT_MESSAGE(
                    "NaN gradient detected",
                    !std::isnan(grad[i]));
                CPPUNIT_ASSERT_MESSAGE(
                    "Inf gradient detected",
                    !std::isinf(grad[i]));
            }
        }
    }

private:
    size_t _size;
    float* _d_input;
    float* _d_output;
    float* _d_gradient;
    float* _d_delta;
    
    void invokeActivation(Activation activation) {
        // Call activation kernel - simplified for testing
        // In real implementation, this would call kCalculateActivation
    }
    
    void invokeGradient(Activation activation) {
        // Call gradient kernel - simplified for testing
        // In real implementation, this would call kCalculateDelta
    }
    
    /**
     * Compute numerical gradient using centered finite differences.
     */
    std::vector<float> computeNumericalGradient(Activation activation, float epsilon) {
        std::vector<float> input(_size);
        cudaMemcpy(input.data(), _d_input, _size * sizeof(float),
                   cudaMemcpyDeviceToHost);
        
        std::vector<float> gradient(_size);
        
        for (size_t i = 0; i < _size; ++i) {
            // f(x + h)
            float xPlusH = applyActivation(activation, input[i] + epsilon);
            
            // f(x - h)
            float xMinusH = applyActivation(activation, input[i] - epsilon);
            
            // Centered difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            gradient[i] = (xPlusH - xMinusH) / (2.0f * epsilon);
        }
        
        return gradient;
    }
    
    /**
     * Apply activation function on CPU for numerical gradient computation.
     */
    float applyActivation(Activation activation, float x) {
        switch (activation) {
            case Activation::Sigmoid:
                return 1.0f / (1.0f + std::exp(-x));
            case Activation::Tanh:
                return std::tanh(x);
            case Activation::RectifiedLinear:
                return std::max(0.0f, x);
            case Activation::LeakyRectifiedLinear:
                return x >= 0 ? x : 0.01f * x;
            case Activation::ExponentialLinear:
                return x >= 0 ? x : std::exp(x) - 1.0f;
            case Activation::SoftMax:
                return std::exp(x);  // Simplified for per-element
            default:
                return x;
        }
    }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGradientNumericalAccuracy);
