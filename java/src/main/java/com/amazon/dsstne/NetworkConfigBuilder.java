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

package com.amazon.dsstne;

import java.util.Objects;

/**
 * Builder pattern for creating NetworkConfig instances.
 * 
 * <p>Example usage:
 * <pre>{@code
 * NetworkConfig config = NetworkConfigBuilder.builder()
 *     .networkFilePath("/path/to/model.nc")
 *     .batchSize(32)
 *     .maxK(100)
 *     .build();
 * }</pre>
 * 
 * @since 2.0
 */
public final class NetworkConfigBuilder {

    private String networkFilePath;
    private int batchSize = 32;
    private int maxK = -1;
    private boolean shuffleIndices = true;
    private String errorFunction = "CrossEntropy";
    private int gpuDeviceId = -1;

    /**
     * Private constructor - use {@link #builder()} to create instances.
     */
    private NetworkConfigBuilder() {
    }

    /**
     * Creates a new builder instance.
     * 
     * @return a new builder
     */
    public static NetworkConfigBuilder builder() {
        return new NetworkConfigBuilder();
    }

    /**
     * Creates a builder initialized from an existing config.
     * 
     * @param config the config to copy from
     * @return a new builder with copied values
     */
    public static NetworkConfigBuilder from(NetworkConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return builder()
            .networkFilePath(config.getNetworkFilePath())
            .batchSize(config.getBatchSize())
            .maxK(config.getMaxK());
    }

    /**
     * Sets the path to the network file.
     * 
     * @param networkFilePath path to the NetCDF model file
     * @return this builder
     */
    public NetworkConfigBuilder networkFilePath(String networkFilePath) {
        this.networkFilePath = Objects.requireNonNull(networkFilePath, "networkFilePath cannot be null");
        return this;
    }

    /**
     * Sets the batch size for inference.
     * 
     * @param batchSize number of samples to process in parallel (must be positive)
     * @return this builder
     * @throws IllegalArgumentException if batchSize is not positive
     */
    public NetworkConfigBuilder batchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("batchSize must be positive, got: " + batchSize);
        }
        this.batchSize = batchSize;
        return this;
    }

    /**
     * Sets the maximum K for top-K predictions.
     * 
     * @param maxK maximum K value (-1 for all predictions)
     * @return this builder
     * @throws IllegalArgumentException if maxK is less than -1
     */
    public NetworkConfigBuilder maxK(int maxK) {
        if (maxK < -1) {
            throw new IllegalArgumentException("maxK must be -1 or positive, got: " + maxK);
        }
        this.maxK = maxK;
        return this;
    }

    /**
     * Sets whether to shuffle indices during training.
     * 
     * @param shuffleIndices true to shuffle indices
     * @return this builder
     */
    public NetworkConfigBuilder shuffleIndices(boolean shuffleIndices) {
        this.shuffleIndices = shuffleIndices;
        return this;
    }

    /**
     * Sets the error function (loss function).
     * 
     * @param errorFunction name of the error function
     * @return this builder
     */
    public NetworkConfigBuilder errorFunction(String errorFunction) {
        this.errorFunction = Objects.requireNonNull(errorFunction, "errorFunction cannot be null");
        return this;
    }

    /**
     * Sets the GPU device ID to use.
     * 
     * @param gpuDeviceId device ID (-1 for auto-select)
     * @return this builder
     */
    public NetworkConfigBuilder gpuDeviceId(int gpuDeviceId) {
        this.gpuDeviceId = gpuDeviceId;
        return this;
    }

    /**
     * Validates the configuration.
     * 
     * @throws IllegalStateException if configuration is invalid
     */
    private void validate() {
        if (networkFilePath == null || networkFilePath.isEmpty()) {
            throw new IllegalStateException("networkFilePath must be set");
        }
    }

    /**
     * Builds the NetworkConfig instance.
     * 
     * @return a new NetworkConfig
     * @throws IllegalStateException if configuration is invalid
     */
    public NetworkConfig build() {
        validate();
        return new NetworkConfig(
            networkFilePath,
            batchSize,
            maxK,
            shuffleIndices,
            errorFunction,
            gpuDeviceId
        );
    }

    @Override
    public String toString() {
        return "NetworkConfigBuilder{" +
            "networkFilePath='" + networkFilePath + '\'' +
            ", batchSize=" + batchSize +
            ", maxK=" + maxK +
            ", shuffleIndices=" + shuffleIndices +
            ", errorFunction='" + errorFunction + '\'' +
            ", gpuDeviceId=" + gpuDeviceId +
            '}';
    }
}
