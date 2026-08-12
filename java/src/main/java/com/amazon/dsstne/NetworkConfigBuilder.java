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
 * Validating builder for creating {@link NetworkConfig} instances.
 *
 * <p>Unlike the Lombok-generated {@code NetworkConfig.with()} builder, this
 * builder validates its arguments eagerly and fails fast with descriptive
 * error messages.</p>
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
    private String networkName;
    private int batchSize = 32;
    private int maxK = NetworkConfig.ALL;

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
            .networkName(config.getNetworkName())
            .batchSize(config.getBatchSize())
            .maxK(config.getK());
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
     * Sets the name of the network. When unset, the name defaults to the
     * network file name (without the .nc suffix).
     *
     * @param networkName name of the network
     * @return this builder
     */
    public NetworkConfigBuilder networkName(String networkName) {
        this.networkName = networkName;
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
     * @param maxK maximum K value ({@link NetworkConfig#ALL} for the entire output layer)
     * @return this builder
     * @throws IllegalArgumentException if maxK is less than -1 or zero
     */
    public NetworkConfigBuilder maxK(int maxK) {
        if (maxK < NetworkConfig.ALL || maxK == 0) {
            throw new IllegalArgumentException("maxK must be -1 (ALL) or positive, got: " + maxK);
        }
        this.maxK = maxK;
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
        return NetworkConfig.with()
            .networkFilePath(networkFilePath)
            .networkName(networkName)
            .batchSize(batchSize)
            .k(maxK)
            .build();
    }

    @Override
    public String toString() {
        return "NetworkConfigBuilder{"
            + "networkFilePath='" + networkFilePath + '\''
            + ", networkName='" + networkName + '\''
            + ", batchSize=" + batchSize
            + ", maxK=" + maxK
            + '}';
    }
}
