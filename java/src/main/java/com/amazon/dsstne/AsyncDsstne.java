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

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Asynchronous wrapper for DSSTNE operations using CompletableFuture.
 * 
 * <p>This class provides async versions of DSSTNE operations that can be
 * composed using CompletableFuture's fluent API.</p>
 * 
 * <p>Example usage:
 * <pre>{@code
 * AsyncDsstne async = new AsyncDsstne(network);
 * 
 * // Single async operation
 * CompletableFuture<TopKOutput[]> future = async.predictAsync(inputData);
 * 
 * // Chained operations
 * async.predictAsync(inputData)
 *     .thenApply(results -> processResults(results))
 *     .thenAccept(processed -> saveResults(processed))
 *     .exceptionally(ex -> { handleError(ex); return null; });
 * 
 * // Multiple parallel operations
 * CompletableFuture.allOf(
 *     async.predictAsync(batch1),
 *     async.predictAsync(batch2),
 *     async.predictAsync(batch3)
 * ).join();
 * }</pre>
 * 
 * @since 2.0
 */
public class AsyncDsstne implements AutoCloseable {

    private final NNNetwork network;
    private final ExecutorService executor;
    private final boolean ownsExecutor;

    /**
     * Creates an AsyncDsstne with a default thread pool.
     * 
     * @param network the underlying network instance
     */
    public AsyncDsstne(NNNetwork network) {
        this(network, createDefaultExecutor(), true);
    }

    /**
     * Creates an AsyncDsstne with a custom executor.
     * 
     * @param network the underlying network instance
     * @param executor the executor to use for async operations
     */
    public AsyncDsstne(NNNetwork network, ExecutorService executor) {
        this(network, executor, false);
    }

    private AsyncDsstne(NNNetwork network, ExecutorService executor, boolean ownsExecutor) {
        this.network = Objects.requireNonNull(network, "network cannot be null");
        this.executor = Objects.requireNonNull(executor, "executor cannot be null");
        this.ownsExecutor = ownsExecutor;
    }

    /**
     * Asynchronously run prediction on input data.
     * 
     * @param inputData the input data for prediction
     * @return a CompletableFuture containing the top-K predictions
     */
    public CompletableFuture<TopKOutput[]> predictAsync(NNDataSet[] inputData) {
        return CompletableFuture.supplyAsync(() -> {
            return network.predict(inputData);
        }, executor);
    }

    /**
     * Run prediction with a callback for progress updates.
     * 
     * @param inputData the input data for prediction
     * @param progressCallback callback invoked with progress (0.0 to 1.0)
     * @return a CompletableFuture containing the predictions
     */
    public CompletableFuture<TopKOutput[]> predictWithProgressAsync(
            NNDataSet[] inputData,
            Consumer<Float> progressCallback) {
        
        return CompletableFuture.supplyAsync(() -> {
            if (progressCallback != null) {
                progressCallback.accept(0.0f);
            }
            
            TopKOutput[] results = network.predict(inputData);
            
            if (progressCallback != null) {
                progressCallback.accept(1.0f);
            }
            
            return results;
        }, executor);
    }

    /**
     * Batch multiple predictions in parallel.
     * 
     * @param batches array of input batches
     * @return a CompletableFuture that completes when all batches are processed
     */
    @SuppressWarnings("unchecked")
    public CompletableFuture<TopKOutput[][]> predictBatchesAsync(NNDataSet[][] batches) {
        CompletableFuture<TopKOutput[]>[] futures = new CompletableFuture[batches.length];
        
        for (int i = 0; i < batches.length; i++) {
            final NNDataSet[] batch = batches[i];
            futures[i] = predictAsync(batch);
        }
        
        return CompletableFuture.allOf(futures)
            .thenApply(v -> {
                TopKOutput[][] results = new TopKOutput[batches.length][];
                for (int i = 0; i < futures.length; i++) {
                    results[i] = futures[i].join();
                }
                return results;
            });
    }

    /**
     * Get the underlying network instance.
     * 
     * @return the network instance
     */
    public NNNetwork getNetwork() {
        return network;
    }

    /**
     * Get the executor service.
     * 
     * @return the executor service
     */
    public ExecutorService getExecutor() {
        return executor;
    }

    /**
     * Shuts down the executor if owned by this instance.
     */
    @Override
    public void close() {
        if (ownsExecutor) {
            executor.shutdown();
        }
    }

    /**
     * Creates a default executor service for async operations.
     */
    private static ExecutorService createDefaultExecutor() {
        int numThreads = Math.max(2, Runtime.getRuntime().availableProcessors() / 2);
        
        return Executors.newFixedThreadPool(numThreads, new ThreadFactory() {
            private final AtomicInteger counter = new AtomicInteger(0);
            
            @Override
            public Thread newThread(Runnable r) {
                Thread thread = new Thread(r);
                thread.setName("dsstne-async-" + counter.incrementAndGet());
                thread.setDaemon(true);
                return thread;
            }
        });
    }

    /**
     * Builder for AsyncDsstne.
     */
    public static class Builder {
        private NNNetwork network;
        private ExecutorService executor;

        public Builder network(NNNetwork network) {
            this.network = network;
            return this;
        }

        public Builder executor(ExecutorService executor) {
            this.executor = executor;
            return this;
        }

        public AsyncDsstne build() {
            if (network == null) {
                throw new IllegalStateException("network must be set");
            }
            
            if (executor != null) {
                return new AsyncDsstne(network, executor);
            } else {
                return new AsyncDsstne(network);
            }
        }
    }

    /**
     * Creates a new builder.
     * 
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }
}
