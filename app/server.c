/**
 * server.c - Inference Server Core Implementation
 */

#include "server.h"
#include "Model.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/time.h>

static ServerState g_server = {0};

// Inference worker thread
static void* inference_worker(void* arg) {
    syslog(LOG_INFO, "Inference worker thread started");

    while (g_server.running) {
        pthread_mutex_lock(&g_server.queue.lock);

        // Wait for requests
        while (g_server.queue.count == 0 && g_server.running) {
            pthread_cond_wait(&g_server.queue.not_empty, &g_server.queue.lock);
        }

        if (!g_server.running) {
            pthread_mutex_unlock(&g_server.queue.lock);
            break;
        }

        // Get request from queue
        InferenceRequest* req = g_server.queue.requests[g_server.queue.head];
        g_server.queue.head = (g_server.queue.head + 1) % MAX_QUEUE_SIZE;
        g_server.queue.count--;

        pthread_cond_signal(&g_server.queue.not_full);
        pthread_mutex_unlock(&g_server.queue.lock);

        // Process inference
        pthread_mutex_lock(&req->lock);

        syslog(LOG_INFO, "Processing inference request (type: %s, index: %d, size: %zu bytes)",
               req->content_type ? req->content_type : "unknown",
               req->image_index, req->image_size);

        // Start timing
        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);

        // Perform inference based on content type
        cJSON* detections = NULL;
        char* error_msg = NULL;

        if (req->content_type && strcmp(req->content_type, "image/jpeg") == 0) {
            // JPEG inference
            detections = Model_InferenceJPEG(req->image_data, req->image_size,
                                            req->image_index, req->image_width, req->image_height,
                                            &error_msg);
        } else if (req->content_type && strcmp(req->content_type, "application/octet-stream") == 0) {
            // Tensor inference
            int width = Model_GetWidth();
            int height = Model_GetHeight();
            detections = Model_InferenceTensor(req->image_data, width, height,
                                              req->image_index, &error_msg);
        } else {
            error_msg = strdup("Unsupported content type");
            req->status_code = 400;
        }

        // Calculate inference time
        gettimeofday(&end_time, NULL);
        double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                           (end_time.tv_usec - start_time.tv_usec) / 1000.0;

        if (detections) {
            req->response_data = detections;
            req->status_code = (cJSON_GetArraySize(detections) > 0) ? 200 : 204;
            g_server.successful_inferences++;

            // Update timing statistics
            g_server.total_inference_time_ms += elapsed_ms;
            if (g_server.successful_inferences == 1) {
                g_server.min_inference_time_ms = elapsed_ms;
                g_server.max_inference_time_ms = elapsed_ms;
            } else {
                if (elapsed_ms < g_server.min_inference_time_ms) {
                    g_server.min_inference_time_ms = elapsed_ms;
                }
                if (elapsed_ms > g_server.max_inference_time_ms) {
                    g_server.max_inference_time_ms = elapsed_ms;
                }
            }

            // Store latest inference for monitoring (JPEG only, best-effort)
            if (req->content_type && strcmp(req->content_type, "image/jpeg") == 0) {
                char* detections_str = cJSON_PrintUnformatted(detections);
                if (detections_str) {
                    Server_StoreLatestInference(req->image_data, req->image_size, detections_str);
                    free(detections_str);
                }
            }

            syslog(LOG_INFO, "Inference successful: %d detections (%.1f ms)",
                   cJSON_GetArraySize(detections), elapsed_ms);
        } else if (error_msg) {
            // Validation error
            req->response_data = error_msg;
            if (req->status_code == 0) {
                req->status_code = 400;
            }
            g_server.failed_inferences++;
            syslog(LOG_WARNING, "Inference validation failed: %s", error_msg);
        } else {
            // Inference error
            req->status_code = 500;
            g_server.failed_inferences++;
            syslog(LOG_ERR, "Inference failed");
        }

        req->processed = true;
        pthread_cond_signal(&req->done);
        pthread_mutex_unlock(&req->lock);
    }

    syslog(LOG_INFO, "Inference worker thread stopped");
    return NULL;
}

// Initialize server
bool Server_Init(void) {
    memset(&g_server, 0, sizeof(ServerState));

    // Initialize queue
    pthread_mutex_init(&g_server.queue.lock, NULL);
    pthread_cond_init(&g_server.queue.not_empty, NULL);
    pthread_cond_init(&g_server.queue.not_full, NULL);

    // Initialize latest inference cache
    pthread_mutex_init(&g_server.latest.lock, NULL);
    g_server.latest.has_data = false;
    g_server.latest.image_data = NULL;
    g_server.latest.detections_json = NULL;

    // Initialize model
    if (!Model_Setup()) {
        syslog(LOG_ERR, "Failed to initialize model");
        return false;
    }

    // Start worker thread
    g_server.running = true;
    if (pthread_create(&g_server.inference_thread, NULL, inference_worker, NULL) != 0) {
        syslog(LOG_ERR, "Failed to create inference worker thread");
        Model_Cleanup();
        return false;
    }

    syslog(LOG_INFO, "Server initialized successfully");
    return true;
}

// Cleanup server
void Server_Cleanup(void) {
    if (!g_server.running) {
        return;
    }

    syslog(LOG_INFO, "Shutting down server...");

    // Stop worker thread
    pthread_mutex_lock(&g_server.queue.lock);
    g_server.running = false;
    pthread_cond_signal(&g_server.queue.not_empty);
    pthread_mutex_unlock(&g_server.queue.lock);

    pthread_join(g_server.inference_thread, NULL);

    // Cleanup remaining requests
    for (int i = 0; i < g_server.queue.count; i++) {
        int idx = (g_server.queue.head + i) % MAX_QUEUE_SIZE;
        if (g_server.queue.requests[idx]) {
            Server_FreeRequest(g_server.queue.requests[idx]);
        }
    }

    // Cleanup model
    Model_Cleanup();

    // Cleanup latest inference cache
    pthread_mutex_lock(&g_server.latest.lock);
    if (g_server.latest.image_data) {
        free(g_server.latest.image_data);
    }
    if (g_server.latest.detections_json) {
        free(g_server.latest.detections_json);
    }
    pthread_mutex_unlock(&g_server.latest.lock);
    pthread_mutex_destroy(&g_server.latest.lock);

    // Cleanup queue
    pthread_mutex_destroy(&g_server.queue.lock);
    pthread_cond_destroy(&g_server.queue.not_empty);
    pthread_cond_destroy(&g_server.queue.not_full);

    syslog(LOG_INFO, "Server shutdown complete");
}

// Check if server is running
bool Server_IsRunning(void) {
    return g_server.running;
}

// Create a new inference request
InferenceRequest* Server_CreateRequest(const uint8_t* data, size_t size,
                                      const char* content_type, int image_index,
                                      int image_width, int image_height) {
    if (!data || size == 0 || size > MAX_IMAGE_SIZE) {
        syslog(LOG_ERR, "Invalid request parameters (size: %zu)", size);
        return NULL;
    }

    InferenceRequest* req = calloc(1, sizeof(InferenceRequest));
    if (!req) {
        syslog(LOG_ERR, "Failed to allocate request");
        return NULL;
    }

    // Copy image data
    req->image_data = malloc(size);
    if (!req->image_data) {
        syslog(LOG_ERR, "Failed to allocate image buffer");
        free(req);
        return NULL;
    }
    memcpy(req->image_data, data, size);
    req->image_size = size;

    // Copy content type
    if (content_type) {
        req->content_type = strdup(content_type);
    }

    req->image_index = image_index;
    req->image_width = image_width;
    req->image_height = image_height;
    req->processed = false;
    req->status_code = 0;
    req->response_data = NULL;

    pthread_mutex_init(&req->lock, NULL);
    pthread_cond_init(&req->done, NULL);

    return req;
}

// Queue a request for processing
bool Server_QueueRequest(InferenceRequest* request) {
    if (!request) {
        return false;
    }

    pthread_mutex_lock(&g_server.queue.lock);

    // Check if queue is full
    if (g_server.queue.count >= MAX_QUEUE_SIZE) {
        pthread_mutex_unlock(&g_server.queue.lock);
        g_server.busy_responses++;
        syslog(LOG_WARNING, "Queue full, rejecting request");
        return false;
    }

    // Add to queue
    g_server.queue.requests[g_server.queue.tail] = request;
    g_server.queue.tail = (g_server.queue.tail + 1) % MAX_QUEUE_SIZE;
    g_server.queue.count++;
    g_server.total_requests++;

    pthread_cond_signal(&g_server.queue.not_empty);
    pthread_mutex_unlock(&g_server.queue.lock);

    return true;
}

// Free request resources
void Server_FreeRequest(InferenceRequest* request) {
    if (!request) {
        return;
    }

    if (request->image_data) {
        free(request->image_data);
    }
    if (request->content_type) {
        free(request->content_type);
    }
    if (request->response_data) {
        cJSON_Delete((cJSON*)request->response_data);
    }

    pthread_mutex_destroy(&request->lock);
    pthread_cond_destroy(&request->done);

    free(request);
}

// Get server statistics
void Server_GetStats(uint64_t* total, uint64_t* success,
                    uint64_t* failed, uint64_t* busy) {
    if (total) *total = g_server.total_requests;
    if (success) *success = g_server.successful_inferences;
    if (failed) *failed = g_server.failed_inferences;
    if (busy) *busy = g_server.busy_responses;
}

// Get timing statistics
void Server_GetTiming(double* avg_ms, double* min_ms, double* max_ms) {
    if (avg_ms) {
        *avg_ms = (g_server.successful_inferences > 0) ?
                  g_server.total_inference_time_ms / g_server.successful_inferences : 0.0;
    }
    if (min_ms) *min_ms = g_server.min_inference_time_ms;
    if (max_ms) *max_ms = g_server.max_inference_time_ms;
}

// Get current queue size
int Server_GetQueueSize(void) {
    pthread_mutex_lock(&g_server.queue.lock);
    int size = g_server.queue.count;
    pthread_mutex_unlock(&g_server.queue.lock);
    return size;
}

// Check if queue is full
bool Server_IsQueueFull(void) {
    pthread_mutex_lock(&g_server.queue.lock);
    bool full = (g_server.queue.count >= MAX_QUEUE_SIZE);
    pthread_mutex_unlock(&g_server.queue.lock);
    return full;
}

// Store latest inference for monitoring (best-effort, non-blocking)
void Server_StoreLatestInference(const uint8_t* image_data, size_t image_size,
                                 const char* detections_json) {
    if (!image_data || !detections_json || image_size == 0) {
        return;
    }

    pthread_mutex_lock(&g_server.latest.lock);

    // Free old data if exists
    if (g_server.latest.image_data) {
        free(g_server.latest.image_data);
        g_server.latest.image_data = NULL;
    }
    if (g_server.latest.detections_json) {
        free(g_server.latest.detections_json);
        g_server.latest.detections_json = NULL;
    }

    // Store new data (copy)
    g_server.latest.image_data = malloc(image_size);
    if (g_server.latest.image_data) {
        memcpy(g_server.latest.image_data, image_data, image_size);
        g_server.latest.image_size = image_size;
        g_server.latest.detections_json = strdup(detections_json);
        g_server.latest.timestamp = time(NULL);
        g_server.latest.has_data = true;
    } else {
        syslog(LOG_WARNING, "Failed to allocate memory for latest inference cache");
    }

    pthread_mutex_unlock(&g_server.latest.lock);
}

// Get latest inference for monitoring
bool Server_GetLatestInference(uint8_t** image_data, size_t* image_size,
                               char** detections_json, time_t* timestamp) {
    if (!image_data || !image_size || !detections_json || !timestamp) {
        return false;
    }

    pthread_mutex_lock(&g_server.latest.lock);

    if (!g_server.latest.has_data) {
        pthread_mutex_unlock(&g_server.latest.lock);
        return false;
    }

    // Copy data out
    *image_data = malloc(g_server.latest.image_size);
    if (!*image_data) {
        pthread_mutex_unlock(&g_server.latest.lock);
        return false;
    }

    memcpy(*image_data, g_server.latest.image_data, g_server.latest.image_size);
    *image_size = g_server.latest.image_size;
    *detections_json = strdup(g_server.latest.detections_json);
    *timestamp = g_server.latest.timestamp;

    pthread_mutex_unlock(&g_server.latest.lock);
    return true;
}
