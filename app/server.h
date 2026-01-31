/**
 * server.h - Inference Server Core
 *
 * Manages HTTP endpoints, request queue, and inference coordination
 */

#ifndef SERVER_H
#define SERVER_H

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include "cJSON.h"

// Configuration
#define MAX_QUEUE_SIZE 3
#define MAX_IMAGE_SIZE (10 * 1024 * 1024)  // 10MB max image size

// Request queue structures
typedef struct {
    uint8_t* image_data;
    size_t image_size;
    int image_index;        // For dataset validation (-1 if not specified)
    char* content_type;
    void* response_data;    // Will hold cJSON* response
    int status_code;
    pthread_mutex_t lock;
    pthread_cond_t done;
    bool processed;
} InferenceRequest;

typedef struct {
    InferenceRequest* requests[MAX_QUEUE_SIZE];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
} RequestQueue;

// Server state
typedef struct {
    bool running;
    pthread_t inference_thread;
    RequestQueue queue;

    // Statistics
    uint64_t total_requests;
    uint64_t successful_inferences;
    uint64_t failed_inferences;
    uint64_t busy_responses;

    // Timing statistics (in milliseconds)
    double total_inference_time_ms;
    double min_inference_time_ms;
    double max_inference_time_ms;
} ServerState;

// Server lifecycle
bool Server_Init(void);
void Server_Cleanup(void);
bool Server_IsRunning(void);

// Request processing
InferenceRequest* Server_CreateRequest(const uint8_t* data, size_t size,
                                      const char* content_type, int image_index);
bool Server_QueueRequest(InferenceRequest* request);
void Server_FreeRequest(InferenceRequest* request);

// Statistics
void Server_GetStats(uint64_t* total, uint64_t* success,
                    uint64_t* failed, uint64_t* busy);
void Server_GetTiming(double* avg_ms, double* min_ms, double* max_ms);

// Queue status
int Server_GetQueueSize(void);
bool Server_IsQueueFull(void);

#endif // SERVER_H
