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
    int image_width;        // Original received image width
    int image_height;       // Original received image height
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

// Latest inference cache (for monitoring)
typedef struct {
    uint8_t* image_data;       // JPEG image data
    size_t image_size;
    char* detections_json;     // JSON string of detections
    time_t timestamp;
    pthread_mutex_t lock;
    bool has_data;
} LatestInference;

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

    // Latest inference cache
    LatestInference latest;
} ServerState;

// Server lifecycle
bool Server_Init(void);
void Server_Cleanup(void);
bool Server_IsRunning(void);

// Request processing
InferenceRequest* Server_CreateRequest(const uint8_t* data, size_t size,
                                      const char* content_type, int image_index,
                                      int image_width, int image_height);
bool Server_QueueRequest(InferenceRequest* request);
void Server_FreeRequest(InferenceRequest* request);

// Statistics
void Server_GetStats(uint64_t* total, uint64_t* success,
                    uint64_t* failed, uint64_t* busy);
void Server_GetTiming(double* avg_ms, double* min_ms, double* max_ms);

// Queue status
int Server_GetQueueSize(void);
bool Server_IsQueueFull(void);

// Latest inference cache (for monitoring)
void Server_StoreLatestInference(const uint8_t* image_data, size_t image_size,
                                 const char* detections_json);
bool Server_GetLatestInference(uint8_t** image_data, size_t* image_size,
                               char** detections_json, time_t* timestamp);

#endif // SERVER_H
