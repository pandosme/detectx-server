/**
 * main.c - Inference Server Main Application
 *
 * Implements HTTP API endpoints:
 * - GET  /capabilities    - Model information and requirements
 * - POST /inference/jpeg  - JPEG image inference endpoint
 * - POST /inference/tensor - Pre-processed tensor inference endpoint
 * - GET  /health          - Server health and statistics
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <signal.h>
#include <unistd.h>
#include <glib.h>

#include "ACAP.h"
#include "server.h"
#include "Model.h"
#include "cJSON.h"
#include "labelparse.h"
#include "jpeg_decoder.h"


#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_WARNING, fmt, ## args); printf(fmt, ## args);}
//#define LOG_TRACE(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args); }
#define LOG_TRACE(fmt, args...)    {}

#define APP_PACKAGE	"detectx"

static GMainLoop* main_loop = NULL;

// Signal handler for graceful shutdown
static void signal_handler(int signo) {
    LOG("Received signal %d, shutting down", signo);
    if (main_loop) {
        g_main_loop_quit(main_loop);
    }
}

// Update ACAP status information
static void update_acap_status(void) {
    // Get statistics
    uint64_t total, success, failed, busy;
    Server_GetStats(&total, &success, &failed, &busy);

    double avg_ms, min_ms, max_ms;
    Server_GetTiming(&avg_ms, &min_ms, &max_ms);

    // Update status values
    ACAP_STATUS_SetBool("server", "running", Server_IsRunning());
    ACAP_STATUS_SetNumber("server", "queue_size", Server_GetQueueSize());
    ACAP_STATUS_SetBool("server", "queue_full", Server_IsQueueFull());

    ACAP_STATUS_SetNumber("statistics", "total_requests", (double)total);
    ACAP_STATUS_SetNumber("statistics", "successful", (double)success);
    ACAP_STATUS_SetNumber("statistics", "failed", (double)failed);
    ACAP_STATUS_SetNumber("statistics", "busy", (double)busy);

    double success_rate = (total > 0) ? ((double)success / total) * 100.0 : 0.0;
    ACAP_STATUS_SetNumber("statistics", "success_rate", success_rate);

    ACAP_STATUS_SetNumber("performance", "avg_inference_ms", avg_ms);
    ACAP_STATUS_SetNumber("performance", "min_inference_ms", min_ms);
    ACAP_STATUS_SetNumber("performance", "max_inference_ms", max_ms);

    ACAP_STATUS_SetNumber("model", "input_width", Model_GetWidth());
    ACAP_STATUS_SetNumber("model", "input_height", Model_GetHeight());
}

// GET /capabilities - Return model capabilities and requirements
static void http_capabilities(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    cJSON* resp_json = cJSON_CreateObject();

    // Model information
    cJSON* model = cJSON_CreateObject();
    int model_width = Model_GetWidth();
    int model_height = Model_GetHeight();

    cJSON_AddNumberToObject(model, "input_width", model_width);
    cJSON_AddNumberToObject(model, "input_height", model_height);
    cJSON_AddNumberToObject(model, "channels", 3);
    cJSON_AddStringToObject(model, "aspect_ratio", "1:1");

    // Supported input formats
    cJSON* formats = cJSON_CreateArray();

    // JPEG format
    cJSON* jpeg_format = cJSON_CreateObject();
    cJSON_AddStringToObject(jpeg_format, "endpoint", "/inference-jpeg");
    cJSON_AddStringToObject(jpeg_format, "method", "POST");
    cJSON_AddStringToObject(jpeg_format, "content_type", "image/jpeg");
    cJSON_AddStringToObject(jpeg_format, "description", "JPEG image (any resolution, square aspect recommended)");
    cJSON_AddStringToObject(jpeg_format, "preprocessing", "letterbox");
    cJSON_AddNumberToObject(jpeg_format, "max_size_mb", MAX_IMAGE_SIZE / (1024 * 1024));
    cJSON_AddItemToArray(formats, jpeg_format);

    // Tensor format (optimal)
    cJSON* tensor_format = cJSON_CreateObject();
    cJSON_AddStringToObject(tensor_format, "endpoint", "/inference-tensor");
    cJSON_AddStringToObject(tensor_format, "method", "POST");
    cJSON_AddStringToObject(tensor_format, "content_type", "application/octet-stream");
    cJSON_AddStringToObject(tensor_format, "description", "Raw RGB tensor data (pre-processed)");
    cJSON_AddStringToObject(tensor_format, "format", "RGB interleaved (RGBRGBRGB...)");

    char size_desc[128];
    snprintf(size_desc, sizeof(size_desc), "Must be exactly %d x %d x 3 = %d bytes",
             model_width, model_height, model_width * model_height * 3);
    cJSON_AddStringToObject(tensor_format, "size_requirement", size_desc);
    cJSON_AddBoolToObject(tensor_format, "strict_dimensions", true);
    cJSON_AddItemToArray(formats, tensor_format);

    cJSON_AddItemToObject(model, "input_formats", formats);

    // Class labels
    cJSON* classes = cJSON_CreateArray();
    char** labels = NULL;
    int num_classes = 0;

    if (labelparse_get_labels(&labels, &num_classes)) {
        for (int i = 0; i < num_classes; i++) {
            cJSON* class_obj = cJSON_CreateObject();
            cJSON_AddNumberToObject(class_obj, "id", i);
            cJSON_AddStringToObject(class_obj, "name", labels[i]);
            cJSON_AddItemToArray(classes, class_obj);
        }
    }

    cJSON_AddItemToObject(model, "classes", classes);
    cJSON_AddNumberToObject(model, "max_queue_size", MAX_QUEUE_SIZE);
    cJSON_AddItemToObject(resp_json, "model", model);

    // Server information
    cJSON_AddStringToObject(resp_json, "server", "detectx");
    cJSON_AddStringToObject(resp_json, "version", "1.0.0");

    ACAP_HTTP_Respond_JSON(response, resp_json);
    cJSON_Delete(resp_json);
}

// GET /health - Return server health and statistics
static void http_health(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    // Update ACAP status
    update_acap_status();

    cJSON* resp_json = cJSON_CreateObject();

    cJSON_AddBoolToObject(resp_json, "running", Server_IsRunning());
    cJSON_AddNumberToObject(resp_json, "queue_size", Server_GetQueueSize());
    cJSON_AddBoolToObject(resp_json, "queue_full", Server_IsQueueFull());

    // Statistics
    uint64_t total, success, failed, busy;
    Server_GetStats(&total, &success, &failed, &busy);

    cJSON* stats = cJSON_CreateObject();
    cJSON_AddNumberToObject(stats, "total_requests", (double)total);
    cJSON_AddNumberToObject(stats, "successful", (double)success);
    cJSON_AddNumberToObject(stats, "failed", (double)failed);
    cJSON_AddNumberToObject(stats, "busy", (double)busy);
    cJSON_AddItemToObject(resp_json, "statistics", stats);

    // Timing statistics
    double avg_ms, min_ms, max_ms;
    Server_GetTiming(&avg_ms, &min_ms, &max_ms);

    cJSON* timing = cJSON_CreateObject();
    cJSON_AddNumberToObject(timing, "average_ms", avg_ms);
    cJSON_AddNumberToObject(timing, "min_ms", min_ms);
    cJSON_AddNumberToObject(timing, "max_ms", max_ms);
    cJSON_AddItemToObject(resp_json, "timing", timing);

    ACAP_HTTP_Respond_JSON(response, resp_json);
    cJSON_Delete(resp_json);
}

// Helper function to process inference request and send response
static void process_and_respond(ACAP_HTTP_Response response, InferenceRequest* request) {
    // Wait for processing to complete
    pthread_mutex_lock(&request->lock);
    while (!request->processed) {
        pthread_cond_wait(&request->done, &request->lock);
    }
    pthread_mutex_unlock(&request->lock);

    // Send response based on status
    if (request->status_code == 200) {
        // Detections found
        cJSON* resp_json = cJSON_CreateObject();
        cJSON_AddItemToObject(resp_json, "detections", (cJSON*)request->response_data);
        request->response_data = NULL; // Transfer ownership

        ACAP_HTTP_Respond_JSON(response, resp_json);
        cJSON_Delete(resp_json);
    } else if (request->status_code == 204) {
        // No detections
        if (request->response_data) {
            cJSON_Delete((cJSON*)request->response_data);
            request->response_data = NULL;
        }
        ACAP_HTTP_Respond_Error(response, 204, "No Content: No detections found");
    } else if (request->status_code == 400 && request->response_data) {
        // Validation error with message
        char* error_msg = (char*)request->response_data;
        char full_msg[512];
        snprintf(full_msg, sizeof(full_msg), "Bad Request: %s", error_msg);
        ACAP_HTTP_Respond_Error(response, 400, full_msg);
        free(error_msg);
        request->response_data = NULL;
    } else {
        // Inference failed
        if (request->response_data) {
            cJSON_Delete((cJSON*)request->response_data);
            request->response_data = NULL;
        }
        ACAP_HTTP_Respond_Error(response, 500, "Internal Server Error: Inference failed");
    }

    // Cleanup
    Server_FreeRequest(request);
}

// POST /inference/jpeg - Process JPEG image inference
static void http_inference_jpeg(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    const char* content_type = request->contentType;

    // Validate content type
    if (!content_type || strncmp(content_type, "image/jpeg", 10) != 0) {
        ACAP_HTTP_Respond_Error(response, 400, "Bad Request: Content-Type must be image/jpeg");
        return;
    }

    // Get image index from query string (optional, format: ?index=N)
    int image_index = -1;
    if (request->queryString) {
        const char* index_param = strstr(request->queryString, "index=");
        if (index_param) {
            image_index = atoi(index_param + 6);
        }
    }

    // Read request body
    const uint8_t* body_data = (const uint8_t*)request->postData;
    size_t body_size = request->postDataLength;

    if (!body_data || body_size == 0) {
        ACAP_HTTP_Respond_Error(response, 400, "Bad Request: Empty body");
        return;
    }

    if (body_size > MAX_IMAGE_SIZE) {
        ACAP_HTTP_Respond_Error(response, 413, "Payload Too Large: Maximum size is 10MB");
        return;
    }

    // Check if queue is full
    if (Server_IsQueueFull()) {
        ACAP_HTTP_Respond_Error(response, 503, "Service Unavailable: Queue full (max 3 concurrent requests)");
        return;
    }

    // Get JPEG dimensions
    int image_width, image_height;
    if (!JPEG_GetDimensions(body_data, body_size, &image_width, &image_height)) {
        ACAP_HTTP_Respond_Error(response, 400, "Bad Request: Invalid JPEG image");
        return;
    }

    // Create inference request
    InferenceRequest* inf_request = Server_CreateRequest(body_data, body_size,
                                                         "image/jpeg", image_index,
                                                         image_width, image_height);
    if (!inf_request) {
        ACAP_HTTP_Respond_Error(response, 500, "Internal Server Error: Failed to create request");
        return;
    }

    // Queue request
    if (!Server_QueueRequest(inf_request)) {
        Server_FreeRequest(inf_request);
        ACAP_HTTP_Respond_Error(response, 503, "Service Unavailable: Queue full");
        return;
    }

    process_and_respond(response, inf_request);
}

// POST /inference/tensor - Process pre-processed tensor inference
static void http_inference_tensor(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    const char* content_type = request->contentType;

    // Validate content type
    if (!content_type || strncmp(content_type, "application/octet-stream", 24) != 0) {
        ACAP_HTTP_Respond_Error(response, 400, "Bad Request: Content-Type must be application/octet-stream");
        return;
    }

    // Get image index from query string (optional, format: ?index=N)
    int image_index = -1;
    if (request->queryString) {
        const char* index_param = strstr(request->queryString, "index=");
        if (index_param) {
            image_index = atoi(index_param + 6);
        }
    }

    // Read request body
    const uint8_t* body_data = (const uint8_t*)request->postData;
    size_t body_size = request->postDataLength;

    if (!body_data || body_size == 0) {
        ACAP_HTTP_Respond_Error(response, 400, "Bad Request: Empty body");
        return;
    }

    // Validate tensor size
    int expected_size = Model_GetWidth() * Model_GetHeight() * 3;
    if (body_size != expected_size) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg),
                 "Bad Request: Invalid tensor size. Expected %d bytes (%dx%dx3), got %zu bytes",
                 expected_size, Model_GetWidth(), Model_GetHeight(), body_size);
        ACAP_HTTP_Respond_Error(response, 400, error_msg);
        return;
    }

    // Check if queue is full
    if (Server_IsQueueFull()) {
        ACAP_HTTP_Respond_Error(response, 503, "Service Unavailable: Queue full (max 3 concurrent requests)");
        return;
    }

    // For tensor input, dimensions are model dimensions
    int tensor_width = Model_GetWidth();
    int tensor_height = Model_GetHeight();

    // Create inference request
    InferenceRequest* inf_request = Server_CreateRequest(body_data, body_size,
                                                         "application/octet-stream", image_index,
                                                         tensor_width, tensor_height);
    if (!inf_request) {
        ACAP_HTTP_Respond_Error(response, 500, "Internal Server Error: Failed to create request");
        return;
    }

    // Queue request
    if (!Server_QueueRequest(inf_request)) {
        Server_FreeRequest(inf_request);
        ACAP_HTTP_Respond_Error(response, 503, "Service Unavailable: Queue full");
        return;
    }

    process_and_respond(response, inf_request);
}

// GET /monitor - Serve monitoring HTML page
static void http_monitor(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    // Read the HTML file
    FILE* fp = ACAP_FILE_Open("html/monitor.html", "r");
    if (!fp) {
        ACAP_HTTP_Respond_Error(response, 404, "Monitoring page not found");
        return;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // Read file content
    char* html_content = malloc(file_size + 1);
    if (!html_content) {
        fclose(fp);
        ACAP_HTTP_Respond_Error(response, 500, "Memory allocation failed");
        return;
    }

    fread(html_content, 1, file_size, fp);
    html_content[file_size] = '\0';
    fclose(fp);

    // Send HTML response
    FCGX_FPrintF(response->out, "HTTP/1.1 200 OK\r\n");
    FCGX_FPrintF(response->out, "Content-Type: text/html; charset=utf-8\r\n");
    FCGX_FPrintF(response->out, "Content-Length: %ld\r\n", file_size);
    FCGX_FPrintF(response->out, "\r\n");
    FCGX_FPrintF(response->out, "%s", html_content);

    free(html_content);
}

// GET /monitor-latest - Return latest inference data as JSON
static void http_monitor_latest(ACAP_HTTP_Response response, const ACAP_HTTP_Request request) {
    uint8_t* image_data = NULL;
    size_t image_size = 0;
    char* detections_json = NULL;
    time_t timestamp = 0;

    // Get latest inference
    if (!Server_GetLatestInference(&image_data, &image_size, &detections_json, &timestamp)) {
        ACAP_HTTP_Respond_Error(response, 404, "No inference data available yet");
        return;
    }

    // Base64 encode the JPEG image using glib
    gchar* image_base64 = g_base64_encode(image_data, image_size);
    if (!image_base64) {
        free(image_data);
        free(detections_json);
        ACAP_HTTP_Respond_Error(response, 500, "Failed to encode image");
        return;
    }

    // Create response JSON
    cJSON* resp = cJSON_CreateObject();
    cJSON_AddStringToObject(resp, "image", image_base64);

    // Parse detections_json string and add as array
    cJSON* detections_array = cJSON_Parse(detections_json);
    if (detections_array) {
        cJSON_AddItemToObject(resp, "detections", detections_array);
    } else {
        cJSON_AddArrayToObject(resp, "detections");  // Empty array if parsing fails
    }

    cJSON_AddNumberToObject(resp, "timestamp", (double)timestamp);

    // Send JSON response
    ACAP_HTTP_Respond_JSON(response, resp);

    // Cleanup
    free(image_data);
    g_free(image_base64);
    free(detections_json);
    cJSON_Delete(resp);
}

int main(void) {
    // Initialize logging
    openlog(APP_PACKAGE, LOG_PID, LOG_USER);
    LOG("-------------- %s --------------\n", APP_PACKAGE);

    // Initialize ACAP framework
    if (!ACAP("detectx", NULL)) {
        LOG_WARN("Failed to initialize ACAP");
        return 1;
    }

    // Initialize server
    if (!Server_Init()) {
        LOG_WARN("Failed to initialize server");
        ACAP_Cleanup();
        return 1;
    }

    // Register HTTP endpoints
    ACAP_HTTP_Node("capabilities", http_capabilities);
    ACAP_HTTP_Node("inference-jpeg", http_inference_jpeg);
    ACAP_HTTP_Node("inference-tensor", http_inference_tensor);
    ACAP_HTTP_Node("health", http_health);
    ACAP_HTTP_Node("monitor", http_monitor);
    ACAP_HTTP_Node("monitor-latest", http_monitor_latest);

    // Initialize ACAP status
    update_acap_status();

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Start event loop
    main_loop = g_main_loop_new(NULL, FALSE);
    LOG("Server running, waiting for requests...");

    g_main_loop_run(main_loop);

    // Cleanup
    LOG("Cleaning up...");
    g_main_loop_unref(main_loop);
    Server_Cleanup();
    ACAP_Cleanup();

    LOG("Server stopped");
    closelog();

    return 0;
}
