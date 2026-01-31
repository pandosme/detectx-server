/**
 * Model.c - Inference Server Model Implementation
 *
 * Adapted from DetectX for server use (no VDO, accepts JPEG/tensor input)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <errno.h>

#include "larod.h"
#include "ACAP.h"
#include "Model.h"
#include "jpeg_decoder.h"
#include "labelparse.h"
#include "model_params.h"
#include "cJSON.h"

#define LOG(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_WARN(fmt, args...)    { syslog(LOG_INFO, fmt, ## args); printf(fmt, ## args);}
#define LOG_TRACE(fmt, args...)   {}

// Helper function prototypes
static bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* fd);
static float iou(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2);
static cJSON* non_maximum_suppression(cJSON* list);
static cJSON* format_detections_for_api(cJSON* raw_detections, int image_index);
static uint8_t* preprocess_rgb_letterbox(const uint8_t* rgb_in, int in_w, int in_h,
                                         int out_w, int out_h);
static int get_class_id_from_label(const char* label);

// Model dimensions and parameters
static unsigned int modelWidth = 640;
static unsigned int modelHeight = 640;
static unsigned int channels = 3;
static unsigned int boxes = 0;
static unsigned int classes = 0;
static size_t inputs = 1;
static size_t outputs = 1;
static float quant = 1.0;
static float quant_zero = 0;
static float objectnessThreshold = 0.25;
static float confidenceThreshold = 0.30;
static float nms = 0.05;

// Letterbox transformation tracking (for bbox coordinate conversion)
static struct {
    int original_width;
    int original_height;
    float scale;
    int offset_x;
    int offset_y;
} letterbox_params = {0};

// Larod handles
static int larodModelFd = -1;
static larodConnection* conn = NULL;
static larodModel* InfModel = NULL;
static larodJobRequest* infReq = NULL;
static void* larodInputAddr = MAP_FAILED;
static void* larodOutput1Addr = MAP_FAILED;
static int larodInputFd = -1;
static int larodOutput1Fd = -1;
static larodTensor** inputTensors = NULL;
static larodTensor** outputTensors = NULL;
static size_t outputBufferSize = 0;

// Labels
static char** modelLabels = NULL;
static size_t numLabels = 0;
static const char* MODEL_PATH = "model/model.tflite";
static const char* LABELS_PATH = "model/labels.txt";

// Temp file patterns
static char OBJECT_DETECTOR_INPUT_FILE_PATTERN[] = "/tmp/larod.in.test-XXXXXX";
static char OBJECT_DETECTOR_OUT1_FILE_PATTERN[]  = "/tmp/larod.out1.test-XXXXXX";

// Reference ID counter
static int currentRefId = 0;

//-----------------------------------------------------------------------------
// Model Setup
//-----------------------------------------------------------------------------

bool Model_Setup(void) {
    larodError* error = NULL;

    // Connect to larod
    if (!larodConnect(&conn, &error)) {
        LOG_WARN("%s: Could not connect to larod\n", __func__);
        larodClearError(&error);
        return false;
    }

    // Open model file
    larodModelFd = open(MODEL_PATH, O_RDONLY);
    if (larodModelFd < 0) {
        LOG_WARN("%s: Could not open model %s: %s\n", __func__, MODEL_PATH, strerror(errno));
        Model_Cleanup();
        return false;
    }

    // Enumerate available devices and select the best one
    const char* chipString = NULL;
    const larodDevice* device = NULL;

    // List all available devices
    size_t numDevices = 0;
    const larodDevice** deviceList = larodListDevices(conn, &numDevices, &error);
    if (!deviceList) {
        LOG_WARN("%s: Could not list devices: %s\n", __func__,
                 error ? error->msg : "unknown");
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    LOG("Available larod devices: %zu\n", numDevices);
    for (size_t i = 0; i < numDevices; i++) {
        const char* name = larodGetDeviceName(deviceList[i], &error);
        if (name) {
            LOG("  [%zu] %s\n", i, name);
        }
        larodClearError(&error);
    }

    // Preferred device order (hardware accelerators first, CPU last)
    const char* preferredDevices[] = {
        "a9-dlpu-tflite",           // ARTPEC-9 DLPU
        "axis-a9-dlpu-tflite",      // Alternative ARTPEC-9 naming
        "axis-a8-dlpu-tflite",      // ARTPEC-8 DLPU
        "ambarella-cvflow-tflite",  // Ambarella CV25
        "google-edge-tpu-tflite",   // Google Coral TPU
        "cpu-tflite"                // CPU fallback
    };

    // Try each preferred device in order
    for (size_t p = 0; p < sizeof(preferredDevices) / sizeof(preferredDevices[0]); p++) {
        for (size_t i = 0; i < numDevices; i++) {
            const char* name = larodGetDeviceName(deviceList[i], &error);
            if (name && strcmp(name, preferredDevices[p]) == 0) {
                device = deviceList[i];
                chipString = name;
                LOG("Selected device: %s\n", chipString);
                larodClearError(&error);
                goto device_selected;
            }
            larodClearError(&error);
        }
    }

device_selected:
    if (!device && numDevices > 0) {
        // Fallback to first available device
        device = deviceList[0];
        chipString = larodGetDeviceName(device, &error);
        LOG("Using fallback device: %s\n", chipString ? chipString : "unknown");
        larodClearError(&error);
    }

    if (!device) {
        LOG_WARN("%s: No larod devices available\n", __func__);
        free(deviceList);
        Model_Cleanup();
        return false;
    }

    // Load model
    InfModel = larodLoadModel(conn, larodModelFd, device, LAROD_ACCESS_PRIVATE,
                             "object_detection", NULL, &error);

    // Clean up device list (just free the array, devices are managed by larod)
    free(deviceList);

    if (!InfModel) {
        LOG_WARN("%s: Unable to load model: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    // Create model tensors for introspection
    larodTensor** tempInputTensors = larodCreateModelInputs(InfModel, &inputs, &error);
    if (!tempInputTensors) {
        LOG_WARN("%s: Failed retrieving input tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    larodTensor** tempOutputTensors = larodCreateModelOutputs(InfModel, &outputs, &error);
    if (!tempOutputTensors) {
        LOG_WARN("%s: Failed retrieving output tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        Model_Cleanup();
        return false;
    }

    // Get input dimensions (NHWC: batch, height, width, channels)
    const larodTensorDims* inputDims = larodGetTensorDims(tempInputTensors[0], &error);
    if (!inputDims) {
        LOG_WARN("%s: Failed to get input tensor dimensions\n", __func__);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        larodDestroyTensors(conn, &tempOutputTensors, outputs, NULL);
        Model_Cleanup();
        return false;
    }

    modelHeight = inputDims->dims[1];
    modelWidth = inputDims->dims[2];
    channels = inputDims->dims[3];

    LOG("Model input: %ux%ux%u\n", modelWidth, modelHeight, channels);

    // Get output dimensions (YOLOv5: [batch, boxes, stride])
    const larodTensorDims* outputDims = larodGetTensorDims(tempOutputTensors[0], &error);
    if (!outputDims) {
        LOG_WARN("%s: Failed to get output tensor dimensions\n", __func__);
        larodDestroyTensors(conn, &tempInputTensors, inputs, NULL);
        larodDestroyTensors(conn, &tempOutputTensors, outputs, NULL);
        Model_Cleanup();
        return false;
    }

    boxes = outputDims->dims[1];
    int stride = outputDims->dims[2];
    classes = stride - 5;  // YOLOv5 format: x,y,w,h,objectness,class1...classN

    LOG("Model output: %u boxes, %u classes, stride=%d\n", boxes, classes, stride);

    // Get quantization parameters
    larodTensorDataType dataType = larodGetTensorDataType(tempOutputTensors[0], &error);
    if (dataType == LAROD_TENSOR_DATA_TYPE_INT8 || dataType == LAROD_TENSOR_DATA_TYPE_UINT8) {
        quant = QUANTIZATION_SCALE;
        quant_zero = QUANTIZATION_ZERO_POINT;
        LOG("Quantized model: data_type=%d, scale=%.15f, zero_point=%d\n",
            dataType, quant, (int)quant_zero);
    } else {
        quant = 1.0;
        quant_zero = 0;
        LOG("Float model detected (data_type=%d)\n", dataType);
    }

    // Clean up temporary tensors
    larodDestroyTensors(conn, &tempInputTensors, inputs, &error);
    larodDestroyTensors(conn, &tempOutputTensors, outputs, &error);

    // Read settings
    cJSON* settings = ACAP_Get_Config("settings");
    if (settings) {
        cJSON* model_settings = cJSON_GetObjectItem(settings, "model");
        if (model_settings) {
            cJSON* nmsItem = cJSON_GetObjectItem(model_settings, "nms");
            if (nmsItem) nms = nmsItem->valuedouble;

            cJSON* objectnessItem = cJSON_GetObjectItem(model_settings, "objectness");
            if (objectnessItem) objectnessThreshold = objectnessItem->valuedouble;

            cJSON* confidenceItem = cJSON_GetObjectItem(model_settings, "confidence");
            if (confidenceItem) confidenceThreshold = confidenceItem->valuedouble;
        }
    }

    LOG("Thresholds: objectness=%.2f, confidence=%.2f, nms=%.2f\n",
        objectnessThreshold, confidenceThreshold, nms);

    // Load labels
    if (!labelparse_get_labels(&modelLabels, (int*)&numLabels)) {
        LOG_WARN("%s: Failed to load labels from %s\n", __func__, LABELS_PATH);
        Model_Cleanup();
        return false;
    }

    LOG("Loaded %zu labels\n", numLabels);

    // Initialize letterbox params to identity (no transformation by default)
    letterbox_params.original_width = modelWidth;
    letterbox_params.original_height = modelHeight;
    letterbox_params.scale = 1.0f;
    letterbox_params.offset_x = 0;
    letterbox_params.offset_y = 0;

    // Create input/output buffers
    size_t inputBufferSize = modelWidth * modelHeight * channels;
    outputBufferSize = boxes * (5 + classes);  // Each box has x,y,w,h,obj + classes

    if (!createAndMapTmpFile(OBJECT_DETECTOR_INPUT_FILE_PATTERN, inputBufferSize,
                            &larodInputAddr, &larodInputFd)) {
        LOG_WARN("%s: Failed to create input buffer\n", __func__);
        Model_Cleanup();
        return false;
    }

    if (!createAndMapTmpFile(OBJECT_DETECTOR_OUT1_FILE_PATTERN, outputBufferSize,
                            &larodOutput1Addr, &larodOutput1Fd)) {
        LOG_WARN("%s: Failed to create output buffer\n", __func__);
        Model_Cleanup();
        return false;
    }

    // Create larod tensors
    inputTensors = larodCreateModelInputs(InfModel, &inputs, &error);
    if (!inputTensors) {
        LOG_WARN("%s: Failed to create input tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    outputTensors = larodCreateModelOutputs(InfModel, &outputs, &error);
    if (!outputTensors) {
        LOG_WARN("%s: Failed to create output tensors: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    // Set tensor file descriptors
    if (!larodSetTensorFd(inputTensors[0], larodInputFd, &error)) {
        LOG_WARN("%s: Failed to set input tensor fd: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    if (!larodSetTensorFd(outputTensors[0], larodOutput1Fd, &error)) {
        LOG_WARN("%s: Failed to set output tensor fd: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    // Create inference job request
    infReq = larodCreateJobRequest(InfModel, inputTensors, inputs,
                                   outputTensors, outputs, NULL, &error);
    if (!infReq) {
        LOG_WARN("%s: Failed to create job request: %s\n", __func__, error->msg);
        larodClearError(&error);
        Model_Cleanup();
        return false;
    }

    LOG("Model setup complete\n");
    return true;
}

//-----------------------------------------------------------------------------
// Model Cleanup
//-----------------------------------------------------------------------------

void Model_Cleanup(void) {
    larodError* error = NULL;

    if (infReq) {
        larodDestroyJobRequest(&infReq);
        infReq = NULL;
    }

    if (inputTensors) {
        larodDestroyTensors(conn, &inputTensors, inputs, &error);
        inputTensors = NULL;
    }

    if (outputTensors) {
        larodDestroyTensors(conn, &outputTensors, outputs, &error);
        outputTensors = NULL;
    }

    if (InfModel) {
        larodDestroyModel(&InfModel);
        InfModel = NULL;
    }

    if (larodInputAddr != MAP_FAILED) {
        munmap(larodInputAddr, modelWidth * modelHeight * channels);
        larodInputAddr = MAP_FAILED;
    }

    if (larodOutput1Addr != MAP_FAILED) {
        munmap(larodOutput1Addr, outputBufferSize);
        larodOutput1Addr = MAP_FAILED;
    }

    if (larodInputFd >= 0) {
        close(larodInputFd);
        larodInputFd = -1;
    }

    if (larodOutput1Fd >= 0) {
        close(larodOutput1Fd);
        larodOutput1Fd = -1;
    }

    if (larodModelFd >= 0) {
        close(larodModelFd);
        larodModelFd = -1;
    }

    if (conn) {
        larodDisconnect(&conn, NULL);
        conn = NULL;
    }

    LOG("Model cleanup complete\n");
}

//-----------------------------------------------------------------------------
// Accessor Functions
//-----------------------------------------------------------------------------

int Model_GetWidth(void) {
    return (int)modelWidth;
}

int Model_GetHeight(void) {
    return (int)modelHeight;
}

//-----------------------------------------------------------------------------
// Continue in next file due to length...
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Inference Functions
//-----------------------------------------------------------------------------

cJSON* Model_InferenceTensor(const uint8_t* rgb_data, int width, int height,
                             int image_index, char** error_msg) {
    if (error_msg) *error_msg = NULL;

    // Validate dimensions
    if (width != (int)modelWidth || height != (int)modelHeight) {
        if (error_msg) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Invalid dimensions: expected %dx%d, got %dx%d",
                     modelWidth, modelHeight, width, height);
            *error_msg = strdup(buf);
        }
        return NULL;
    }

    // Copy RGB data directly to larod input tensor
    size_t data_size = width * height * 3;
    memcpy(larodInputAddr, rgb_data, data_size);

    // Note: letterbox_params are already set by caller or initialized to identity in Model_Setup
    // Do NOT overwrite them here, as Model_InferenceJPEG relies on preprocess_rgb_letterbox
    // setting the correct transformation parameters before calling this function

    // Run inference
    larodError* error = NULL;
    if (lseek(larodOutput1Fd, 0, SEEK_SET) == -1) {
        LOG_WARN("%s: Unable to rewind output file: %s\n", __func__, strerror(errno));
        if (error_msg) *error_msg = strdup("Failed to prepare output buffer");
        return NULL;
    }

    if (!larodRunJob(conn, infReq, &error)) {
        LOG_WARN("%s: Inference failed: %s\n", __func__, error->msg);
        if (error_msg) *error_msg = strdup("Inference execution failed");
        larodClearError(&error);
        return NULL;
    }

    // Parse inference results
    uint8_t* output_tensor = (uint8_t*)larodOutput1Addr;
    cJSON* raw_detections = cJSON_CreateArray();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long timestamp = tv.tv_sec * 1000LL + tv.tv_usec / 1000;

    int detections = 0;
    for (unsigned int i = 0; i < boxes; i++) {
        int box = i * (5 + classes);

        // Dequantize objectness
        float objectness = (float)(output_tensor[box + 4] - quant_zero) * quant;

        if (objectness >= objectnessThreshold) {
            float x = (float)(output_tensor[box + 0] - quant_zero) * quant;
            float y = (float)(output_tensor[box + 1] - quant_zero) * quant;
            float w = (float)(output_tensor[box + 2] - quant_zero) * quant;
            float h = (float)(output_tensor[box + 3] - quant_zero) * quant;

            // Find best class
            int classId = -1;
            float maxConfidence = 0;
            for (unsigned int c = 0; c < classes; c++) {
                float confidence = (float)(output_tensor[box + 5 + c] - quant_zero) * quant * objectness;
                if (confidence > maxConfidence) {
                    classId = c;
                    maxConfidence = confidence;
                }
            }

            if (maxConfidence > confidenceThreshold) {
                detections++;
                cJSON* detection = cJSON_CreateObject();
                const char* label = labels_get(modelLabels, numLabels, classId);
                cJSON_AddStringToObject(detection, "label", label);
                cJSON_AddNumberToObject(detection, "c", maxConfidence);

                // Convert to top-left corner format
                double x_norm = x - (w / 2);
                double y_norm = y - (h / 2);

                cJSON_AddNumberToObject(detection, "x", x_norm);
                cJSON_AddNumberToObject(detection, "y", y_norm);
                cJSON_AddNumberToObject(detection, "w", w);
                cJSON_AddNumberToObject(detection, "h", h);
                cJSON_AddNumberToObject(detection, "timestamp", timestamp);
                cJSON_AddNumberToObject(detection, "refId", currentRefId++);
                cJSON_AddItemToArray(raw_detections, detection);
            }
        }
    }

    LOG("Found %d detections before NMS\n", detections);

    // Apply NMS
    cJSON* nms_detections = non_maximum_suppression(raw_detections);
    cJSON_Delete(raw_detections);

    // Format for API
    cJSON* formatted = format_detections_for_api(nms_detections, image_index);
    cJSON_Delete(nms_detections);

    return formatted;
}

cJSON* Model_InferenceJPEG(const uint8_t* jpeg_data, size_t jpeg_size,
                           int image_index, char** error_msg) {
    if (error_msg) *error_msg = NULL;

    // Decode JPEG
    DecodedImage img;
    if (!JPEG_Decode(jpeg_data, jpeg_size, &img)) {
        if (error_msg) *error_msg = strdup("Failed to decode JPEG image");
        return NULL;
    }

    LOG("Decoded JPEG: %dx%d\n", img.width, img.height);

    // Check aspect ratio (warning only)
    float aspect = (float)img.width / (float)img.height;
    if (aspect < 0.9 || aspect > 1.1) {
        LOG_WARN("Non-square image: %dx%d (aspect %.2f). Letterboxing applied.",
               img.width, img.height, aspect);
    }

    // Preprocess RGB with letterboxing
    uint8_t* preprocessed_rgb = preprocess_rgb_letterbox(
        img.data, img.width, img.height, modelWidth, modelHeight
    );

    JPEG_FreeImage(&img);

    if (!preprocessed_rgb) {
        if (error_msg) *error_msg = strdup("Preprocessing failed");
        return NULL;
    }

    // Run inference with preprocessed data
    cJSON* result = Model_InferenceTensor(preprocessed_rgb, modelWidth, modelHeight,
                                         image_index, error_msg);

    free(preprocessed_rgb);

    return result;
}

//-----------------------------------------------------------------------------
// Helper Functions
//-----------------------------------------------------------------------------

static uint8_t* preprocess_rgb_letterbox(const uint8_t* rgb_in, int in_w, int in_h,
                                         int out_w, int out_h) {
    uint8_t* out = calloc(out_w * out_h * 3, 1);  // Black background
    if (!out) {
        LOG_WARN("%s: Failed to allocate output buffer\n", __func__);
        return NULL;
    }

    // Calculate scale to fit inside output while preserving aspect ratio
    float scale = fminf((float)out_w / in_w, (float)out_h / in_h);
    int scaled_w = (int)(in_w * scale);
    int scaled_h = (int)(in_h * scale);

    // Center the image
    int offset_x = (out_w - scaled_w) / 2;
    int offset_y = (out_h - scaled_h) / 2;

    // Store letterbox parameters for bbox transformation
    letterbox_params.original_width = in_w;
    letterbox_params.original_height = in_h;
    letterbox_params.scale = scale;
    letterbox_params.offset_x = offset_x;
    letterbox_params.offset_y = offset_y;

    LOG_TRACE("Letterbox: %dx%d -> %dx%d (scale %.3f, offset %d,%d)\n",
              in_w, in_h, scaled_w, scaled_h, scale, offset_x, offset_y);

    // Nearest-neighbor scaling
    for (int y = 0; y < scaled_h; y++) {
        for (int x = 0; x < scaled_w; x++) {
            int src_x = (int)((float)x / scale);
            int src_y = (int)((float)y / scale);

            // Clamp to input bounds
            if (src_x >= in_w) src_x = in_w - 1;
            if (src_y >= in_h) src_y = in_h - 1;

            int dst_idx = ((offset_y + y) * out_w + (offset_x + x)) * 3;
            int src_idx = (src_y * in_w + src_x) * 3;

            out[dst_idx + 0] = rgb_in[src_idx + 0];  // R
            out[dst_idx + 1] = rgb_in[src_idx + 1];  // G
            out[dst_idx + 2] = rgb_in[src_idx + 2];  // B
        }
    }

    return out;
}

static cJSON* format_detections_for_api(cJSON* raw_detections, int image_index) {
    cJSON* formatted = cJSON_CreateArray();

    cJSON* detection;
    cJSON_ArrayForEach(detection, raw_detections) {
        cJSON* formatted_det = cJSON_CreateObject();

        // Add image index
        cJSON_AddNumberToObject(formatted_det, "index", image_index);

        // Add original image dimensions for client reference
        cJSON* image_info = cJSON_CreateObject();
        cJSON_AddNumberToObject(image_info, "width", letterbox_params.original_width);
        cJSON_AddNumberToObject(image_info, "height", letterbox_params.original_height);
        cJSON_AddItemToObject(formatted_det, "image", image_info);

        // Copy label
        cJSON* label = cJSON_GetObjectItem(detection, "label");
        if (label && label->valuestring) {
            cJSON_AddStringToObject(formatted_det, "label", label->valuestring);

            // Lookup class_id
            int class_id = get_class_id_from_label(label->valuestring);
            cJSON_AddNumberToObject(formatted_det, "class_id", class_id);
        }

        // Copy confidence
        cJSON* conf = cJSON_GetObjectItem(detection, "c");
        if (conf) {
            cJSON_AddNumberToObject(formatted_det, "confidence", conf->valuedouble);
        }

        // Get coordinates (DetectX format: top-left, normalized 0-1)
        cJSON* x_item = cJSON_GetObjectItem(detection, "x");
        cJSON* y_item = cJSON_GetObjectItem(detection, "y");
        cJSON* w_item = cJSON_GetObjectItem(detection, "w");
        cJSON* h_item = cJSON_GetObjectItem(detection, "h");

        if (x_item && y_item && w_item && h_item) {
            // Coordinates are normalized 0-1 in model space
            double x_norm = x_item->valuedouble;
            double y_norm = y_item->valuedouble;
            double w_norm = w_item->valuedouble;
            double h_norm = h_item->valuedouble;

            // Convert to model pixel coordinates
            double x_model = x_norm * modelWidth;
            double y_model = y_norm * modelHeight;
            double w_model = w_norm * modelWidth;
            double h_model = h_norm * modelHeight;

            // Transform back to original image coordinates
            // (accounting for letterbox offset and scale)
            double x_orig = (x_model - letterbox_params.offset_x) / letterbox_params.scale;
            double y_orig = (y_model - letterbox_params.offset_y) / letterbox_params.scale;
            double w_orig = w_model / letterbox_params.scale;
            double h_orig = h_model / letterbox_params.scale;

            // Clamp to original image bounds
            if (x_orig < 0) x_orig = 0;
            if (y_orig < 0) y_orig = 0;
            if (x_orig + w_orig > letterbox_params.original_width) {
                w_orig = letterbox_params.original_width - x_orig;
            }
            if (y_orig + h_orig > letterbox_params.original_height) {
                h_orig = letterbox_params.original_height - y_orig;
            }

            // bbox_pixels (top-left, absolute pixels in ORIGINAL image coordinates)
            cJSON* bbox_pixels = cJSON_CreateObject();
            cJSON_AddNumberToObject(bbox_pixels, "x", (int)x_orig);
            cJSON_AddNumberToObject(bbox_pixels, "y", (int)y_orig);
            cJSON_AddNumberToObject(bbox_pixels, "w", (int)w_orig);
            cJSON_AddNumberToObject(bbox_pixels, "h", (int)h_orig);
            cJSON_AddItemToObject(formatted_det, "bbox_pixels", bbox_pixels);

            // bbox_yolo (center, normalized 0-1 in ORIGINAL image space)
            double cx_orig_norm = (x_orig + w_orig / 2.0) / letterbox_params.original_width;
            double cy_orig_norm = (y_orig + h_orig / 2.0) / letterbox_params.original_height;
            double w_orig_norm = w_orig / letterbox_params.original_width;
            double h_orig_norm = h_orig / letterbox_params.original_height;

            cJSON* bbox_yolo = cJSON_CreateObject();
            cJSON_AddNumberToObject(bbox_yolo, "x", cx_orig_norm);  // center x
            cJSON_AddNumberToObject(bbox_yolo, "y", cy_orig_norm);  // center y
            cJSON_AddNumberToObject(bbox_yolo, "w", w_orig_norm);
            cJSON_AddNumberToObject(bbox_yolo, "h", h_orig_norm);
            cJSON_AddItemToObject(formatted_det, "bbox_yolo", bbox_yolo);
        }

        cJSON_AddItemToArray(formatted, formatted_det);
    }

    return formatted;
}

static int get_class_id_from_label(const char* label) {
    if (!label || !modelLabels) return -1;

    for (size_t i = 0; i < numLabels; i++) {
        if (strcmp(modelLabels[i], label) == 0) {
            return (int)i;
        }
    }

    return -1;
}

// NMS implementation (from DetectX)
static float iou(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2) {
    float area1 = w1 * h1;
    float area2 = w2 * h2;

    float xi1 = fmaxf(x1, x2);
    float yi1 = fmaxf(y1, y2);
    float xi2 = fminf(x1 + w1, x2 + w2);
    float yi2 = fminf(y1 + h1, y2 + h2);

    float intersection_width = fmaxf(0, xi2 - xi1);
    float intersection_height = fmaxf(0, yi2 - yi1);
    float intersection_area = intersection_width * intersection_height;

    float union_area = area1 + area2 - intersection_area;

    return (union_area > 0) ? (intersection_area / union_area) : 0;
}

static cJSON* non_maximum_suppression(cJSON* list) {
    if (!list || cJSON_GetArraySize(list) == 0) {
        return cJSON_CreateArray();
    }

    cJSON* result = cJSON_CreateArray();
    int size = cJSON_GetArraySize(list);

    // Track which detections to keep
    bool* keep = calloc(size, sizeof(bool));
    if (!keep) return list;

    for (int i = 0; i < size; i++) {
        keep[i] = true;
    }

    // Compare each detection with every other detection
    for (int i = 0; i < size; i++) {
        if (!keep[i]) continue;

        cJSON* det1 = cJSON_GetArrayItem(list, i);
        float x1 = cJSON_GetObjectItem(det1, "x")->valuedouble;
        float y1 = cJSON_GetObjectItem(det1, "y")->valuedouble;
        float w1 = cJSON_GetObjectItem(det1, "w")->valuedouble;
        float h1 = cJSON_GetObjectItem(det1, "h")->valuedouble;
        float c1 = cJSON_GetObjectItem(det1, "c")->valuedouble;

        for (int j = i + 1; j < size; j++) {
            if (!keep[j]) continue;

            cJSON* det2 = cJSON_GetArrayItem(list, j);
            
            // Only suppress if same class
            const char* label1 = cJSON_GetObjectItem(det1, "label")->valuestring;
            const char* label2 = cJSON_GetObjectItem(det2, "label")->valuestring;
            if (strcmp(label1, label2) != 0) continue;

            float x2 = cJSON_GetObjectItem(det2, "x")->valuedouble;
            float y2 = cJSON_GetObjectItem(det2, "y")->valuedouble;
            float w2 = cJSON_GetObjectItem(det2, "w")->valuedouble;
            float h2 = cJSON_GetObjectItem(det2, "h")->valuedouble;
            float c2 = cJSON_GetObjectItem(det2, "c")->valuedouble;

            float iou_val = iou(x1, y1, w1, h1, x2, y2, w2, h2);

            if (iou_val > nms) {
                // Suppress the one with lower confidence
                if (c1 > c2) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }

    // Build result array
    for (int i = 0; i < size; i++) {
        if (keep[i]) {
            cJSON* det = cJSON_GetArrayItem(list, i);
            cJSON_AddItemToArray(result, cJSON_Duplicate(det, 1));
        }
    }

    free(keep);

    LOG("NMS: %d -> %d detections\n", size, cJSON_GetArraySize(result));

    return result;
}

static bool createAndMapTmpFile(char* fileName, size_t fileSize, void** mappedAddr, int* fd) {
    *fd = mkstemp(fileName);
    if (*fd < 0) {
        LOG_WARN("%s: Failed to create temp file: %s\n", __func__, strerror(errno));
        return false;
    }

    if (unlink(fileName) < 0) {
        LOG_WARN("%s: Failed to unlink temp file: %s\n", __func__, strerror(errno));
        close(*fd);
        *fd = -1;
        return false;
    }

    if (ftruncate(*fd, fileSize) < 0) {
        LOG_WARN("%s: Failed to truncate file: %s\n", __func__, strerror(errno));
        close(*fd);
        *fd = -1;
        return false;
    }

    *mappedAddr = mmap(NULL, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, 0);
    if (*mappedAddr == MAP_FAILED) {
        LOG_WARN("%s: Failed to mmap file: %s\n", __func__, strerror(errno));
        close(*fd);
        *fd = -1;
        return false;
    }

    return true;
}
