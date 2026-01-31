#ifndef MODEL_H
#define MODEL_H

#include "larod.h"
#include "cJSON.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes and configures the detection model for inference.
 *
 * This function sets up the neural network, allocates all required buffers,
 * and reads model parameters and configuration.
 * It must be called before any inference or image processing.
 *
 * @return true on success, false on failure.
 */
bool Model_Setup(void);

/**
 * @brief Get model input width
 * @return Width in pixels
 */
int Model_GetWidth(void);

/**
 * @brief Get model input height
 * @return Height in pixels
 */
int Model_GetHeight(void);

/**
 * @brief Perform inference on JPEG image data and return detected objects.
 *
 * This function decodes JPEG, runs preprocessing and inference pipeline.
 * Input validation:
 * - JPEG must be valid format
 * - Recommended: Square aspect ratio (1:1) for best results
 * - Will be automatically scaled/letterboxed to model input size
 *
 * Returns a cJSON array of detection objects. Each detection object includes:
 *   - "label": Detected object class as a string
 *   - "class_id": Numeric class identifier
 *   - "confidence": Confidence value (0.0-1.0)
 *   - "bbox_pixels": Object with x, y, w, h in pixels
 *   - "bbox_yolo": Object with x, y, w, h normalized (0.0-1.0, center format)
 *   - "index": Image index (if provided, otherwise -1)
 *
 * @param jpeg_data  JPEG image data buffer
 * @param jpeg_size  Size of JPEG data in bytes
 * @param image_index  Image index for dataset validation (-1 if not applicable)
 * @param error_msg  Output: Error message if validation fails (can be NULL)
 * @return A cJSON array of detection objects, or NULL on error.
 *         Caller is responsible for freeing (cJSON_Delete).
 */
cJSON* Model_InferenceJPEG(const uint8_t* jpeg_data, size_t jpeg_size,
                           int image_index, char** error_msg);

/**
 * @brief Perform inference on pre-processed tensor data (optimal performance).
 *
 * This function accepts raw RGB data that is already at the exact model input size.
 * Input validation (strict):
 * - Data must be RGB interleaved format (RGBRGBRGB...)
 * - Size must be exactly: width * height * 3 bytes
 * - Dimensions must match model input (typically 640x640)
 *
 * Returns a cJSON array of detection objects (same format as InferenceJPEG).
 *
 * @param rgb_data  Raw RGB pixel data (interleaved)
 * @param width  Image width (must match model input width)
 * @param height  Image height (must match model input height)
 * @param image_index  Image index for dataset validation (-1 if not applicable)
 * @param error_msg  Output: Error message if validation fails (can be NULL)
 * @return A cJSON array of detection objects, or NULL on error.
 *         Caller is responsible for freeing (cJSON_Delete).
 */
cJSON* Model_InferenceTensor(const uint8_t* rgb_data, int width, int height,
                             int image_index, char** error_msg);

/**
 * @brief Clean up and free all model resources and buffers.
 *
 * Call this once on shutdown to properly release all memory and handles used by the model.
 */
void Model_Cleanup(void);


#ifdef __cplusplus
}
#endif

#endif // MODEL_H
