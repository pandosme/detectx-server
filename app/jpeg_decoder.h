/**
 * jpeg_decoder.h - JPEG Image Decoder
 *
 * Decodes JPEG images to RGB format for model inference
 */

#ifndef JPEG_DECODER_H
#define JPEG_DECODER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct {
    uint8_t* data;      // RGB pixel data (RGB interleaved)
    int width;
    int height;
    int channels;       // Always 3 for RGB
    size_t size;        // Total buffer size in bytes
} DecodedImage;

/**
 * @brief Decode JPEG data to RGB format
 *
 * @param jpeg_data  Input JPEG buffer
 * @param jpeg_size  Size of JPEG data
 * @param out_image  Output decoded image structure
 * @return true on success, false on error
 */
bool JPEG_Decode(const uint8_t* jpeg_data, size_t jpeg_size, DecodedImage* out_image);

/**
 * @brief Free resources allocated for decoded image
 *
 * @param image  Image to free
 */
void JPEG_FreeImage(DecodedImage* image);

#endif // JPEG_DECODER_H
