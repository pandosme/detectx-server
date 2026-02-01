/**
 * jpeg_decoder.c - JPEG Image Decoder Implementation
 */

#include "jpeg_decoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syslog.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <jerror.h>

// Error handler for libjpeg
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

typedef struct my_error_mgr* my_error_ptr;

static void my_error_exit(j_common_ptr cinfo) {
    my_error_ptr myerr = (my_error_ptr)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

bool JPEG_Decode(const uint8_t* jpeg_data, size_t jpeg_size, DecodedImage* out_image) {
    if (!jpeg_data || jpeg_size == 0 || !out_image) {
        syslog(LOG_ERR, "JPEG_Decode: Invalid parameters");
        return false;
    }

    memset(out_image, 0, sizeof(DecodedImage));

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    // Setup error handling
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        // Error occurred during JPEG decoding
        syslog(LOG_ERR, "JPEG decode error");
        jpeg_destroy_decompress(&cinfo);
        if (out_image->data) {
            free(out_image->data);
            out_image->data = NULL;
        }
        return false;
    }

    // Initialize decompressor
    jpeg_create_decompress(&cinfo);

    // Set source to memory buffer
    jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);

    // Read JPEG header
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        syslog(LOG_ERR, "Invalid JPEG header");
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    // Set output format to RGB
    cinfo.out_color_space = JCS_RGB;

    // Start decompression
    jpeg_start_decompress(&cinfo);

    // Allocate output buffer
    out_image->width = cinfo.output_width;
    out_image->height = cinfo.output_height;
    out_image->channels = cinfo.output_components;  // Should be 3 for RGB
    out_image->size = out_image->width * out_image->height * out_image->channels;

    if (out_image->channels != 3) {
        syslog(LOG_ERR, "Unexpected number of channels: %d", out_image->channels);
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    out_image->data = malloc(out_image->size);
    if (!out_image->data) {
        syslog(LOG_ERR, "Failed to allocate image buffer (%zu bytes)", out_image->size);
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    // Read scanlines
    int row_stride = cinfo.output_width * cinfo.output_components;
    JSAMPROW row_pointer[1];

    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = &out_image->data[cinfo.output_scanline * row_stride];
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    syslog(LOG_DEBUG, "JPEG decoded: %dx%d, %d channels, %zu bytes",
           out_image->width, out_image->height, out_image->channels, out_image->size);

    return true;
}

void JPEG_FreeImage(DecodedImage* image) {
    if (!image) {
        return;
    }

    if (image->data) {
        free(image->data);
        image->data = NULL;
    }

    memset(image, 0, sizeof(DecodedImage));
}

bool JPEG_GetDimensions(const uint8_t* jpeg_data, size_t jpeg_size,
                        int* width, int* height) {
    if (!jpeg_data || jpeg_size == 0 || !width || !height) {
        syslog(LOG_ERR, "JPEG_GetDimensions: Invalid parameters");
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    // Setup error handling
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer)) {
        syslog(LOG_ERR, "JPEG header read error");
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    // Initialize decompressor
    jpeg_create_decompress(&cinfo);

    // Set source to memory buffer
    jpeg_mem_src(&cinfo, jpeg_data, jpeg_size);

    // Read JPEG header only
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        syslog(LOG_ERR, "Invalid JPEG header");
        jpeg_destroy_decompress(&cinfo);
        return false;
    }

    // Extract dimensions
    *width = cinfo.image_width;
    *height = cinfo.image_height;

    // Cleanup (no decompression needed)
    jpeg_destroy_decompress(&cinfo);

    return true;
}
