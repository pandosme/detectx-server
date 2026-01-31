# Model.c Implementation Guide

## Current Status

The server infrastructure is complete. The **only remaining task** is adapting Model.c to work with:
1. JPEG input (decode → preprocess → inference)
2. Raw tensor input (direct inference, no preprocessing)

## Required Functions

### 1. `bool Model_Setup(void)`

**Current DetectX version**: Returns `cJSON*` and expects video streams

**Required changes**:
- Change return type from `cJSON*` to `bool`
- Remove VDO/video stream initialization code
- Keep: larod connection, model loading, tensor setup
- Extract and store model dimensions (modelWidth, modelHeight)
- Read settings.json for thresholds
- Initialize preprocessing model (for JPEG path)

**Key code to preserve from DetectX**:
```c
// Lines 574-685 approximately:
// - larodConnect()
// - open MODEL_PATH
// - Platform detection (ARTPEC-8/9)
// - larodLoadModel()
// - larodCreateModelInputs/Outputs()
// - Extract dimensions, quantization parameters
// - Label parsing
```

**Code to remove**:
- Any references to `videoWidth`, `videoHeight` from VDO
- VDO buffer setup
- Crop cache initialization

---

### 2. `int Model_GetWidth(void)` and `int Model_GetHeight(void)`

**Simple accessor functions**:
```c
static unsigned int modelWidth = 640;  // Already exists in DetectX
static unsigned int modelHeight = 640;

int Model_GetWidth(void) {
    return (int)modelWidth;
}

int Model_GetHeight(void) {
    return (int)modelHeight;
}
```

---

### 3. `cJSON* Model_InferenceJPEG(const uint8_t* jpeg_data, size_t jpeg_size, int image_index, char** error_msg)`

**Implementation approach**:

```c
cJSON* Model_InferenceJPEG(const uint8_t* jpeg_data, size_t jpeg_size,
                           int image_index, char** error_msg) {
    if (error_msg) *error_msg = NULL;

    // Step 1: Decode JPEG to RGB
    DecodedImage img;
    if (!JPEG_Decode(jpeg_data, jpeg_size, &img)) {
        if (error_msg) *error_msg = strdup("Failed to decode JPEG image");
        return NULL;
    }

    // Step 2: Validate aspect ratio (optional warning, not error)
    float aspect = (float)img.width / (float)img.height;
    if (aspect < 0.9 || aspect > 1.1) {
        syslog(LOG_WARNING, "Non-square image: %dx%d (aspect %.2f). "
               "Letterboxing will be applied.",
               img.width, img.height, aspect);
    }

    // Step 3: Preprocess RGB to model input size
    // Option A: Use DetectX's existing preprocess.c functions
    //   - Requires adapting preprocess_letterbox() to work with RGB instead of YUV
    //
    // Option B: Implement simple RGB scaling + letterboxing
    //   - Scale/pad RGB to modelWidth x modelHeight
    //   - Write to larodInputAddr buffer
    //
    // DetectX uses larod preprocessing model (ppModel) which expects YUV
    // For server, we need RGB preprocessing

    uint8_t* preprocessed_rgb = preprocess_rgb_letterbox(
        img.data, img.width, img.height,
        modelWidth, modelHeight
    );

    if (!preprocessed_rgb) {
        JPEG_FreeImage(&img);
        if (error_msg) *error_msg = strdup("Preprocessing failed");
        return NULL;
    }

    JPEG_FreeImage(&img);

    // Step 4: Copy to larod input tensor
    memcpy(larodInputAddr, preprocessed_rgb, modelWidth * modelHeight * 3);
    free(preprocessed_rgb);

    // Step 5: Run inference (reuse DetectX logic)
    if (!larodRunJob(conn, infReq, &error)) {
        if (error_msg) *error_msg = strdup("Inference failed");
        return NULL;
    }

    // Step 6: Parse results (reuse DetectX parse_inference_result)
    cJSON* raw_detections = parse_inference_result();
    if (!raw_detections) {
        return cJSON_CreateArray(); // Empty array, not error
    }

    // Step 7: Format detections for server API
    cJSON* formatted = format_detections_for_api(raw_detections, image_index);
    cJSON_Delete(raw_detections);

    return formatted;
}
```

---

### 4. `cJSON* Model_InferenceTensor(const uint8_t* rgb_data, int width, int height, int image_index, char** error_msg)`

**Implementation approach**:

```c
cJSON* Model_InferenceTensor(const uint8_t* rgb_data, int width, int height,
                             int image_index, char** error_msg) {
    if (error_msg) *error_msg = NULL;

    // Step 1: Validate dimensions
    if (width != modelWidth || height != modelHeight) {
        if (error_msg) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "Invalid dimensions: expected %dx%d, got %dx%d",
                     modelWidth, modelHeight, width, height);
            *error_msg = strdup(buf);
        }
        return NULL;
    }

    // Step 2: Copy directly to larod input tensor (no preprocessing needed)
    memcpy(larodInputAddr, rgb_data, width * height * 3);

    // Step 3: Run inference
    if (!larodRunJob(conn, infReq, &error)) {
        if (error_msg) *error_msg = strdup("Inference failed");
        return NULL;
    }

    // Step 4: Parse and format results
    cJSON* raw_detections = parse_inference_result();
    if (!raw_detections) {
        return cJSON_CreateArray();
    }

    cJSON* formatted = format_detections_for_api(raw_detections, image_index);
    cJSON_Delete(raw_detections);

    return formatted;
}
```

---

## Helper Functions Needed

### `format_detections_for_api(cJSON* raw, int image_index)`

Converts DetectX format to server API format:

**DetectX format** (normalized 0-1 coords):
```json
{
  "label": "car",
  "c": 0.87,        // confidence
  "x": 0.234,       // top-left normalized
  "y": 0.156,
  "w": 0.312,
  "h": 0.234
}
```

**Server API format**:
```json
{
  "index": 0,
  "label": "car",
  "class_id": 2,
  "confidence": 0.87,
  "bbox_pixels": {
    "x": 150, "y": 100, "w": 200, "h": 150
  },
  "bbox_yolo": {
    "x": 0.390, "y": 0.273, "w": 0.312, "h": 0.234  // center format
  }
}
```

**Implementation**:
```c
cJSON* format_detections_for_api(cJSON* raw_detections, int image_index) {
    cJSON* formatted = cJSON_CreateArray();

    cJSON* detection;
    cJSON_ArrayForEach(detection, raw_detections) {
        cJSON* formatted_det = cJSON_CreateObject();

        // Add image index
        cJSON_AddNumberToObject(formatted_det, "index", image_index);

        // Copy label
        cJSON* label = cJSON_GetObjectItem(detection, "label");
        if (label) {
            cJSON_AddStringToObject(formatted_det, "label", label->valuestring);

            // Lookup class_id from labels
            int class_id = get_class_id_from_label(label->valuestring);
            cJSON_AddNumberToObject(formatted_det, "class_id", class_id);
        }

        // Copy confidence
        cJSON* conf = cJSON_GetObjectItem(detection, "c");
        if (conf) {
            cJSON_AddNumberToObject(formatted_det, "confidence", conf->valuedouble);
        }

        // Get coordinates (DetectX uses top-left, normalized 0-1)
        double x = cJSON_GetObjectItem(detection, "x")->valuedouble;
        double y = cJSON_GetObjectItem(detection, "y")->valuedouble;
        double w = cJSON_GetObjectItem(detection, "w")->valuedouble;
        double h = cJSON_GetObjectItem(detection, "h")->valuedouble;

        // bbox_pixels (top-left, absolute pixels)
        cJSON* bbox_pixels = cJSON_CreateObject();
        cJSON_AddNumberToObject(bbox_pixels, "x", (int)(x * modelWidth));
        cJSON_AddNumberToObject(bbox_pixels, "y", (int)(y * modelHeight));
        cJSON_AddNumberToObject(bbox_pixels, "w", (int)(w * modelWidth));
        cJSON_AddNumberToObject(bbox_pixels, "h", (int)(h * modelHeight));
        cJSON_AddItemToObject(formatted_det, "bbox_pixels", bbox_pixels);

        // bbox_yolo (center, normalized 0-1)
        cJSON* bbox_yolo = cJSON_CreateObject();
        cJSON_AddNumberToObject(bbox_yolo, "x", x + w/2);  // center x
        cJSON_AddNumberToObject(bbox_yolo, "y", y + h/2);  // center y
        cJSON_AddNumberToObject(bbox_yolo, "w", w);
        cJSON_AddNumberToObject(bbox_yolo, "h", h);
        cJSON_AddItemToObject(formatted_det, "bbox_yolo", bbox_yolo);

        cJSON_AddItemToArray(formatted, formatted_det);
    }

    return formatted;
}
```

---

## Preprocessing Challenge

**The main challenge**: DetectX's preprocessing (preprocess.c) is designed for YUV input from VDO streams. We need RGB preprocessing.

### Option 1: Adapt DetectX preprocess.c
- Modify `preprocess_letterbox()` to accept RGB
- Keep the scaling/letterboxing logic
- More complex but reuses tested code

### Option 2: Implement simple RGB preprocessing
- Write `preprocess_rgb_letterbox()` from scratch
- Simpler to understand
- Need to ensure coordinate transformations match

### Recommended: Option 2 (Simple RGB preprocessing)

```c
uint8_t* preprocess_rgb_letterbox(const uint8_t* rgb_in, int in_w, int in_h,
                                   int out_w, int out_h) {
    uint8_t* out = calloc(out_w * out_h * 3, 1);  // Black background
    if (!out) return NULL;

    // Calculate scale to fit inside output while preserving aspect ratio
    float scale = fminf((float)out_w / in_w, (float)out_h / in_h);
    int scaled_w = (int)(in_w * scale);
    int scaled_h = (int)(in_h * scale);

    // Center the image
    int offset_x = (out_w - scaled_w) / 2;
    int offset_y = (out_h - scaled_h) / 2;

    // Simple nearest-neighbor scaling (can optimize later)
    for (int y = 0; y < scaled_h; y++) {
        for (int x = 0; x < scaled_w; x++) {
            int src_x = (int)((float)x / scale);
            int src_y = (int)((float)y / scale);

            int dst_idx = ((offset_y + y) * out_w + (offset_x + x)) * 3;
            int src_idx = (src_y * in_w + src_x) * 3;

            out[dst_idx + 0] = rgb_in[src_idx + 0];  // R
            out[dst_idx + 1] = rgb_in[src_idx + 1];  // G
            out[dst_idx + 2] = rgb_in[src_idx + 2];  // B
        }
    }

    return out;
}
```

---

## Files to Modify

1. **Model.c** (main work):
   - Modify `Model_Setup()` - remove VDO, return bool
   - Add `Model_GetWidth()`, `Model_GetHeight()`
   - Add `Model_InferenceJPEG()`
   - Add `Model_InferenceTensor()`
   - Add `format_detections_for_api()`
   - Add `preprocess_rgb_letterbox()` or adapt preprocess.c

2. **Model.h** (already updated):
   - ✅ Function signatures are correct

3. **jpeg_decoder.c/h** (already implemented):
   - ✅ JPEG decoding ready

---

## Testing Strategy

1. **Build test**: Verify compilation succeeds
2. **Capabilities test**: `curl http://camera/local/inference-server/capabilities.cgi`
3. **Health test**: `curl http://camera/local/inference-server/health.cgi`
4. **Tensor test**: Post raw RGB data of exact size
5. **JPEG test**: Post JPEG image

---

## Next Steps

1. Read DetectX Model.c lines 566-900 (setup and inference)
2. Extract core larod inference logic
3. Implement the 4 required functions above
4. Test compilation
5. Deploy to camera and test

---

## Estimated Work

- **Model_Setup adaptation**: 30 minutes (remove VDO refs)
- **Model_GetWidth/Height**: 5 minutes (trivial)
- **Model_InferenceTensor**: 20 minutes (straightforward)
- **Model_InferenceJPEG**: 1 hour (JPEG decode + preprocessing)
- **format_detections_for_api**: 20 minutes
- **preprocess_rgb_letterbox**: 30 minutes
- **Testing**: 30 minutes

**Total**: ~3 hours of focused development
