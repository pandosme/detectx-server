# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DetectX Server is a high-performance TensorFlow Lite inference server for Axis cameras that hosts inference for detectx-client applications. The server MUST run on ARTPEC-9 cameras for hardware acceleration, while clients can run on armv7hf, ARTPEC-8, or ARTPEC-9 platforms.

**Primary Use Cases:**
1. **Client/Server Inference:** detectx-client (running on same or different camera) sends images to server for inference
2. **Dataset Validation:** Test custom models with real images before deployment using Python/Node.js scripts

**Platform Requirements:**
- **Server:** ARTPEC-9 only (required for hardware acceleration)
- **Clients:** armv7hf, ARTPEC-8, or ARTPEC-9 (any Axis camera platform)

**Language:** C (ACAP application), with Python/Node.js client libraries

**Custom Models:** The included COCO model (90 classes) is just an example. The design is to replace `app/model/model.tflite` and `app/model/labels.txt` with your own custom TensorFlow Lite INT8 quantized model.

## Custom Model Support

**IMPORTANT:** The COCO model (90 classes) included in `app/model/` is just an example. The server is designed to work with ANY TensorFlow Lite INT8 quantized model.

**To use your own model:**
1. Replace `app/model/model.tflite` with your custom TFLite INT8 model
2. Replace `app/model/labels.txt` with your model's class labels (one per line)
3. Rebuild: `./build.sh`
4. Deploy the new `.eap` file to your ARTPEC-9 camera

**Model Requirements:**
- Format: TensorFlow Lite (`.tflite`)
- Quantization: INT8 (required for ARTPEC-9 DLPU acceleration)
- Input: RGB tensor (any square dimension, e.g., 640×640, 320×320)
- Output: Detection boxes, classes, scores (YOLO or SSD format)

The server will automatically extract model parameters during build and configure itself accordingly.

## Build Commands

### Docker Build (Recommended)
```bash
./build.sh                    # Build with cache (fast, reuses TensorFlow layer)
./build.sh --clean            # Full rebuild without cache

# Manual Docker build
docker build --build-arg CHIP=aarch64 -t detectx .
```

The build process:
1. Multi-stage Docker build using ACAP SDK v12.8.0
2. Extracts model parameters using TensorFlow Python
3. Compiles C application with acap-build
4. Outputs `.eap` file (ACAP package) in current directory

### Local Build (inside container)
```bash
cd app
make                          # Build
make clean                    # Remove build artifacts
```

### Testing

**Prerequisites:** Server must be deployed and running on an ARTPEC-9 camera. Clients can run from anywhere.

```bash
# Test endpoints (replace camera-ip with your ARTPEC-9 camera IP)
curl http://camera-ip:8080/local/detectx/health
curl http://camera-ip:8080/local/detectx/capabilities

# JPEG inference
curl -X POST --data-binary @image.jpg \
  http://camera-ip:8080/local/detectx/inference-jpeg

# Python client (can run on any machine)
cd scripts/python
pip install -r requirements.txt
python inference_client.py --host 192.168.1.100 test.png

# Node.js client (can run on any machine)
cd scripts/nodejs
npm install
npm test

# detectx-client (ACAP application on armv7hf/ARTPEC-8/ARTPEC-9 camera)
# Install detectx-client on any Axis camera and configure it to point to this server
```

## Architecture

### High-Level Design

```
┌─────────────┐
│  HTTP/FCGI  │  main.c - 4 endpoints, FastCGI request handling
│  Interface  │
└──────┬──────┘
       │
┌──────▼──────┐
│   Request   │  server.c - Circular queue (max 3 concurrent)
│    Queue    │            Background worker thread
└──────┬──────┘
       │
┌──────▼──────┐
│   Model     │  Model.c - TFLite inference (INCOMPLETE - see below)
│  Inference  │
└──────┬──────┘
       │
┌──────▼──────┐
│  larod API  │  Axis inference engine, ARTPEC-9 DLPU acceleration
└─────────────┘
```

### Request Processing Pipeline

```
HTTP Request (JPEG or Tensor)
    ↓
Server_QueueRequest()              [server.c] Adds to queue, signals worker
    ↓
inference_worker thread            [server.c] Dequeues and processes
    ↓
Model_InferenceJPEG()     OR       [Model.c] JPEG: decode → preprocess → inference
Model_InferenceTensor()            [Model.c] Tensor: direct inference (optimal)
    ↓
Detection post-processing          NMS, confidence filtering
    ↓
cJSON response array               Sent as HTTP response
    ↓
Statistics update                  Update queue, timing metrics
```

### Key Components

**app/main.c** (HTTP Interface - ~350 lines)
- Entry point, GLib event loop, FastCGI handling
- 4 API endpoints:
  - `GET /capabilities` - Model info, dimensions, class labels
  - `POST /inference-jpeg` - JPEG image inference (≤10MB)
  - `POST /inference-tensor` - Raw RGB tensor inference (exact size required)
  - `GET /health` - Server status, queue size, statistics
- Request validation, queuing, and response handling
- Thread-safe statistics tracking (avg/min/max inference time)

**app/server.c/h** (Request Queue - ~200 lines)
- Producer/consumer pattern with pthread condition variables
- Circular queue: `requests[MAX_QUEUE_SIZE=3]`
- Background worker thread continuously processes queue
- Synchronization: `done`, `not_full`, `not_empty` condition variables
- Returns 503 when queue is full

**app/Model.c/h** (Inference Engine - INCOMPLETE)
- **CRITICAL:** Model.c is currently a stub requiring implementation
- See [MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md) for detailed guide
- Required functions:
  - `Model_Setup()` - Initialize larod, load model, setup tensors
  - `Model_GetWidth()/GetHeight()` - Return model dimensions (640×640)
  - `Model_InferenceJPEG()` - JPEG decode → preprocess → inference
  - `Model_InferenceTensor()` - Direct tensor inference
- Must adapt DetectX's existing Model.c by removing VDO/video stream code

**app/jpeg_decoder.c/h** (JPEG Decoding - ~100 lines)
- Uses TurboJPEG for fast JPEG → RGB conversion
- `JPEG_Decode()` returns `DecodedImage` struct with RGB data
- Already implemented and ready to use

**app/preprocess.c/h** (Image Preprocessing)
- Designed for YUV from VDO streams (DetectX legacy)
- Needs adaptation for RGB letterboxing/scaling
- Three modes: letterbox, crop, stretch
- See MODEL_IMPLEMENTATION.md for RGB preprocessing approach

**app/labelparse.c/h** (Label Parsing)
- Reads labels.txt file into memory
- Cached for performance
- Maps class IDs to label strings

**app/ACAP.c/h** (ACAP SDK Wrapper - ~170 lines)
- FastCGI HTTP handling
- File I/O utilities
- Device info helpers
- Signal handling for graceful shutdown

**app/cJSON.c/h** (JSON Library)
- Embedded JSON library (no external dependency)
- Used for API responses and settings parsing

**app/imgutils.c/h** (Image Utilities)
- Image buffer management
- Pixel format conversions

### Threading Model

- **Main thread:** GLib event loop, handles HTTP requests via FastCGI
- **Worker thread:** Background inference processing (server.c:inference_worker)
- **Synchronization:** pthread mutexes and condition variables
- **Queue limit:** MAX_QUEUE_SIZE=3 to prevent resource exhaustion

### Data Flow: JPEG Inference Example

1. `POST /inference-jpeg` with JPEG bytes
2. `http_inference_jpeg()` validates content-type, size, queue availability
3. `Server_CreateRequest()` allocates request, copies data
4. `Server_QueueRequest()` adds to queue, signals worker thread
5. Worker thread wakes up, dequeues request
6. `Model_InferenceJPEG()` calls:
   - `JPEG_Decode()` → RGB image
   - Preprocessing → 640×640 letterboxed RGB
   - larod inference → detection tensors
   - Post-processing → NMS, confidence filtering
7. Returns cJSON array of detections
8. Main thread sends HTTP response (200/204/error)
9. `Server_FreeRequest()` cleans up memory

## Configuration

### app/settings/settings.json
Runtime configuration for model inference:
```json
{
  "model": {
    "path": "model/model.tflite",      // TFLite INT8 model (replace with custom model)
    "labels": "model/labels.txt",      // Class labels (replace with custom labels)
    "scaleMode": "letterbox",          // or "crop", "stretch"
    "objectness": 0.25,                // NMS objectness threshold
    "confidence": 0.30,                // Detection confidence threshold
    "nms": 0.05                        // NMS IoU threshold
  },
  "server": {
    "max_queue_size": 3,               // Concurrent request limit
    "max_image_size_mb": 10            // JPEG size limit
  }
}
```

**Note:** The paths reference the model files packaged in the ACAP. To use a custom model, replace the files in `app/model/` before building.

### app/manifest.json
ACAP package metadata:
- Package name: `detectx`
- Version: 1.0.0
- 7 FastCGI endpoints (4 API + 3 web UI)
- Access levels: admin (settings), viewer (inference)
- DBus resource for VAPIX credentials

### Dockerfile Build Arguments
- `ARCH=aarch64` - Target architecture
- `CHIP=aarch64` - ARTPEC-9 chip
- `VERSION=12.8.0` - ACAP SDK version
- `UBUNTU_VERSION=24.04` - Base OS

## Important Implementation Notes

### Model.c is INCOMPLETE

The server infrastructure (HTTP, queuing, JSON, JPEG decoding) is fully implemented. The **only remaining work** is implementing Model.c inference functions. See [MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md) for:

- Detailed implementation guide
- Code examples for each required function
- Preprocessing approach (RGB letterboxing)
- API response format conversion
- Estimated ~3 hours of focused development

The existing DetectX Model.c can be adapted by:
1. Removing VDO/video stream code
2. Changing `Model_Setup()` return type from `cJSON*` to `bool`
3. Implementing JPEG and tensor inference paths
4. Converting DetectX detection format to server API format

### Hardware-Specific Optimizations

- **ARTPEC-9 DLPU:** Uses larod API v3 for hardware acceleration
- **INT8 quantization:** Model must be INT8 quantized for performance
- **TurboJPEG:** Fast JPEG decoding on ARM
- Platform detection ensures compatibility across ARTPEC-7/8/9

### API Response Format

The server returns detections in a specific format that differs from DetectX:

**Server API format:**
```json
{
  "detections": [
    {
      "index": 0,                      // Image index (for dataset validation)
      "label": "car",
      "class_id": 2,
      "confidence": 0.87,
      "bbox_pixels": {                 // Top-left corner, pixels
        "x": 150, "y": 100, "w": 200, "h": 150
      },
      "bbox_yolo": {                   // Center format, normalized 0-1
        "x": 0.390, "y": 0.273, "w": 0.312, "h": 0.234
      }
    }
  ]
}
```

**DetectX format (needs conversion):**
```json
{
  "label": "car",
  "c": 0.87,                          // confidence
  "x": 0.234,                         // top-left normalized
  "y": 0.156,
  "w": 0.312,
  "h": 0.234
}
```

### HTTP Status Codes

- **200 OK:** Inference successful, detections found
- **204 No Content:** Inference successful, no detections (normal)
- **400 Bad Request:** Invalid input (wrong format, size, headers)
- **503 Service Unavailable:** Queue full (max 3 concurrent)
- **500 Internal Server Error:** Inference failed

## Client Libraries

### Python Client (scripts/python/)
```python
from inference_client import InferenceClient

client = InferenceClient('192.168.1.100', 'root', 'pass')
detections = client.infer_jpeg('image.jpg')

# Batch processing
python batch_inference.py /path/to/images --output results.json
```

### Node.js Client (scripts/nodejs/)
```javascript
const InferenceClient = require('./inference-client');
const client = new InferenceClient('192.168.1.100', 'root', 'pass');
const detections = await client.inferJpeg('image.jpg');

// Batch processing
node batch-inference.js /path/to/images --output results.json
```

### Key Client Features
- Digest authentication support
- JPEG and tensor inference endpoints
- Batch processing with progress tracking
- Error retry logic (exponential backoff for 503)
- Preprocessing utilities (letterbox, resize)

## Performance

Performance on ARTPEC-9 with COCO model (640×640):

- **JPEG Inference:** ~150-300ms (includes decode + preprocess + inference)
- **Tensor Inference:** ~50-100ms (inference only, no overhead)
- **Throughput:** ~5-10 FPS
- **Queue Size:** 3 concurrent requests maximum
- **Recommended Parallelism:** 3 workers to match queue size

**Note:** Performance will vary based on your custom model's size and complexity. Smaller models (e.g., 320×320) will be faster; larger models (e.g., 1024×1024) will be slower.

## Development Workflow

1. **Modify code** in `app/` directory
2. **Build:** `./build.sh` (or `./build.sh --clean` for fresh build)
3. **Extract:** Build outputs `.eap` file to current directory
4. **Install:** Upload `.eap` to camera via web interface or API
5. **Test:** Use curl or client libraries to test endpoints
6. **Monitor:** Check `/health` endpoint for statistics

## Common Issues

**Model.c not implemented:**
- Current Model.c is a stub returning errors
- Follow MODEL_IMPLEMENTATION.md guide
- Adapt existing DetectX Model.c code

**Preprocessing mismatch:**
- DetectX uses YUV from VDO streams
- Server needs RGB preprocessing
- Implement RGB letterboxing or adapt preprocess.c

**Queue full (503 errors):**
- Reduce parallel workers to ≤3
- Add retry logic with exponential backoff
- Monitor `/health` for queue size

**Authentication failures:**
- Use digest authentication
- Verify camera credentials
- Check user permissions (viewer role minimum)

## Relationship to DetectX and detectx-client

This project is based on [DetectX](https://github.com/pandosme/DetectX) by Fredrik Persson, split into a client/server architecture.

**Architecture:**
- **detectx-server (this project):** Hosts inference on ARTPEC-9 cameras
- **detectx-client:** Runs on any camera (armv7hf, ARTPEC-8, ARTPEC-9), sends images to server
- **Deployment:** Client and server can run on the same camera OR different cameras

**Key differences from DetectX standalone:**
- **Architecture:** Client/server vs. all-in-one
- **Platform:** Server MUST be ARTPEC-9; clients can be armv7hf/ARTPEC-8/ARTPEC-9
- **Use case:** Multiple cameras sharing inference vs. single camera local inference
- **Input:** HTTP/REST API vs. VDO video streams
- **Preprocessing:** RGB (HTTP images) vs. YUV (VDO streams)

**Client Integration:**
The primary consumer of this server is detectx-client (ACAP application), but any HTTP client can use the REST API (Python scripts, Node.js, curl, etc.) for dataset validation and testing.

**Code reuse strategy:**
- Server infrastructure (HTTP, queue, JSON): Custom implementation
- Model inference (larod API): Adaptable from DetectX Model.c
- JPEG decoding: New (TurboJPEG)
- Preprocessing: Needs RGB adaptation from DetectX YUV code

## Files to Modify for Model Implementation

When implementing Model.c, you will primarily modify:

1. **app/Model.c** - Main implementation work
   - `Model_Setup()` - Remove VDO, load model via larod
   - `Model_GetWidth()`, `Model_GetHeight()` - Trivial accessors
   - `Model_InferenceJPEG()` - JPEG decode → preprocess → inference
   - `Model_InferenceTensor()` - Direct tensor inference
   - Helper: `format_detections_for_api()` - Convert DetectX format to server format
   - Helper: `preprocess_rgb_letterbox()` - RGB preprocessing

2. **app/Model.h** - Already correct, no changes needed

3. **app/preprocess.c** (optional) - If adapting existing preprocessing
   - Modify to accept RGB instead of YUV
   - Keep letterbox/scaling logic

Reference DetectX Model.c (lines 566-900) for larod setup and inference logic.

## Additional Resources

- **[README.md](README.md)** - Full project documentation, API reference
- **[MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md)** - Detailed Model.c implementation guide
- **[scripts/README.md](scripts/README.md)** - Client library usage and examples
- **DetectX (original):** https://github.com/pandosme/DetectX
