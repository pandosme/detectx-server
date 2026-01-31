# DetectX Server

High-performance TensorFlow Lite inference server ACAP for Axis cameras. Enable AI object detection on older cameras by sharing compute from a single ARTPEC-9 camera.

[![Platform](https://img.shields.io/badge/Platform-ARTPEC--9-blue)]()
[![ACAP SDK](https://img.shields.io/badge/ACAP%20SDK-12.8.0-green)]()
[![Language](https://img.shields.io/badge/Language-C-orange)]()

> **📦 Complete Solution**: This is the **server component**. For camera clients, see [detectx-client](https://github.com/pandosme/detectx-client) - the ACAP that runs on ARTPEC-7/8/9 cameras and sends images to this server for inference.

---

## Why DetectX Server?

**Enable object detection on older Axis cameras without replacing hardware.**

DetectX Server is an ACAP (Axis Camera Application Platform) application that runs on ARTPEC-9 cameras and provides TensorFlow Lite inference as an HTTP service. This allows:

- **🎯 Upgrade existing infrastructure**: Add AI capabilities to ARTPEC-7 and ARTPEC-8 cameras
- **💰 Cost optimization**: Share one ARTPEC-9's processing power across multiple cameras
- **🔧 Flexible deployment**: Server and clients can run on the same camera or separate cameras
- **🧪 Model validation**: Test custom models with real camera images before production deployment
- **📊 Dataset validation**: Process test datasets to evaluate model performance

### Primary Use Cases

| Use Case | Description |
|----------|-------------|
| **Multi-camera detection** | Multiple ARTPEC-7/8/9 cameras send images to one ARTPEC-9 server for inference |
| **Legacy camera upgrade** | Add AI to existing ARTPEC-7/8 cameras without replacement |
| **Model testing** | Validate custom TFLite models with real images before deployment |
| **Centralized inference** | Manage models and inference logic on one camera |
| **Dataset evaluation** | Process test datasets via scripts to measure accuracy |

---

## When to Use DetectX Server

### ✅ Use DetectX Server When:

- You have **ARTPEC-7 or ARTPEC-8 cameras** that need object detection capabilities
- You want to **share inference compute** across multiple cameras
- You need to **test custom models** with real camera images before production
- You're building a **multi-camera system** and want centralized model management
- You have **reliable network connectivity** between cameras (LAN recommended)
- You want to **avoid buying ARTPEC-9 for every camera**

### ❌ Don't Use DetectX Server When:

- You only have **one camera** → Use [DetectX standalone](https://github.com/pandosme/DetectX) instead
- You need **sub-50ms latency** → Network overhead adds ~50-100ms
- Cameras are on **separate networks** with restricted connectivity
- You don't have an **ARTPEC-9 camera** available (required for server)
- You need **offline operation** without network dependency

---

## Relationship to DetectX

This project is based on [DetectX](https://github.com/pandosme/DetectX) by Fredrik Persson, reimagined as a client/server architecture.

| Feature | DetectX (Standalone) | DetectX Server (This Project) |
|---------|---------------------|-------------------------------|
| **Architecture** | All-in-one on single camera | Client/Server split |
| **Platform Required** | ARTPEC-9 only | Server: ARTPEC-9<br>Clients: ARTPEC-7/8/9 |
| **Input Source** | VDO video streams | HTTP REST API (JPEG/Tensor) |
| **Deployment** | Single camera standalone | Multiple cameras, shared server |
| **Use Case** | One powerful camera | Multiple cameras OR dataset testing |
| **Network** | No network required | Requires HTTP connectivity |

**Key Innovation**: DetectX Server decouples inference from capture, enabling older cameras to leverage modern AI capabilities through HTTP requests.

---

## Architecture

### High-Level Overview

```
┌─────────────────┐
│  Client Camera  │  ARTPEC-7/8/9
│ (detectx-client │  Captures images
│   ACAP or any   │  Sends via HTTP
│   HTTP client)  │
└────────┬────────┘
         │ POST /inference-jpeg
         │ (JPEG image)
         ▼
┌─────────────────────────────────────┐
│       ARTPEC-9 Camera               │
│  ┌───────────────────────────────┐  │
│  │   DetectX Server ACAP         │  │
│  │                               │  │
│  │  FastCGI → Request Queue      │  │
│  │         ↓                     │  │
│  │  Worker Thread → TFLite       │  │
│  │         ↓                     │  │
│  │  DLPU Hardware Acceleration   │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
         │
         ▼
    JSON Response
    (Detections with bboxes)
```

### Internal Architecture

```
HTTP/FastCGI Interface (main.c)
    ↓
Request Queue (server.c)
  • Circular buffer (max 3 concurrent)
  • Background worker thread
  • Condition variables for sync
    ↓
Inference Engine (Model.c)
  • JPEG decode (TurboJPEG)
  • RGB preprocessing (letterbox/crop/stretch)
  • larod API → ARTPEC-9 DLPU
  • NMS post-processing
    ↓
JSON Response (cJSON)
```

**Key Components**:
- **main.c**: FastCGI HTTP handler, 4 REST endpoints
- **server.c**: Thread-safe request queue (producer/consumer pattern)
- **Model.c**: TFLite inference via larod API (hardware accelerated)
- **jpeg_decoder.c**: TurboJPEG for fast JPEG → RGB conversion
- **preprocess.c**: Image scaling/letterboxing for model input

---

## Deployment Scenarios

### Scenario 1: Multi-Camera Installation
**Best for**: Cost-effective surveillance with mixed camera generations

```
┌──────────────────┐
│  ARTPEC-7 Camera │ ──┐
│  (Lobby)         │   │
└──────────────────┘   │
                       │
┌──────────────────┐   │  HTTP POST
│  ARTPEC-8 Camera │ ──┼─────────────► ┌─────────────────────┐
│  (Entrance)      │   │                │  ARTPEC-9 Camera    │
└──────────────────┘   │                │  (Server Room)      │
                       │                │  DetectX Server     │
┌──────────────────┐   │                └─────────────────────┘
│  ARTPEC-9 Camera │ ──┘
│  (Parking)       │       All cameras share inference compute
└──────────────────┘
```

**Benefits**:
- Central model management (update once, affects all)
- Lower hardware costs (only one ARTPEC-9 needed)
- Uniform detection across all cameras

---

### Scenario 2: Single Camera (Self-Contained)
**Best for**: ARTPEC-9 standalone with both server and client

```
┌──────────────────────────────────────┐
│        ARTPEC-9 Camera               │
│                                      │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ detectx-     │  │ DetectX      │ │
│  │ client ACAP  │─►│ Server ACAP  │ │
│  │ (captures)   │  │ (inference)  │ │
│  └──────────────┘  └──────────────┘ │
│         │ localhost HTTP │          │
│         └─────────────────┘          │
└──────────────────────────────────────┘
```

**Benefits**:
- No network latency (localhost)
- Self-contained operation
- Can still serve remote clients if needed

---

### Scenario 3: Dataset Validation
**Best for**: Testing custom models before production deployment

```
┌─────────────────────┐
│   Your Workstation  │
│                     │
│  Python/Node.js     │
│  Validation Scripts │
└──────────┬──────────┘
           │ POST /inference-jpeg
           │ Batch processing
           ▼
┌─────────────────────────────────┐
│     ARTPEC-9 Camera             │
│     DetectX Server              │
│                                 │
│  Processes test dataset         │
│  Returns detections             │
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Results & Metrics │
│   • Precision       │
│   • Recall          │
│   • mAP             │
│   • Conf. threshold │
└─────────────────────┘
```

**Benefits**:
- Test models on actual camera hardware
- Validate before deploying to production
- Tune confidence thresholds
- Generate accuracy reports

---

## Example Model: COCO Dataset

The server ships with a pre-trained **YOLOv8 COCO model** supporting **90 object classes**:

**Categories**: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

**Model Details**:
- **Framework**: YOLOv8 TensorFlow Lite INT8 quantized
- **Input Size**: 640×640×3 RGB
- **Model Size**: 2.1 MB
- **Quantization**: INT8 (required for ARTPEC-9 DLPU acceleration)
- **Performance**: ~150-300ms per image (JPEG inference)

This is an **example model only**. You can replace it with any custom TFLite INT8 model.

---

## Requirements

### Server Requirements (ARTPEC-9 Camera)

- **Hardware**: Axis camera with **ARTPEC-9** chip
  - Examples: Q1656, P1468-LE, M3116-LVE, Q6215-LE
- **Firmware**: Compatible with ACAP SDK 12.8.0
- **Network**: HTTP accessible from client cameras
- **Storage**: ~10 MB for ACAP + model files

### Client Requirements (Optional)

**For Camera Clients** (Recommended):
- Install [**detectx-client**](https://github.com/pandosme/detectx-client) ACAP on any Axis camera
  - **Supported platforms**: ARTPEC-7, ARTPEC-8, or ARTPEC-9
  - **Purpose**: Automatically captures video frames and sends them to DetectX Server
  - **Configuration**: Point client to server IP address
  - See [detectx-client documentation](https://github.com/pandosme/detectx-client) for installation guide

**For Script Clients** (Testing/Validation):
- Any machine with HTTP client capability
  - Python (see [scripts/python/](scripts/python/))
  - Node.js (see [scripts/nodejs/](scripts/nodejs/))
  - curl or any HTTP client

**Network**:
- HTTP access from clients to server camera
- LAN recommended for best performance
- Typical bandwidth: ~100 KB per image (JPEG)

### Build Requirements (For Developers)

- **ACAP SDK**: Version 12.8.0 (Docker-based)
- **Docker**: For building the ACAP
- **Build Host**: Linux, macOS, or Windows with Docker

---

## Installation

### Step 1: Build the ACAP

```bash
# Clone repository
git clone https://github.com/your-repo/detectx-server.git
cd detectx-server

# Build (creates .eap file)
./build.sh

# Output: DetectX_COCO_Hub_1_0_0_aarch64.eap (~6.4 MB)
```

**Build options**:
```bash
./build.sh              # Fast build with cache
./build.sh --clean      # Clean rebuild (slower, downloads TensorFlow)
```

### Step 2: Install on ARTPEC-9 Camera

#### Option A: Via Web Interface
1. Open camera web interface: `http://camera-ip`
2. Navigate to **Settings → Apps**
3. Click **Add app** (+)
4. Upload `DetectX_COCO_Hub_1_0_0_aarch64.eap`
5. Click **Install**
6. Start the application

#### Option B: Via Command Line (eap-install.sh)
```bash
# Install via ACAP SDK tools
eap-install.sh install DetectX_COCO_Hub_1_0_0_aarch64.eap <camera-ip> <password>
```

### Step 3: Verify Installation

```bash
# Check server health
curl http://camera-ip:8080/local/detectx/health

# Get model capabilities
curl http://camera-ip:8080/local/detectx/capabilities

# Test inference with sample image
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg \
  -H "Content-Type: image/jpeg" \
  --data-binary @test.jpg
```

**Expected response**:
```json
{
  "detections": [
    {
      "index": 0,
      "label": "person",
      "class_id": 0,
      "confidence": 0.87,
      "bbox_pixels": {"x": 150, "y": 100, "w": 200, "h": 300},
      "bbox_yolo": {"x": 0.390, "y": 0.273, "w": 0.312, "h": 0.468}
    }
  ]
}
```

---

## API Reference

DetectX Server provides 4 REST endpoints under `/local/detectx`:

### GET `/local/detectx/capabilities`

Get server capabilities and model information.

**Authentication**: Optional (viewer role)

**Response**:
```json
{
  "model": {
    "name": "COCO_Hub",
    "input_width": 640,
    "input_height": 640,
    "classes": [
      {"id": 0, "name": "person"},
      {"id": 1, "name": "bicycle"},
      ...
    ],
    "input_formats": [
      {"endpoint": "inference-jpeg", "mime": "image/jpeg", "description": "JPEG image (any size)"},
      {"endpoint": "inference-tensor", "mime": "application/octet-stream", "description": "RGB tensor (640x640x3)"}
    ]
  },
  "server": {
    "version": "1.0.0",
    "max_queue_size": 3
  }
}
```

---

### POST `/local/detectx/inference-jpeg`

Perform inference on JPEG image. Server handles decoding and preprocessing.

**Authentication**: Optional (viewer role)

**Query Parameters**:
- `index` (optional): Image index for dataset validation (integer)

**Request**:
- **Content-Type**: `image/jpeg`
- **Body**: Binary JPEG data (max 10 MB)

**Example**:
```bash
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg

# With authentication
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg?index=0 \
  -H "Content-Type: image/jpeg" \
  --digest -u root:password \
  --data-binary @image.jpg
```

**Response** (200 OK):
```json
{
  "detections": [
    {
      "index": 0,
      "label": "car",
      "class_id": 2,
      "confidence": 0.87,
      "bbox_pixels": {
        "x": 150,
        "y": 100,
        "w": 200,
        "h": 150
      },
      "bbox_yolo": {
        "x": 0.390,
        "y": 0.273,
        "w": 0.312,
        "h": 0.234
      }
    }
  ]
}
```

**Response** (204 No Content): No detections found (normal, not an error)

**Response** (503 Service Unavailable): Queue full, retry with backoff

---

### POST `/local/detectx/inference-tensor`

Perform inference on preprocessed RGB tensor. Faster than JPEG (no decode overhead).

**Authentication**: Optional (viewer role)

**Query Parameters**:
- `index` (optional): Image index for dataset validation

**Request**:
- **Content-Type**: `application/octet-stream`
- **Body**: Raw RGB bytes (width × height × 3 = 640 × 640 × 3 = 1,228,800 bytes)
- **Format**: RGB interleaved, uint8, no padding

**Example** (Python):
```python
import numpy as np
from PIL import Image
import requests

# Load and preprocess image to 640x640 RGB tensor
img = Image.open('photo.jpg').convert('RGB').resize((640, 640))
tensor = np.array(img, dtype=np.uint8)

# Send tensor
response = requests.post(
    'http://camera-ip:8080/local/detectx/inference-tensor',
    data=tensor.tobytes(),
    headers={'Content-Type': 'application/octet-stream'}
)

detections = response.json()['detections']
```

**Response**: Same format as `/inference-jpeg`

---

### GET `/local/detectx/health`

Get server health status and statistics.

**Authentication**: Optional (viewer role)

**Response**:
```json
{
  "running": true,
  "queue_size": 1,
  "statistics": {
    "total_requests": 1234,
    "successful_requests": 1200,
    "failed_requests": 34,
    "avg_inference_time_ms": 185.3,
    "min_inference_time_ms": 152.1,
    "max_inference_time_ms": 298.7
  }
}
```

---

## HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| **200 OK** | Inference successful, detections found | Process detections |
| **204 No Content** | Inference successful, no detections | Normal (empty result) |
| **400 Bad Request** | Invalid input (wrong format, size, headers) | Check request format |
| **503 Service Unavailable** | Queue full (max 3 concurrent requests) | Retry with exponential backoff |
| **500 Internal Server Error** | Inference failed | Check server logs |

---

## Client Integration

### detectx-client ACAP (Recommended for Production)

**[detectx-client](https://github.com/pandosme/detectx-client)** is the companion ACAP for production camera deployments. It runs on ARTPEC-7, ARTPEC-8, or ARTPEC-9 cameras and automatically captures video frames and sends them to DetectX Server for inference.

#### What detectx-client Does:
- ✅ Captures video frames from camera's video stream
- ✅ Sends frames to DetectX Server via HTTP
- ✅ Receives detection results
- ✅ Triggers camera events based on detections
- ✅ Supports zones and detection filtering
- ✅ Configurable frame rate and quality

#### Quick Start with detectx-client:

**1. Install DetectX Server** (this ACAP) on ARTPEC-9 camera:
```bash
# Build and install DetectX Server first
./build.sh
# Upload DetectX_COCO_Hub_1_0_0_aarch64.eap to ARTPEC-9 camera
```

**2. Install detectx-client** on any camera (ARTPEC-7/8/9):
```bash
# Download from https://github.com/pandosme/detectx-client
# Upload detectx-client .eap to your camera (can be older ARTPEC-7/8)
```

**3. Configure detectx-client** to point to server:
```json
{
  "server": {
    "host": "192.168.1.100",
    "port": 8080,
    "endpoint": "/local/detectx/inference-jpeg"
  },
  "capture": {
    "fps": 2,
    "resolution": "640x480"
  },
  "detection": {
    "classes": ["person", "car"],
    "confidence_threshold": 0.5
  }
}
```

**4. Start both ACAPs** and verify:
```bash
# Check server health
curl http://192.168.1.100:8080/local/detectx/health

# Client will now automatically send frames to server
# Detections will trigger camera events
```

#### Architecture with detectx-client:

```
┌─────────────────────────┐
│  ARTPEC-7/8 Camera      │
│  ┌───────────────────┐  │
│  │ detectx-client    │──┼────HTTP POST────► ┌─────────────────────┐
│  │ • Captures frames │  │                    │  ARTPEC-9 Camera    │
│  │ • Sends to server │  │                    │  DetectX Server     │
│  └───────────────────┘  │ ◄──JSON Response── │  • Runs inference   │
│         │               │                    │  • Returns detections│
│         ▼               │                    └─────────────────────┘
│  Camera Events/Actions  │
└─────────────────────────┘
```

For detailed documentation, installation guide, and configuration options, see:
**[detectx-client GitHub Repository](https://github.com/pandosme/detectx-client)**

---

### Python Client

See [scripts/python/](scripts/python/) for full client library.

```python
#!/usr/bin/env python3
from inference_client import InferenceClient

# Initialize client
client = InferenceClient(
    host='192.168.1.100',
    username='root',  # Optional
    password='pass'   # Optional
)

# Single image inference
detections = client.infer_jpeg('image.jpg')

for det in detections:
    print(f"{det['label']}: {det['confidence']:.2%} at "
          f"({det['bbox_pixels']['x']}, {det['bbox_pixels']['y']})")

# Batch processing
from batch_inference import batch_inference

results = batch_inference(
    client=client,
    image_dir='./dataset/images',
    output_file='results.json',
    num_workers=3
)
```

---

### Node.js Client

See [scripts/nodejs/](scripts/nodejs/) for full client library.

```javascript
const InferenceClient = require('./inference-client');

const client = new InferenceClient('192.168.1.100', 'root', 'pass');

// Single image
const detections = await client.inferJpeg('image.jpg');
console.log(`Found ${detections.length} objects`);

// Batch processing
node batch-inference.js ./images --output results.json --workers 3
```

---

### cURL Examples

```bash
# Health check
curl http://camera-ip:8080/local/detectx/health

# Capabilities
curl http://camera-ip:8080/local/detectx/capabilities

# Inference (no auth)
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg \
  -H "Content-Type: image/jpeg" \
  --data-binary @test.jpg

# Inference (with digest auth)
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg \
  --digest -u root:password \
  -H "Content-Type: image/jpeg" \
  --data-binary @test.jpg

# With image index
curl -X POST "http://camera-ip:8080/local/detectx/inference-jpeg?index=42" \
  -H "Content-Type: image/jpeg" \
  --data-binary @test.jpg
```

---

## Dataset Validation

A powerful use case for DetectX Server is testing custom models with real camera images before production deployment.

### Use Case

**Problem**: You've trained a custom TFLite model and want to validate its real-world performance on actual camera hardware before deploying to production.

**Solution**: Use DetectX Server with validation scripts to process test datasets and measure accuracy.

### Example: Python Validation Script

```python
#!/usr/bin/env python3
"""
Dataset validation example for DetectX Server.
Tests model accuracy against ground truth annotations.
"""

import json
from pathlib import Path
from inference_client import InferenceClient
from collections import Counter
from tqdm import tqdm

def validate_dataset(client, image_dir, confidence_threshold=0.25):
    """
    Process all images in a directory and analyze results.

    Args:
        client: InferenceClient instance
        image_dir: Path to directory with test images
        confidence_threshold: Minimum confidence to count detection

    Returns:
        Dictionary with validation statistics
    """
    results = {
        'total_images': 0,
        'total_detections': 0,
        'class_distribution': Counter(),
        'images_with_detections': 0,
        'avg_confidence': 0.0,
        'confidence_scores': []
    }

    image_files = list(Path(image_dir).glob('*.jpg'))

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing")):
        try:
            # Run inference with image index
            detections = client.infer_jpeg(str(image_path), image_index=idx)

            # Filter by confidence threshold
            detections = [d for d in detections
                         if d['confidence'] >= confidence_threshold]

            results['total_images'] += 1
            results['total_detections'] += len(detections)

            if detections:
                results['images_with_detections'] += 1

            # Collect statistics
            for det in detections:
                results['class_distribution'][det['label']] += 1
                results['confidence_scores'].append(det['confidence'])

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")

    # Calculate averages
    if results['confidence_scores']:
        results['avg_confidence'] = sum(results['confidence_scores']) / len(results['confidence_scores'])

    return results

# Run validation
client = InferenceClient('192.168.1.100', 'root', 'pass')

results = validate_dataset(
    client=client,
    image_dir='./test_dataset/images',
    confidence_threshold=0.30
)

# Print report
print("\n" + "="*50)
print("Dataset Validation Report")
print("="*50)
print(f"Total images processed: {results['total_images']}")
print(f"Images with detections: {results['images_with_detections']} "
      f"({results['images_with_detections']/results['total_images']*100:.1f}%)")
print(f"Total detections: {results['total_detections']}")
print(f"Average confidence: {results['avg_confidence']:.2%}")
print(f"\nClass distribution:")
for label, count in results['class_distribution'].most_common(10):
    print(f"  {label:20s}: {count:4d}")

# Save detailed results
with open('validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nDetailed results saved to validation_results.json")
```

**Run validation**:
```bash
cd scripts/python
python dataset_validation.py
```

**Sample output**:
```
Processing: 100%|████████████| 500/500 [02:15<00:00,  3.70it/s]

==================================================
Dataset Validation Report
==================================================
Total images processed: 500
Images with detections: 487 (97.4%)
Total detections: 1834
Average confidence: 76.3%

Class distribution:
  person              : 645
  car                 : 423
  bicycle             : 189
  dog                 : 157
  truck               : 92
  traffic light       : 78
  chair               : 64
  backpack            : 53
  handbag             : 48
  bottle              : 42

Detailed results saved to validation_results.json
```

### Benefits of Dataset Validation

- ✅ Test models on actual camera hardware (not simulated)
- ✅ Validate preprocessing (letterbox/crop) affects accuracy
- ✅ Tune confidence thresholds for your environment
- ✅ Compare multiple models quantitatively
- ✅ Identify problematic classes or edge cases
- ✅ Generate reports for stakeholders

---

## Custom Models

The COCO model is just an example. Replace it with your own custom TensorFlow Lite model:

### Step 1: Prepare Your Model

**Requirements**:
- **Format**: TensorFlow Lite (`.tflite`)
- **Quantization**: INT8 (required for ARTPEC-9 DLPU acceleration)
- **Input**: RGB tensor, any square dimension (e.g., 320×320, 640×640, 1024×1024)
- **Output**: Detection format (YOLO or SSD)

### Step 2: Replace Model Files

```bash
# Navigate to model directory
cd app/model/

# Replace model
cp /path/to/your/custom_model.tflite model.tflite

# Replace labels (one class per line)
cat > labels.txt << EOF
background
your_class_1
your_class_2
your_class_3
EOF
```

### Step 3: Rebuild and Deploy

```bash
# Rebuild ACAP
cd ../..
./build.sh

# Install new .eap on camera
# Upload DetectX_COCO_Hub_1_0_0_aarch64.eap via camera web interface
```

### Step 4: Verify New Model

```bash
# Check capabilities show new classes
curl http://camera-ip:8080/local/detectx/capabilities

# Test inference
curl -X POST http://camera-ip:8080/local/detectx/inference-jpeg \
  -H "Content-Type: image/jpeg" \
  --data-binary @test_image.jpg
```

**Note**: The build process automatically extracts model parameters (dimensions, classes, quantization) using TensorFlow Python during Docker build.

---

## Configuration

Runtime settings are in `app/settings/settings.json`:

```json
{
  "model": {
    "path": "model/model.tflite",
    "labels": "model/labels.txt",
    "scaleMode": "letterbox",
    "objectness": 0.25,
    "confidence": 0.30,
    "nms": 0.05
  },
  "server": {
    "max_queue_size": 3,
    "max_image_size_mb": 10
  }
}
```

**Parameters**:
- **scaleMode**: `letterbox` (preserve aspect ratio), `crop`, or `stretch`
- **objectness**: YOLO objectness threshold (0.0-1.0)
- **confidence**: Minimum detection confidence (0.0-1.0)
- **nms**: Non-maximum suppression IoU threshold (0.0-1.0)
- **max_queue_size**: Maximum concurrent inference requests (default: 3)
- **max_image_size_mb**: Maximum JPEG size in megabytes (default: 10)

**Note**: Changes to `settings.json` require rebuilding the ACAP.

---

## Performance

Performance benchmarks on ARTPEC-9 with COCO model (640×640):

| Metric | JPEG Inference | Tensor Inference |
|--------|----------------|------------------|
| **Latency** | 150-300 ms | 50-100 ms |
| **Throughput** | ~5 FPS | ~10 FPS |
| **Components** | Decode + Preprocess + Inference | Inference only |

**Breakdown** (JPEG endpoint):
- JPEG decode: ~30-50 ms
- Preprocessing (letterbox): ~20-30 ms
- Inference (DLPU): ~100-150 ms
- Post-processing (NMS): ~10-20 ms

**Recommendations**:
- Use **JPEG endpoint** for simplicity and variable image sizes
- Use **Tensor endpoint** for maximum performance (preprocess once, reuse)
- Limit concurrent requests to queue size (3) to avoid 503 errors
- Smaller models (320×320) will be ~2x faster
- Larger models (1024×1024) will be ~3x slower

---

## Development

### Build from Source

```bash
# Clone repository
git clone https://github.com/your-repo/detectx-server.git
cd detectx-server

# Build with Docker (includes ACAP SDK 12.8.0)
./build.sh

# Output: DetectX_COCO_Hub_1_0_0_aarch64.eap
```

### Local Development (Inside Container)

```bash
# Enter ACAP SDK container
docker run -it --rm -v $(pwd)/app:/opt/app \
  axisecp/acap-native-sdk:12.8.0-aarch64-ubuntu24.04 \
  /bin/bash

# Inside container
cd /opt/app
. /opt/axis/acapsdk/environment-setup*
make clean
make

# Build package
acap-build . -a settings/settings.json -a model/model.tflite -a model/labels.txt
```

### Project Structure

```
detectx-server/
├── app/
│   ├── main.c              # FastCGI HTTP interface (4 endpoints)
│   ├── server.c/h          # Request queue & worker thread
│   ├── Model.c/h           # TFLite inference (larod API)
│   ├── jpeg_decoder.c/h    # TurboJPEG decoder
│   ├── preprocess.c/h      # Image preprocessing
│   ├── labelparse.c/h      # Label file parsing
│   ├── imgutils.c/h        # Image utilities
│   ├── ACAP.c/h            # ACAP SDK wrappers
│   ├── cJSON.c/h           # JSON library
│   ├── manifest.json       # ACAP package metadata
│   ├── Makefile            # Build configuration
│   ├── settings/
│   │   └── settings.json   # Runtime configuration
│   ├── model/
│   │   ├── model.tflite    # TFLite INT8 model (COCO example)
│   │   └── labels.txt      # Class labels (90 COCO classes)
│   └── lib/
│       ├── libjpeg.so*     # TurboJPEG library
│       └── libturbojpeg.so*
├── scripts/
│   ├── python/             # Python client library & examples
│   ├── nodejs/             # Node.js client library & examples
│   └── node-red/           # Node-RED flow examples
├── Dockerfile              # Multi-stage ACAP build
├── build.sh                # Build automation script
├── CLAUDE.md               # Claude Code project guide
└── README.md               # This file
```

---

## Troubleshooting

### Build Issues

**Problem**: `cannot find -ljpeg: No such file or directory`

**Solution**: Missing library symlinks. Run:
```bash
cd app/lib
ln -sf libjpeg.so.62.4.0 libjpeg.so
ln -sf libturbojpeg.so.0.3.0 libturbojpeg.so
```

---

**Problem**: `build.sh: command not found`

**Solution**: Make script executable:
```bash
chmod +x build.sh
./build.sh
```

---

### Installation Issues

**Problem**: Cannot install `.eap` on camera

**Solution**:
- Verify camera has ARTPEC-9 chip (check camera specs)
- Ensure camera firmware supports ACAP SDK 12.8.0
- Check available storage on camera (need ~10 MB)

---

### Runtime Issues

**Problem**: `503 Service Unavailable` errors

**Solution**: Server queue is full (max 3 concurrent requests)
- Reduce parallel workers to ≤3
- Implement retry logic with exponential backoff:
```python
import time

for attempt in range(3):
    try:
        detections = client.infer_jpeg('image.jpg')
        break
    except Exception as e:
        if '503' in str(e) and attempt < 2:
            time.sleep(0.5 * (attempt + 1))  # 0.5s, 1.0s
        else:
            raise
```

---

**Problem**: `204 No Content` (empty detections)

**Solution**: This is normal - no objects detected. Try:
- Use images with clear, recognizable objects
- Check which classes your model supports: `curl http://camera-ip:8080/local/detectx/capabilities`
- Lower confidence threshold in `settings.json`
- Verify model is appropriate for your scene

---

**Problem**: Slow inference (>500ms)

**Solution**:
- Verify running on ARTPEC-9 camera (check `/local/detectx/health`)
- Ensure model is INT8 quantized (not FP32)
- Check camera isn't thermally throttling
- Use smaller model (e.g., 320×320 instead of 640×640)

---

**Problem**: High memory usage

**Solution**:
- Reduce model input size
- Decrease `max_detections` in settings
- Limit concurrent requests (reduce `max_queue_size`)

---

**Problem**: Authentication failures

**Solution**:
- Use digest authentication (not basic)
- Verify username/password are correct
- Check user has at least "viewer" role on camera
- Test with curl: `curl --digest -u root:password http://camera-ip:8080/local/detectx/health`

---

## FAQ

**Q: Can I run this on ARTPEC-7 or ARTPEC-8?**
A: No. The server requires ARTPEC-9 for hardware acceleration. However, ARTPEC-7/8 cameras can be *clients* using detectx-client ACAP.

**Q: Can I use my own custom model?**
A: Yes! Replace `app/model/model.tflite` and `app/model/labels.txt` with your custom TFLite INT8 model and rebuild.

**Q: What's the maximum image size?**
A: 10 MB by default (configurable in `settings.json`). Images are automatically resized to model input dimensions.

**Q: How many concurrent requests can the server handle?**
A: 3 concurrent requests maximum (queue size). Additional requests will get HTTP 503 and should retry with backoff.

**Q: Can I run server and client on the same camera?**
A: Yes! Install both ACAPs on the same ARTPEC-9 camera and use localhost communication.

**Q: Does this work with non-Axis cameras?**
A: No. This is an ACAP application designed specifically for Axis cameras with ARTPEC chips.

**Q: What's the difference from running TFLite directly?**
A: DetectX Server integrates with larod API for ARTPEC-9 DLPU hardware acceleration, provides HTTP API, handles queuing, and includes JPEG decoding.

**Q: Can I change the port from 8080?**
A: The port is defined in `manifest.json`. You'd need to modify and rebuild the ACAP.

**Q: How do I update the model without reinstalling?**
A: Currently requires rebuilding and reinstalling the ACAP. Hot-reload is not implemented.

---

## License

Copyright © 2026 Fred Juhlin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

**Attribution Requirement**: Anyone using this software must include the NOTICE file and provide attribution to Fred Juhlin. See [NOTICE](NOTICE) for details.

---

## Links

### Related Projects
- **🎥 [detectx-client](https://github.com/pandosme/detectx-client)** - Camera client ACAP (install on ARTPEC-7/8/9 cameras)

### Documentation & Resources
- **[ACAP SDK Documentation](https://axiscommunications.github.io/acap-documentation/)** - Official Axis ACAP documentation
- **[CLAUDE.md](CLAUDE.md)** - Detailed technical documentation for this project
- **[scripts/](scripts/)** - Client examples (Python, Node.js, Node-RED)

### Support
- **[GitHub Issues](https://github.com/your-repo/detectx-server/issues)** - Report bugs
- **[GitHub Discussions](https://github.com/your-repo/detectx-server/discussions)** - Ask questions

---

## Author

**Fred Juhlin**
Email: fred.juhlin@gmail.com

DetectX Server brings AI object detection to older Axis cameras through a client/server architecture.

---

## Acknowledgments

**Key Technologies**:
- ACAP SDK 12.8.0 (Axis Camera Application Platform)
- TensorFlow Lite (inference engine)
- larod API (ARTPEC-9 hardware acceleration)
- TurboJPEG (fast JPEG decoding)
- FastCGI (HTTP interface)

---

## Support

For questions, issues, or contributions:

- **Author**: Fred Juhlin (fred.juhlin@gmail.com)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed technical documentation
- **Examples**: Check [scripts/](scripts/) for Python, Node.js, and Node-RED examples
- **Issues**: [GitHub Issues](https://github.com/your-repo/detectx-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/detectx-server/discussions)
