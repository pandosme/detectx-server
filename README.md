# DetectX Server

High-performance TensorFlow Lite inference server for Axis cameras, optimized for ARTPEC-9 (aarch64) hardware acceleration.

## Overview

DetectX Server is a FastAPI-based inference service that provides real-time object detection using TensorFlow Lite models. It's designed to run on edge devices and supports JPEG and raw tensor inputs with multiple preprocessing modes. The server ships with a COCO dataset model (90 classes) but can be configured to use any TFLite INT8 quantized model.

## Relationship to DetectX

This project is based on [DetectX](https://github.com/pandosme/DetectX), a standalone ACAP application for object detection on Axis cameras.

**Key Differences:**

| Feature | [DetectX](https://github.com/pandosme/DetectX) (Original) | DetectX Server (This Project) |
|---------|-------------|------------------------------|
| **Architecture** | All-in-one on single camera | Client/Server split |
| **Platform** | ARTPEC-9 only | Server: ARTPEC-9<br>Client: ARTPEC-7/8/9 |
| **Inference** | Local on same camera | Remote via HTTP |
| **Use Case** | Single powerful camera | Multiple cameras sharing one inference server |

**Advantages of Client/Server:**
- ✅ **Older cameras supported**: ARTPEC-7 and ARTPEC-8 cameras can now do object detection by using an ARTPEC-9 camera as the inference server
- ✅ **Resource sharing**: One powerful ARTPEC-9 camera can serve multiple client cameras
- ✅ **Flexible deployment**: Server and client can run on the same camera OR separate cameras
- ✅ **Lower cost**: Don't need ARTPEC-9 for every camera

**When to Use Each:**

- **Use DetectX (original)** if you have ARTPEC-9 cameras and want standalone operation
- **Use DetectX Server** if you want to:
  - Enable object detection on older ARTPEC-7/8 cameras
  - Share inference compute across multiple cameras
  - Centralize model management on one camera
  - **Test models with dataset images** using scripts before deployment

## Features

- **Multiple Input Formats**: JPEG images and preprocessed RGB tensors
- **Flexible Preprocessing**: Crop, letterbox, and balanced scaling modes
- **Hardware Acceleration**: Optimized for ARTPEC-9 DLPU
- **REST API**: Simple HTTP interface for easy integration
- **Health Monitoring**: Built-in health checks and statistics
- **Model Hot-Loading**: Update models without service restart

## Architecture

```
┌─────────────┐
│   Client    │ (DetectX Client ACAP or any HTTP client)
│  (Camera)   │
└──────┬──────┘
       │ HTTP POST
       │ JPEG/Tensor
       ▼
┌─────────────┐
│   FastAPI   │
│   Server    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ TFLite Int8 │
│   Model     │
│  (YOLO/SSD) │
└─────────────┘
```

## Example Model: COCO Dataset

The server ships with a pre-trained **COCO (Common Objects in Context)** model supporting **90 object classes**:

**People & Body Parts**: person
**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
**Traffic**: traffic light, fire hydrant, stop sign, parking meter
**Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
**Accessories**: backpack, umbrella, handbag, tie, suitcase
**Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
**Kitchen**: bottle, wine glass, cup, fork, knife, spoon, bowl
**Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
**Furniture**: chair, couch, potted plant, bed, dining table, toilet
**Electronics**: tv, laptop, mouse, remote, keyboard, cell phone
**Appliances**: microwave, oven, toaster, sink, refrigerator
**Indoor**: book, clock, vase, scissors, teddy bear, hair drier, toothbrush

The model achieves good accuracy on these common everyday objects, making it ideal for general-purpose surveillance and monitoring applications.

## Requirements

- **Platform**: ARTPEC-9 (aarch64) or compatible
- **Docker**: For containerized deployment
- **Model**: TensorFlow Lite INT8 quantized model (.tflite) - **COCO example included**
- **Labels**: Class labels file - **COCO example included** (90 classes)

## Quick Start

### 1. Model Files

The server ships with a COCO example model in `app/model/`:
- `model.tflite` - YOLOv8 INT8 quantized (2.1 MB, 640x640 input, 90 COCO classes)
- `labels.txt` - 90 COCO class labels

**To use your own model**, simply replace these files with your custom TFLite INT8 model and labels.

### 2. Build and Run with Docker

```bash
# Build for ARTPEC-9
docker build --build-arg CHIP=aarch64 -t detectx-server .

# Run
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  --name detectx-server \
  detectx-server
```

### 3. Test the Server

```bash
# Health check
curl http://localhost:8080/local/detectx/health

# Get capabilities
curl http://localhost:8080/local/detectx/capabilities

# Run inference
curl -X POST \
  http://localhost:8080/local/detectx/inference-jpeg \
  -H "Content-Type: image/jpeg" \
  --data-binary @test_image.jpg
```

## API Endpoints

### GET `/local/detectx/capabilities`
Returns model information and supported features.

**Response:**
```json
{
  "model": {
    "input_width": 640,
    "input_height": 640,
    "classes": ["bird", "person", "car"],
    "input_formats": [
      {"endpoint": "inference-jpeg", "mime": "image/jpeg"},
      {"endpoint": "inference-tensor", "mime": "application/octet-stream"}
    ]
  }
}
```

### POST `/local/detectx/inference-jpeg`
Perform inference on JPEG image.

**Query Parameters:**
- `index` (optional): Image index for dataset validation
- `scale_mode` (optional): `crop`, `letterbox`, or `balanced` (default: `crop`)

**Request:** Binary JPEG data
**Response:**
```json
{
  "detections": [
    {
      "index": 0,
      "label": "bird",
      "class_id": 0,
      "confidence": 0.95,
      "bbox_pixels": {"x": 100, "y": 150, "w": 200, "h": 180},
      "bbox_yolo": {"cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3},
      "image": {"width": 640, "height": 640}
    }
  ]
}
```

### POST `/local/detectx/inference-tensor`
Perform inference on preprocessed RGB tensor.

**Request:** Raw RGB bytes (width × height × 3)
**Response:** Same as inference-jpeg

### GET `/local/detectx/health`
Get server health and statistics.

## Configuration

Edit `config.yaml`:

```yaml
model:
  path: "./models/model.tflite"
  labels: "./models/labels.txt"

server:
  host: "0.0.0.0"
  port: 8080
  workers: 1

inference:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 100
```

## Client Integration

This server is designed to work with [DetectX Client](https://github.com/pandosme/detectx-client), an ACAP application for Axis cameras supporting ARTPEC-7, ARTPEC-8, and ARTPEC-9.

**Deployment Options:**
1. **Separate Cameras**: Server on ARTPEC-9 camera, clients on other cameras (any ARTPEC-7/8/9)
2. **Same Camera**: Both server and client on the same ARTPEC-9 camera (localhost communication)
3. **Hybrid**: Mix of same-camera and remote-camera clients

Any HTTP client can also use the REST API directly.

**Python Example:**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://camera-ip:8080/local/detectx/inference-jpeg',
        data=f.read(),
        headers={'Content-Type': 'image/jpeg'}
    )

detections = response.json()['detections']
for det in detections:
    print(f"{det['label']}: {det['confidence']:.2%}")
```

## Dataset Validation

A key use case for DetectX Server is **dataset validation** - testing how well your model performs in real-world camera conditions before deployment.

**Use Case**: You have a dataset of images and want to understand model performance in your specific camera environment.

**Python Script Example:**
```python
#!/usr/bin/env python3
"""
Dataset validation script for DetectX Server
Tests model performance against ground truth annotations
"""

import os
import json
from pathlib import Path
import requests
from collections import Counter

# Server configuration
SERVER_URL = "http://camera-ip:8080"
USERNAME = "root"  # Optional
PASSWORD = "pass"   # Optional

def validate_dataset(image_dir, ground_truth_file, confidence_threshold=0.25):
    """
    Run inference on dataset images and compare with ground truth

    Args:
        image_dir: Directory containing JPEG images
        ground_truth_file: JSON file with annotations
        confidence_threshold: Minimum confidence to count detection
    """
    # Load ground truth
    with open(ground_truth_file) as f:
        ground_truth = json.load(f)

    results = {
        'total_images': 0,
        'total_detections': 0,
        'class_distribution': Counter(),
        'images_with_detections': 0,
        'avg_confidence': 0.0
    }

    # Process each image
    for image_file in Path(image_dir).glob('*.jpg'):
        with open(image_file, 'rb') as f:
            response = requests.post(
                f"{SERVER_URL}/local/detectx/inference-jpeg",
                data=f.read(),
                headers={'Content-Type': 'image/jpeg'},
                auth=(USERNAME, PASSWORD) if USERNAME else None
            )

        if response.status_code == 200:
            detections = response.json()['detections']

            # Filter by confidence
            detections = [d for d in detections if d['confidence'] >= confidence_threshold]

            results['total_images'] += 1
            results['total_detections'] += len(detections)
            if detections:
                results['images_with_detections'] += 1

            # Collect statistics
            for det in detections:
                results['class_distribution'][det['label']] += 1
                results['avg_confidence'] += det['confidence']

            print(f"{image_file.name}: {len(detections)} detections")
        else:
            print(f"Error processing {image_file.name}: {response.status_code}")

    # Calculate averages
    if results['total_detections'] > 0:
        results['avg_confidence'] /= results['total_detections']

    return results

# Run validation
results = validate_dataset(
    image_dir='./dataset/images',
    ground_truth_file='./dataset/annotations.json',
    confidence_threshold=0.25
)

# Print summary
print("\n=== Dataset Validation Summary ===")
print(f"Total images: {results['total_images']}")
print(f"Images with detections: {results['images_with_detections']}")
print(f"Total detections: {results['total_detections']}")
print(f"Average confidence: {results['avg_confidence']:.2%}")
print(f"\nClass distribution:")
for label, count in results['class_distribution'].most_common():
    print(f"  {label}: {count}")
```

**Benefits:**
- Test model accuracy before camera deployment
- Understand class distribution in your environment
- Validate confidence thresholds
- Compare different models or preprocessing modes
- Generate performance reports

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Run tests
pytest tests/
```

## Performance

- **JPEG Inference**: ~150-300ms (including decode + inference)
- **Tensor Inference**: ~50-100ms (inference only)
- **Throughput**: ~5-10 FPS on ARTPEC-9

## Troubleshooting

**Server won't start:**
- Check model file exists and is valid TFLite INT8
- Verify port 8080 is not already in use
- Check Docker logs: `docker logs detectx-server`

**Low performance:**
- Ensure running on ARTPEC-9 hardware
- Verify model is INT8 quantized
- Check CPU usage and thermal throttling

**High memory usage:**
- Reduce model input size
- Decrease max_detections in config
- Limit concurrent requests

## License

[Your License Here]

## Links

- **Original Project**: [DetectX](https://github.com/pandosme/DetectX) by Fredrik Persson
- **Client**: [DetectX Client](https://github.com/pandosme/detectx-client)
- **Issues**: [Report bugs](https://github.com/pandosme/detectx-server/issues)
