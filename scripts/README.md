# Inference Server Usage Examples

This directory contains client examples and usage patterns for the Axis Camera Inference Server in multiple languages and platforms.

## Quick Links

- **[Python Examples](python/)** - Python client library with batch processing
- **[Node.js Examples](nodejs/)** - JavaScript/Node.js client and examples
- **[Node-RED Flows](node-red/)** - Visual programming flows for Node-RED

## Overview

The inference server provides a REST API with two main endpoints:

### POST /inference/jpeg
Submit JPEG images for inference (easy, automatic preprocessing)
- **Input**: JPEG image (any size)
- **Processing**: Server decodes → letterbox → inference
- **Best for**: Simple scripts, variable image sizes

### POST /inference/tensor
Submit pre-processed RGB tensors (optimal performance)
- **Input**: Raw RGB data at exact model dimensions (e.g., 640x640x3)
- **Processing**: Direct inference (no overhead)
- **Best for**: Maximum performance, batch processing

### GET /capabilities
Query server information, model details, and supported input formats

### GET /health
Check server status, queue size, and statistics

## Quick Start by Platform

### Python

```bash
cd python
pip install -r requirements.txt
python inference_client.py image.jpg
```

### Node.js

```bash
cd nodejs
npm install
node inference-client.js image.jpg
```

### Node-RED

1. Import `node-red/simple-inference-flow.json`
2. Configure camera IP and credentials
3. Deploy and trigger

## Common Use Cases

### 1. Single Image Inference

**Python:**
```python
from inference_client import InferenceClient

client = InferenceClient('192.168.1.100', 'root', 'pass')
detections = client.infer_jpeg('photo.jpg')
```

**Node.js:**
```javascript
const client = new InferenceClient('192.168.1.100', 'root', 'pass');
const detections = await client.inferJpeg('photo.jpg');
```

### 2. Batch Processing

**Python:**
```bash
python batch_inference.py /path/to/images --output results.json
```

**Node.js:**
```bash
node batch-inference.js /path/to/images --output results.json
```

**Node-RED:**
Import `batch-inference-flow.json`

### 3. Real-time Stream Processing

**Python:**
```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # Preprocess frame
    tensor = preprocess_frame(frame)
    # Fast tensor inference
    detections = client.infer_tensor(tensor, 640, 640)
```

**Node.js:**
```javascript
// Watch directory for new images
chokidar.watch('./incoming')
    .on('add', async (path) => {
        const detections = await client.inferJpeg(path);
    });
```

### 4. Dataset Validation

**Python:**
```python
# Process test dataset
for idx, image_path in enumerate(test_images):
    detections = client.infer_jpeg(image_path, image_index=idx)
    # Compare with ground_truth[idx]
```

## API Response Format

All endpoints return detections in the same format:

```json
{
    "detections": [
        {
            "index": 0,
            "label": "car",
            "class_id": 2,
            "confidence": 0.87,
            "bbox_pixels": {
                "x": 100,
                "y": 50,
                "w": 200,
                "h": 150
            },
            "bbox_yolo": {
                "x": 0.234,
                "y": 0.156,
                "w": 0.312,
                "h": 0.234
            }
        }
    ]
}
```

**Coordinate Formats:**
- **bbox_pixels**: Top-left corner (x, y) + dimensions, in pixels
- **bbox_yolo**: Center (x, y) + dimensions, normalized 0.0-1.0

## HTTP Status Codes

- **200 OK**: Inference successful, detections found
- **204 No Content**: Inference successful, no detections
- **400 Bad Request**: Invalid input (wrong format, size, etc.)
- **503 Service Unavailable**: Queue full (server busy)
- **500 Internal Server Error**: Inference failed

## Performance Optimization

### 1. Choose the Right Endpoint

| Use Case | Endpoint | Reason |
|----------|----------|--------|
| Simple scripts | `/inference/jpeg` | Easy, no preprocessing needed |
| Maximum speed | `/inference/tensor` | No decoding overhead |
| Variable sizes | `/inference/jpeg` | Auto letterboxing |
| Batch processing | `/inference/tensor` | Preprocess once, cache |

### 2. Parallel Processing

- Server queue size: **3 requests**
- Use 3 parallel workers maximum
- Implement exponential backoff for 503 errors

**Python:**
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(client.infer_jpeg, img) for img in images]
```

**Node.js:**
```javascript
const workers = 3;
for (let i = 0; i < workers; i++) {
    processWorker(images, i, workers);
}
```

### 3. Error Handling

Always handle queue-full scenarios:

```python
try:
    detections = client.infer_jpeg('image.jpg')
except Exception as e:
    if '503' in str(e):
        time.sleep(0.5)  # Wait and retry
    else:
        raise
```

### 4. Preprocessing Tips

For tensor endpoint:
- Use letterbox preprocessing (preserve aspect ratio)
- Cache preprocessed tensors when possible
- Batch preprocessing before inference

## Authentication

All examples support digest authentication:

```python
client = InferenceClient('192.168.1.100', 'username', 'password')
```

The camera manages authentication. Typical setup:
- **Username**: `root` or camera admin user
- **Password**: Camera password
- **Method**: Digest authentication (automatic)

## Example Workflows

### Computer Vision Pipeline

```
1. Capture/Load Image
   ↓
2. Preprocess (optional for tensor endpoint)
   ↓
3. Inference (via REST API)
   ↓
4. Filter Detections (confidence threshold)
   ↓
5. Post-process (NMS already applied by server)
   ↓
6. Draw Bounding Boxes / Save Results
```

### Surveillance System

```
Camera → Inference Server → Filter High-Value Detections → Alert System
                ↓
         Store Detections in Database
                ↓
         Generate Statistics Dashboard
```

### Dataset Validation

```
Load Test Images → Inference with Index → Compare with Ground Truth → Calculate mAP
```

## Language-Specific Features

| Feature | Python | Node.js | Node-RED |
|---------|--------|---------|----------|
| JPEG inference | ✅ | ✅ | ✅ |
| Tensor inference | ✅ | ✅ | ❌ |
| Batch processing | ✅ | ✅ | ✅ |
| Preprocessing | ✅ | ✅ | Limited |
| Progress tracking | ✅ | ✅ | ❌ |
| Error retry | ✅ | ✅ | Manual |
| Type hints | ✅ | TypeScript | ❌ |

## Dependencies

### Python
- requests, numpy, PIL/Pillow, tqdm

### Node.js
- axios, sharp, progress

### Node-RED
- Built-in nodes sufficient
- Optional: node-red-contrib-image-tools

## Best Practices

1. **Check capabilities first**: Query `/capabilities` to get model info
2. **Validate input size**: Ensure images aren't too large (10MB limit)
3. **Handle all status codes**: Especially 204 (no detections) and 503 (busy)
4. **Limit parallel requests**: Match server queue size (3)
5. **Use appropriate endpoint**: JPEG for ease, tensor for speed
6. **Implement retry logic**: Exponential backoff for 503 errors
7. **Monitor server health**: Check `/health` for queue and statistics
8. **Cache preprocessed data**: When using tensor endpoint repeatedly

## Testing

Each platform includes test commands:

```bash
# Python
python inference_client.py test.jpg

# Node.js
node inference-client.js test.jpg
npm test

# Node-RED
Import simple-inference-flow.json and trigger manually
```

## Troubleshooting

**Connection refused**:
- Check camera IP address
- Verify server is running: `curl http://<camera-ip>/local/detectx/health`

**Authentication failed**:
- Verify username/password
- Ensure digest auth is used
- Check camera user permissions

**503 Service Unavailable**:
- Server queue is full (3 max)
- Reduce parallel workers
- Add retry with delay

**400 Bad Request**:
- Check image format (must be valid JPEG)
- Verify tensor size matches model (for tensor endpoint)
- Check request headers (Content-Type)

**Empty detections (204)**:
- Normal response when no objects detected
- Try images with clear, recognizable objects
- Check model classes in `/capabilities`

## Support & Resources

- **Server README**: [../README.md](../README.md)
- **API Documentation**: See server README for full API spec
- **Model Training**: See DetectX documentation for custom models
- **Issues**: Report bugs to project repository

## Contributing

To add examples for a new language/platform:

1. Create a directory: `usage/<platform>/`
2. Include:
   - Client library/wrapper
   - Example scripts
   - README.md with usage
   - Dependencies file
3. Update this README with links and quick start

## License

Same as parent project
