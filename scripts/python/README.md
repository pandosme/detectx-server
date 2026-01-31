# Python Client Examples

Python examples for using the Axis Inference Server.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single Image Inference

```python
from inference_client import InferenceClient

# Initialize client
client = InferenceClient(
    host="192.168.1.100",
    username="root",
    password="pass"
)

# Perform inference on JPEG
detections = client.infer_jpeg("image.jpg")

for det in detections:
    print(f"{det['label']}: {det['confidence']:.2%}")
```

### Check Server Capabilities

```python
capabilities = client.get_capabilities()
print(f"Model size: {capabilities['model']['input_width']}x{capabilities['model']['input_height']}")
print(f"Classes: {[c['name'] for c in capabilities['model']['classes']]}")
```

### Batch Processing

Process an entire directory of images:

```bash
python batch_inference.py /path/to/images \
  --host 192.168.1.100 \
  --username root \
  --password pass \
  --output results.json \
  --workers 3
```

## Examples

### inference_client.py

Core client library with:
- `InferenceClient` class
- JPEG inference (`infer_jpeg`)
- Tensor inference (`infer_tensor`)
- Image preprocessing (`preprocess_image_to_tensor`)
- Capabilities and health endpoints

### batch_inference.py

Batch processing script with:
- Parallel inference using ThreadPoolExecutor
- Progress tracking with tqdm
- Automatic retry with exponential backoff
- Error handling
- Statistics and performance metrics
- JSON output

## Usage Patterns

### 1. JPEG Inference (Easiest)

```python
detections = client.infer_jpeg("photo.jpg", image_index=0)
```

Best for:
- Simple scripts
- Variable image sizes
- When preprocessing isn't critical

### 2. Tensor Inference (Fastest)

```python
# Preprocess once
tensor = client.preprocess_image_to_tensor("photo.jpg")

# Reuse preprocessed tensor
detections = client.infer_tensor(tensor, image_index=0)
```

Best for:
- Maximum performance
- When you control preprocessing
- Batch processing where you can cache tensors

### 3. Custom Preprocessing

```python
import numpy as np
from PIL import Image

# Load and preprocess image
img = Image.open("photo.jpg").convert('RGB')
img = img.resize((640, 640))  # Match model size
tensor = np.array(img, dtype=np.uint8)

# Inference
detections = client.infer_tensor(tensor)
```

## Error Handling

The client handles common errors:

```python
try:
    detections = client.infer_jpeg("image.jpg")
except requests.HTTPError as e:
    if e.response.status_code == 503:
        print("Server busy - retry later")
    elif e.response.status_code == 400:
        print(f"Bad request: {e.response.text}")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Tips

1. **Use tensor endpoint** for repeated inference on same-sized images
2. **Limit parallel workers** to server queue size (default: 3)
3. **Preprocess images** in batches to reduce overhead
4. **Reuse session** - InferenceClient maintains a session
5. **Handle 503 errors** - implement exponential backoff

## Response Format

Both endpoints return the same detection format:

```python
{
    "index": 0,                    # Image index (from request)
    "label": "car",                # Class name
    "class_id": 2,                 # Numeric class ID
    "confidence": 0.87,            # 0.0-1.0
    "bbox_pixels": {               # Top-left corner, pixels
        "x": 100,
        "y": 50,
        "w": 200,
        "h": 150
    },
    "bbox_yolo": {                 # Center format, normalized 0-1
        "x": 0.234,
        "y": 0.156,
        "w": 0.312,
        "h": 0.234
    }
}
```

## Advanced Usage

### Dataset Validation

For validating model performance:

```python
# Process images with indices matching ground truth
for idx, image_path in enumerate(image_files):
    detections = client.infer_jpeg(image_path, image_index=idx)

    # Compare with ground_truth[idx]
    # Calculate precision, recall, mAP
```

### Integration with OpenCV

```python
import cv2
import numpy as np

# Load image with OpenCV
img_bgr = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize to model input size
img_resized = cv2.resize(img_rgb, (640, 640))

# Inference
detections = client.infer_tensor(img_resized)

# Draw bounding boxes
for det in detections:
    x = det['bbox_pixels']['x']
    y = det['bbox_pixels']['y']
    w = det['bbox_pixels']['w']
    h = det['bbox_pixels']['h']

    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_bgr, f"{det['label']} {det['confidence']:.2f}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('Detections', img_bgr)
cv2.waitKey(0)
```
