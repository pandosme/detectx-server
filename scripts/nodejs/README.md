# Node.js Client Examples

Node.js client library and examples for the Axis Inference Server.

## Installation

```bash
npm install
```

## Quick Start

### Single Image Inference

```javascript
const InferenceClient = require('./inference-client');

const client = new InferenceClient('192.168.1.100', 'root', 'pass');

// JPEG inference
const detections = await client.inferJpeg('image.jpg');

detections.forEach(det => {
    console.log(`${det.label}: ${(det.confidence * 100).toFixed(1)}%`);
});
```

### Batch Processing

```bash
node batch-inference.js ./images \
  --host 192.168.1.100 \
  --username root \
  --password pass \
  --output results.json \
  --workers 3
```

## API Reference

### InferenceClient

#### Constructor

```javascript
const client = new InferenceClient(host, username, password);
```

- `host`: Camera IP or hostname
- `username`: Optional digest auth username
- `password`: Optional digest auth password

#### Methods

**`getCapabilities()`**
```javascript
const capabilities = await client.getCapabilities();
console.log(capabilities.model.classes);
```

**`getHealth()`**
```javascript
const health = await client.getHealth();
console.log(`Queue size: ${health.queue_size}`);
```

**`inferJpeg(imagePath, imageIndex)`**
```javascript
const detections = await client.inferJpeg('photo.jpg', 0);
```

**`inferTensor(tensorBuffer, width, height, imageIndex)`**
```javascript
const tensor = await client.preprocessImageToTensor('photo.jpg');
const detections = await client.inferTensor(tensor, 640, 640, 0);
```

**`preprocessImageToTensor(imagePath, targetWidth, targetHeight)`**
```javascript
const tensor = await client.preprocessImageToTensor('photo.jpg', 640, 640);
// Returns Buffer with RGB data ready for tensor endpoint
```

## Examples

### Stream Processing with Express

```javascript
const express = require('express');
const multer = require('multer');
const InferenceClient = require('./inference-client');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const client = new InferenceClient('192.168.1.100', 'root', 'pass');

app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        const detections = await client.inferJpeg(req.file.buffer);
        res.json({ detections });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000);
```

### Real-time Processing

```javascript
const chokidar = require('chokidar');
const client = new InferenceClient('192.168.1.100', 'root', 'pass');

// Watch directory for new images
const watcher = chokidar.watch('./incoming', {
    ignored: /^\./,
    persistent: true
});

watcher.on('add', async (path) => {
    console.log(`Processing: ${path}`);
    try {
        const detections = await client.inferJpeg(path);
        console.log(`Found ${detections.length} objects`);

        // Process detections...

    } catch (error) {
        console.error(`Error processing ${path}:`, error.message);
    }
});
```

### Error Handling with Retry

```javascript
async function inferWithRetry(client, imagePath, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            return await client.inferJpeg(imagePath);
        } catch (error) {
            if (error.message.includes('busy') && i < maxRetries - 1) {
                // Exponential backoff
                await new Promise(resolve => setTimeout(resolve, 500 * (i + 1)));
                continue;
            }
            throw error;
        }
    }
}

const detections = await inferWithRetry(client, 'image.jpg');
```

### Integration with TypeScript

```typescript
import InferenceClient from './inference-client';

interface Detection {
    index: number;
    label: string;
    class_id: number;
    confidence: number;
    bbox_pixels: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
    bbox_yolo: {
        x: number;
        y: number;
        w: number;
        h: number;
    };
}

const client = new InferenceClient('192.168.1.100', 'root', 'pass');
const detections: Detection[] = await client.inferJpeg('image.jpg');
```

## Performance Tips

1. **Reuse client instance** - Connection pooling is maintained
2. **Use tensor endpoint** for maximum performance
3. **Limit parallel requests** to queue size (3)
4. **Handle 503 errors** with exponential backoff
5. **Preprocess images** in batches when possible

## Response Format

```javascript
{
    index: 0,                    // Image index
    label: "car",                // Class name
    class_id: 2,                 // Numeric class ID
    confidence: 0.87,            // 0.0-1.0
    bbox_pixels: {               // Top-left, pixels
        x: 100,
        y: 50,
        w: 200,
        h: 150
    },
    bbox_yolo: {                 // Center, normalized
        x: 0.234,
        y: 0.156,
        w: 0.312,
        h: 0.234
    }
}
```

## Batch Processing

The `batch-inference.js` script provides:
- Parallel processing with configurable workers
- Progress bar with success rate
- Automatic retry with exponential backoff
- Error handling and reporting
- Statistics and performance metrics
- JSON output

```bash
# Process all images in directory
node batch-inference.js ./test-images --output results.json

# Custom settings
node batch-inference.js ./images \
  --host 10.0.0.100 \
  --username admin \
  --password secret \
  --workers 5 \
  --output detections.json
```

## Dependencies

- **axios**: HTTP client with digest auth support
- **sharp**: Fast image processing (resize, format conversion)
- **progress**: Terminal progress bars
