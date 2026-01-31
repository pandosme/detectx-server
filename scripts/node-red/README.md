# Node-RED Examples

Node-RED flows for integrating with the Axis Inference Server.

## Installation

1. Install Node-RED if not already installed:
```bash
npm install -g --unsafe-perm node-red
```

2. Start Node-RED:
```bash
node-red
```

3. Open Node-RED editor: http://localhost:1880

## Importing Flows

1. In Node-RED editor, click the menu (☰) → Import
2. Select "Select a file to import" or paste JSON
3. Choose one of the example flows:
   - `simple-inference-flow.json` - Basic single image inference
   - `batch-inference-flow.json` - Process multiple images
   - `capabilities-flow.json` - Query server capabilities

4. Click "Import"

## Configuration

After importing, configure the flows:

### HTTP Request Node (Inference API)
- **URL**: `http://<camera-ip>/local/detectx/inference-jpeg`
- **Method**: POST
- **Authentication**: Digest
  - Username: `root` (or your camera username)
  - Password: Your camera password

### File Input Node (Read Image)
- Update the file path to point to your test image

## Example Flows

### 1. Simple Inference Flow

**simple-inference-flow.json**

Basic flow that:
1. Reads a JPEG image from disk
2. Sends it to the inference server
3. Processes and displays detections

**Nodes**:
- Inject → File In → Change → HTTP Request → Function → Debug

**Configuration**:
- Update image path in "Read Image" node
- Update camera IP in "Inference API" node
- Set digest auth credentials

### 2. Batch Inference Flow

**batch-inference-flow.json**

Advanced flow that:
1. Lists all JPEG files in a directory
2. Processes each image
3. Collects results
4. Outputs summary statistics

**Features**:
- Automatic file discovery
- Sequential processing
- Result aggregation
- Class counting
- Summary statistics

**Configuration**:
- Update `imageDir` path in "List Images" function
- Configure camera credentials

### 3. Real-time Processing

Example of continuous processing:

```json
[Inject (repeat every 1s)]
  → [Capture from camera or file system]
  → [HTTP Request to inference server]
  → [Function: Filter detections]
  → [Switch: Route by class]
     → [Debug: Cars]
     → [Debug: People]
     → [Debug: Others]
```

## Custom Function Nodes

### Process Detections

```javascript
// Extract and format detections
const detections = msg.payload.detections || [];

// Filter by confidence threshold
const filtered = detections.filter(det => det.confidence > 0.7);

// Group by class
const byClass = {};
filtered.forEach(det => {
    if (!byClass[det.label]) {
        byClass[det.label] = [];
    }
    byClass[det.label].push(det);
});

msg.payload = {
    total: filtered.length,
    by_class: byClass
};

return msg;
```

### Draw Bounding Boxes (with node-red-contrib-image-tools)

```javascript
// Install: npm install node-red-contrib-image-tools

const Jimp = require('jimp');

// Load image
const image = await Jimp.read(msg.filename);

// Draw bounding boxes
msg.payload.detections.forEach(det => {
    const box = det.bbox_pixels;
    const color = Jimp.rgbaToInt(0, 255, 0, 255);  // Green

    // Draw rectangle
    for (let i = 0; i < 3; i++) {  // 3px thick
        image.scan(
            box.x + i, box.y + i,
            box.w - 2*i, box.h - 2*i,
            function(x, y, idx) {
                if (x === box.x + i || x === box.x + box.w - i - 1 ||
                    y === box.y + i || y === box.y + box.h - i - 1) {
                    this.setPixelColor(color, x, y);
                }
            }
        );
    }

    // Add label text
    Jimp.loadFont(Jimp.FONT_SANS_16_WHITE).then(font => {
        image.print(font, box.x, box.y - 20,
                   `${det.label} ${(det.confidence * 100).toFixed(0)}%`);
    });
});

// Save annotated image
await image.writeAsync('/path/to/output.jpg');

msg.payload = 'Image saved with annotations';
return msg;
```

### Error Handling with Retry

```javascript
// Check for errors
if (msg.statusCode === 503) {
    // Server busy, retry after delay
    context.set('retryCount', (context.get('retryCount') || 0) + 1);

    if (context.get('retryCount') < 3) {
        // Trigger retry after delay
        setTimeout(() => {
            node.send(msg);
        }, 1000 * context.get('retryCount'));
        return null;
    } else {
        msg.payload = { error: 'Max retries exceeded' };
        context.set('retryCount', 0);
    }
} else {
    context.set('retryCount', 0);
}

return msg;
```

## Integration Examples

### MQTT Publishing

```
[Inference Result]
  → [Function: Format for MQTT]
  → [MQTT Out]
```

Function node:
```javascript
// Format detections for MQTT
msg.topic = 'camera/detections';
msg.payload = {
    timestamp: Date.now(),
    camera_id: 'front_door',
    detections: msg.payload.detections.map(det => ({
        class: det.label,
        confidence: det.confidence,
        bbox: det.bbox_yolo
    }))
};
return msg;
```

### Dashboard Display

Install: `npm install node-red-dashboard`

```
[Inference Result]
  → [Function: Format for dashboard]
  → [Chart/Gauge/Text nodes]
```

### Webhook Trigger

```
[HTTP In]
  → [Function: Extract image from POST]
  → [HTTP Request: Inference]
  → [Function: Format response]
  → [HTTP Response]
```

## Useful Node-RED Nodes

For working with the inference server:

- **node-red-contrib-fs-ops**: File system operations
- **node-red-contrib-image-tools**: Image processing
- **node-red-dashboard**: UI dashboard
- **node-red-contrib-s3**: S3 storage integration
- **node-red-contrib-telegrambot**: Telegram notifications

Install:
```bash
cd ~/.node-red
npm install node-red-contrib-fs-ops
npm install node-red-contrib-image-tools
npm install node-red-dashboard
```

## Tips

1. **Use digest authentication** for camera access
2. **Limit parallel requests** - Server queue size is 3
3. **Handle 204 responses** - No detections found (not an error)
4. **Check server health** before batch processing
5. **Use context storage** for batch result aggregation
6. **Add retry logic** for 503 (busy) responses
7. **Filter by confidence** to reduce false positives

## Troubleshooting

**Flow doesn't execute**:
- Check that all nodes are properly wired
- Verify camera IP and credentials
- Check Node-RED debug panel for errors

**Authentication fails**:
- Ensure digest auth is selected
- Verify username/password
- Check camera user permissions

**503 Service Unavailable**:
- Server queue is full
- Add delay between requests
- Implement retry logic with exponential backoff

**Empty detections**:
- 204 response means no objects detected (normal)
- Try images with clear objects
- Check confidence thresholds
