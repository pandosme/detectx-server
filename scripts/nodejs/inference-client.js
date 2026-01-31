/**
 * Axis Inference Server Client for Node.js
 *
 * Simple client library for interfacing with the Axis camera inference server.
 */

const axios = require('axios');
const { AxiosDigestAuth } = require('@mhoc/axios-digest-auth');
const fs = require('fs').promises;
const sharp = require('sharp');

class InferenceClient {
    /**
     * Create an inference client
     * @param {string} host - Camera IP or hostname
     * @param {string} username - Optional digest auth username
     * @param {string} password - Optional digest auth password
     */
    constructor(host, username = null, password = null) {
        this.baseUrl = `http://${host}/local/detectx`;

        if (username && password) {
            // Use digest authentication for Axis cameras
            this.client = new AxiosDigestAuth({
                username: username,
                password: password
            });
        } else {
            this.client = axios.create({
                baseURL: this.baseUrl,
                timeout: 30000
            });
        }

        this.baseUrl = `http://${host}/local/detectx`;
    }

    /**
     * Get server capabilities and model information
     * @returns {Promise<Object>} Capabilities object
     */
    async getCapabilities() {
        const url = `${this.baseUrl}/capabilities`;
        const response = await this.client.request({ method: 'GET', url });
        return response.data;
    }

    /**
     * Get server health and statistics
     * @returns {Promise<Object>} Health status object
     */
    async getHealth() {
        const url = `${this.baseUrl}/health`;
        const response = await this.client.request({ method: 'GET', url });
        return response.data;
    }

    /**
     * Perform inference on a JPEG image
     * @param {string|Buffer} imagePath - Path to JPEG file or Buffer
     * @param {number} imageIndex - Optional image index for dataset validation
     * @returns {Promise<Array>} Array of detections
     */
    async inferJpeg(imagePath, imageIndex = -1) {
        let imageData;

        if (Buffer.isBuffer(imagePath)) {
            imageData = imagePath;
        } else {
            imageData = await fs.readFile(imagePath);
        }

        const headers = {
            'Content-Type': 'image/jpeg'
        };

        let url = `${this.baseUrl}/inference-jpeg`;
        if (imageIndex >= 0) {
            url += `?index=${imageIndex}`;
        }

        try {
            const response = await this.client.request({
                method: 'POST',
                url: url,
                data: imageData,
                headers: headers
            });

            if (response.status === 200) {
                return response.data.detections;
            } else if (response.status === 204) {
                return [];  // No detections
            }
        } catch (error) {
            if (error.response) {
                if (error.response.status === 503) {
                    throw new Error('Server busy - queue full');
                } else if (error.response.status === 204) {
                    return [];  // No detections
                }
            }
            throw error;
        }
    }

    /**
     * Perform inference on a preprocessed RGB tensor
     * @param {Buffer} tensorBuffer - RGB tensor data (width * height * 3 bytes)
     * @param {number} width - Image width
     * @param {number} height - Image height
     * @param {number} imageIndex - Optional image index
     * @returns {Promise<Array>} Array of detections
     */
    async inferTensor(tensorBuffer, width, height, imageIndex = -1) {
        const expectedSize = width * height * 3;

        if (tensorBuffer.length !== expectedSize) {
            throw new Error(
                `Invalid tensor size. Expected ${expectedSize} bytes (${width}x${height}x3), ` +
                `got ${tensorBuffer.length} bytes`
            );
        }

        const headers = {
            'Content-Type': 'application/octet-stream'
        };

        let url = `${this.baseUrl}/inference-tensor`;
        if (imageIndex >= 0) {
            url += `?index=${imageIndex}`;
        }

        try {
            const response = await this.client.request({
                method: 'POST',
                url: url,
                data: tensorBuffer,
                headers: headers
            });

            if (response.status === 200) {
                return response.data.detections;
            } else if (response.status === 204) {
                return [];
            }
        } catch (error) {
            if (error.response) {
                if (error.response.status === 503) {
                    throw new Error('Server busy - queue full');
                } else if (error.response.status === 204) {
                    return [];
                }
            }
            throw error;
        }
    }

    /**
     * Preprocess image to tensor format with letterboxing
     * @param {string} imagePath - Path to image file
     * @param {number} targetWidth - Target width (default 640)
     * @param {number} targetHeight - Target height (default 640)
     * @returns {Promise<Buffer>} RGB tensor buffer
     */
    async preprocessImageToTensor(imagePath, targetWidth = 640, targetHeight = 640) {
        const image = sharp(imagePath);
        const metadata = await image.metadata();

        // Calculate scale to maintain aspect ratio
        const scale = Math.min(targetWidth / metadata.width, targetHeight / metadata.height);
        const newWidth = Math.round(metadata.width * scale);
        const newHeight = Math.round(metadata.height * scale);

        // Calculate offsets for centering
        const offsetX = Math.floor((targetWidth - newWidth) / 2);
        const offsetY = Math.floor((targetHeight - newHeight) / 2);

        // Create letterboxed image with black background
        const tensorBuffer = await image
            .resize(newWidth, newHeight)
            .extend({
                top: offsetY,
                bottom: targetHeight - newHeight - offsetY,
                left: offsetX,
                right: targetWidth - newWidth - offsetX,
                background: { r: 0, g: 0, b: 0 }
            })
            .raw()
            .toBuffer();

        return tensorBuffer;
    }
}

module.exports = InferenceClient;

// Example usage
if (require.main === module) {
    (async () => {
        const client = new InferenceClient('192.168.1.100', 'root', 'pass');

        try {
            // Get capabilities
            console.log('=== Server Capabilities ===');
            const capabilities = await client.getCapabilities();
            console.log(`Model: ${capabilities.model.input_width}x${capabilities.model.input_height}`);
            console.log(`Classes: ${capabilities.model.classes.length}`);
            console.log();

            // Get health
            console.log('=== Server Health ===');
            const health = await client.getHealth();
            console.log(`Running: ${health.running}`);
            console.log(`Queue size: ${health.queue_size}`);
            console.log(`Total requests: ${health.statistics.total_requests}`);
            console.log();

            // Inference on image
            if (process.argv.length < 3) {
                console.log('Usage: node inference-client.js <image.jpg>');
                return;
            }

            const imagePath = process.argv[2];

            // Method 1: JPEG inference
            console.log('=== JPEG Inference ===');
            const detections = await client.inferJpeg(imagePath, 0);
            console.log(`Found ${detections.length} objects:`);
            detections.forEach(det => {
                console.log(`  - ${det.label}: ${(det.confidence * 100).toFixed(1)}% at ` +
                    `(${det.bbox_pixels.x}, ${det.bbox_pixels.y}) ` +
                    `${det.bbox_pixels.w}x${det.bbox_pixels.h}`);
            });
            console.log();

            // Method 2: Tensor inference
            console.log('=== Tensor Inference ===');
            const tensor = await client.preprocessImageToTensor(imagePath);
            console.log(`Preprocessed tensor size: ${tensor.length} bytes`);
            const tensorDetections = await client.inferTensor(tensor, 640, 640, 0);
            console.log(`Found ${tensorDetections.length} objects`);

        } catch (error) {
            console.error('Error:', error.message);
            process.exit(1);
        }
    })();
}
