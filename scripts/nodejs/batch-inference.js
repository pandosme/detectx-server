/**
 * Batch Inference Example for Node.js
 *
 * Process multiple images in parallel with the inference server.
 */

const fs = require('fs').promises;
const path = require('path');
const InferenceClient = require('./inference-client');
const ProgressBar = require('progress');

/**
 * Process a single image with retry logic
 */
async function processSingleImage(client, imagePath, index, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const startTime = Date.now();
            const detections = await client.inferJpeg(imagePath, index);
            const inferenceTime = (Date.now() - startTime) / 1000;

            return {
                index,
                image: path.basename(imagePath),
                success: true,
                detections,
                inferenceTime,
                attempts: attempt + 1
            };

        } catch (error) {
            const errorMsg = error.message;

            // If server is busy, wait and retry
            if (errorMsg.includes('busy') || errorMsg.includes('503')) {
                if (attempt < maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500 * (attempt + 1)));
                    continue;
                }
            }

            // Other errors
            return {
                index,
                image: path.basename(imagePath),
                success: false,
                error: errorMsg,
                attempts: attempt + 1
            };
        }
    }

    return {
        index,
        image: path.basename(imagePath),
        success: false,
        error: 'Max retries exceeded',
        attempts: maxRetries
    };
}

/**
 * Process all images in a directory
 */
async function batchInference(client, imageDir, options = {}) {
    const {
        outputFile = null,
        numWorkers = 3,
        imageExtensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    } = options;

    // Find all images
    const files = await fs.readdir(imageDir);
    const imageFiles = files
        .filter(file => imageExtensions.some(ext => file.endsWith(ext)))
        .map(file => path.join(imageDir, file))
        .sort();

    const totalImages = imageFiles.length;

    if (totalImages === 0) {
        console.log(`No images found in ${imageDir}`);
        return {};
    }

    console.log(`Processing ${totalImages} images with ${numWorkers} workers...`);

    // Create progress bar
    const progressBar = new ProgressBar('[:bar] :current/:total :percent :etas | Success: :success', {
        complete: '=',
        incomplete: ' ',
        width: 40,
        total: totalImages
    });

    // Process images in parallel with worker pool
    const results = [];
    const startTime = Date.now();

    async function worker(imageFiles, startIdx) {
        for (let i = startIdx; i < imageFiles.length; i += numWorkers) {
            const result = await processSingleImage(client, imageFiles[i], i);
            results.push(result);

            const successCount = results.filter(r => r.success).length;
            progressBar.tick({ success: `${successCount}/${results.length}` });
        }
    }

    // Start workers
    const workers = [];
    for (let i = 0; i < numWorkers; i++) {
        workers.push(worker(imageFiles, i));
    }

    await Promise.all(workers);

    const totalTime = (Date.now() - startTime) / 1000;

    // Sort results by index
    results.sort((a, b) => a.index - b.index);

    // Calculate statistics
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    const totalDetections = successful.reduce((sum, r) => sum + r.detections.length, 0);
    const avgInferenceTime = successful.reduce((sum, r) => sum + r.inferenceTime, 0) / successful.length || 0;

    // Count detections by class
    const classCounts = {};
    successful.forEach(result => {
        result.detections.forEach(det => {
            classCounts[det.label] = (classCounts[det.label] || 0) + 1;
        });
    });

    const stats = {
        totalImages,
        successful: successful.length,
        failed: failed.length,
        totalDetections,
        totalTimeSeconds: totalTime,
        avgInferenceTime,
        imagesPerSecond: totalImages / totalTime,
        classCounts
    };

    const output = {
        statistics: stats,
        results
    };

    // Print summary
    console.log('\n=== Batch Inference Summary ===');
    console.log(`Total images: ${stats.totalImages}`);
    console.log(`Successful: ${stats.successful}`);
    console.log(`Failed: ${stats.failed}`);
    console.log(`Total detections: ${stats.totalDetections}`);
    console.log(`Total time: ${stats.totalTimeSeconds.toFixed(2)}s`);
    console.log(`Average inference time: ${stats.avgInferenceTime.toFixed(3)}s`);
    console.log(`Throughput: ${stats.imagesPerSecond.toFixed(2)} images/sec`);

    if (Object.keys(classCounts).length > 0) {
        console.log('\nDetections by class:');
        Object.entries(classCounts)
            .sort((a, b) => b[1] - a[1])
            .forEach(([label, count]) => {
                console.log(`  ${label}: ${count}`);
            });
    }

    if (failed.length > 0) {
        console.log(`\nFailed images (${failed.length}):`);
        failed.slice(0, 10).forEach(result => {
            console.log(`  ${result.image}: ${result.error || 'Unknown error'}`);
        });
        if (failed.length > 10) {
            console.log(`  ... and ${failed.length - 10} more`);
        }
    }

    // Save results to JSON
    if (outputFile) {
        await fs.writeFile(outputFile, JSON.stringify(output, null, 2));
        console.log(`\nResults saved to ${outputFile}`);
    }

    return output;
}

// CLI usage
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length < 1) {
        console.log('Usage: node batch-inference.js <image_directory> [options]');
        console.log('');
        console.log('Options:');
        console.log('  --host <ip>        Camera IP address (default: 192.168.1.100)');
        console.log('  --username <user>  Camera username (default: root)');
        console.log('  --password <pass>  Camera password (default: pass)');
        console.log('  --output <file>    Output JSON file path');
        console.log('  --workers <n>      Number of parallel workers (default: 3)');
        console.log('');
        console.log('Example:');
        console.log('  node batch-inference.js ./images --output results.json --workers 3');
        process.exit(1);
    }

    const imageDir = args[0];
    const host = args.includes('--host') ? args[args.indexOf('--host') + 1] : '192.168.1.100';
    const username = args.includes('--username') ? args[args.indexOf('--username') + 1] : 'root';
    const password = args.includes('--password') ? args[args.indexOf('--password') + 1] : 'pass';
    const outputFile = args.includes('--output') ? args[args.indexOf('--output') + 1] : null;
    const numWorkers = args.includes('--workers') ? parseInt(args[args.indexOf('--workers') + 1]) : 3;

    (async () => {
        try {
            const client = new InferenceClient(host, username, password);

            await batchInference(client, imageDir, {
                outputFile,
                numWorkers
            });

        } catch (error) {
            console.error('Error:', error.message);
            process.exit(1);
        }
    })();
}

module.exports = { batchInference, processSingleImage };
