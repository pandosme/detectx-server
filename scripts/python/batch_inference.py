#!/usr/bin/env python3
"""
Batch Inference Example

Demonstrates how to process multiple images efficiently with the inference server.
Includes error handling, retry logic, and progress tracking.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from inference_client import InferenceClient


def process_single_image(client: InferenceClient, image_path: str, index: int, max_retries: int = 3) -> Dict:
    """
    Process a single image with retry logic.

    Args:
        client: InferenceClient instance
        image_path: Path to image
        index: Image index
        max_retries: Maximum number of retry attempts

    Returns:
        Dictionary with results or error information
    """
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            detections = client.infer_jpeg(image_path, image_index=index)
            inference_time = time.time() - start_time

            return {
                'index': index,
                'image': os.path.basename(image_path),
                'success': True,
                'detections': detections,
                'inference_time': inference_time,
                'attempts': attempt + 1
            }

        except Exception as e:
            error_msg = str(e)

            # If server is busy, wait and retry
            if 'busy' in error_msg.lower() or '503' in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue

            # Other errors
            return {
                'index': index,
                'image': os.path.basename(image_path),
                'success': False,
                'error': error_msg,
                'attempts': attempt + 1
            }

    return {
        'index': index,
        'image': os.path.basename(image_path),
        'success': False,
        'error': 'Max retries exceeded',
        'attempts': max_retries
    }


def batch_inference(
    client: InferenceClient,
    image_dir: str,
    output_file: str = None,
    num_workers: int = 3,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
) -> Dict:
    """
    Process all images in a directory.

    Args:
        client: InferenceClient instance
        image_dir: Directory containing images
        output_file: Optional JSON output file path
        num_workers: Number of parallel workers (should match server queue size)
        image_extensions: List of image file extensions to process

    Returns:
        Dictionary with aggregated results and statistics
    """
    # Find all images
    image_dir = Path(image_dir)
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    image_files = sorted(image_files)
    total_images = len(image_files)

    if total_images == 0:
        print(f"No images found in {image_dir}")
        return {}

    print(f"Processing {total_images} images with {num_workers} workers...")

    # Process images in parallel
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_image, client, str(img_path), idx): idx
            for idx, img_path in enumerate(image_files)
        }

        # Process results with progress bar
        with tqdm(total=total_images, desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

                # Update progress bar description with success rate
                success_count = sum(1 for r in results if r['success'])
                pbar.set_postfix({'success': f"{success_count}/{len(results)}"})

    total_time = time.time() - start_time

    # Sort results by index
    results.sort(key=lambda x: x['index'])

    # Calculate statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    total_detections = sum(len(r.get('detections', [])) for r in successful)
    avg_inference_time = sum(r['inference_time'] for r in successful) / len(successful) if successful else 0

    # Count detections by class
    class_counts = {}
    for result in successful:
        for det in result.get('detections', []):
            label = det['label']
            class_counts[label] = class_counts.get(label, 0) + 1

    stats = {
        'total_images': total_images,
        'successful': len(successful),
        'failed': len(failed),
        'total_detections': total_detections,
        'total_time_seconds': total_time,
        'avg_inference_time': avg_inference_time,
        'images_per_second': total_images / total_time,
        'class_counts': class_counts
    }

    output = {
        'statistics': stats,
        'results': results
    }

    # Print summary
    print("\n=== Batch Inference Summary ===")
    print(f"Total images: {stats['total_images']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Total time: {stats['total_time_seconds']:.2f}s")
    print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
    print(f"Throughput: {stats['images_per_second']:.2f} images/sec")

    if class_counts:
        print("\nDetections by class:")
        for label, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}")

    if failed:
        print(f"\nFailed images ({len(failed)}):")
        for result in failed[:10]:  # Show first 10
            print(f"  {result['image']}: {result.get('error', 'Unknown error')}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    # Save results to JSON
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return output


def validate_dataset(
    client: InferenceClient,
    images_dir: str,
    ground_truth_file: str,
    iou_threshold: float = 0.5
) -> Dict:
    """
    Validate model performance against ground truth annotations.

    Args:
        client: InferenceClient instance
        images_dir: Directory with images
        ground_truth_file: Path to ground truth annotations (YOLO format)
        iou_threshold: IOU threshold for matching detections

    Returns:
        Dictionary with validation metrics (precision, recall, mAP)
    """
    # TODO: Implement validation logic
    # This would load ground truth, run inference, compute IOU, calculate metrics
    print("Dataset validation - To be implemented")
    return {}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch image inference")
    parser.add_argument('image_dir', help='Directory containing images')
    parser.add_argument('--host', default='192.168.1.100', help='Camera IP address')
    parser.add_argument('--username', default='root', help='Camera username')
    parser.add_argument('--password', default='pass', help='Camera password')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers')

    args = parser.parse_args()

    # Initialize client
    client = InferenceClient(
        host=args.host,
        username=args.username,
        password=args.password
    )

    # Run batch inference
    results = batch_inference(
        client=client,
        image_dir=args.image_dir,
        output_file=args.output,
        num_workers=args.workers
    )
