#!/usr/bin/env python3
"""
Axis Inference Server Client (Python)

Supports all image formats: JPEG, PNG, BMP, etc.
Non-JPEG formats are automatically converted when using JPEG inference mode.

Usage examples:

  # JPEG + tensor inference with auth (works with any image format)
  python3 inference_client.py \
      --host 192.168.1.100 \
      --username root \
      --password pass \
      --mode both \
      --index 0 \
      /path/to/image.jpg

  # PNG file with tensor inference
  python3 inference_client.py \
      --host 192.168.1.100 \
      --username root \
      --password pass \
      --mode tensor \
      /path/to/image.png

Arguments:
  positional:
    image                 Path to the input image (JPEG, PNG, BMP, etc.)

  options:
    -H, --host            Camera IP or hostname (required)
    -u, --username        Username for camera auth (optional)
    -p, --password        Password for camera auth (optional)
    -m, --mode            Inference mode: jpeg, tensor, or both (default: both)
    -i, --index           Image index metadata sent to server (default: 0)
    -c, --confidence      Minimum confidence threshold 0.0-1.0 (default: 0.0)

This script talks to an Axis camera inference server at /local/detectx
and runs JPEG and/or tensor inference on the provided image using the model
configured on the camera.
"""

import argparse
import io
import json
import sys
import time
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from PIL import Image


class InferenceClient:
    """Client for Axis Camera Inference Server"""

    def __init__(self, host: str, username: str = None, password: str = None):
        """
        Initialize the inference client.

        Args:
            host: Camera IP or hostname (e.g., "192.168.1.100")
            username: Optional digest auth username
            password: Optional digest auth password
        """
        self.base_url = f"http://{host}/local/detectx"
        self.auth = HTTPDigestAuth(username, password) if username and password else None
        self.session = requests.Session()

    def get_capabilities(self) -> Dict:
        """
        Get server capabilities and model information.

        Returns:
            Dictionary with model info, input formats, and classes
        """
        url = f"{self.base_url}/capabilities"
        response = self.session.get(url, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def get_health(self) -> Dict:
        """
        Get server health and statistics.

        Returns:
            Dictionary with server status and statistics
        """
        url = f"{self.base_url}/health"
        response = self.session.get(url, auth=self.auth)
        response.raise_for_status()
        return response.json()

    def infer_jpeg(self, image_path: str, image_index: int = -1) -> List[Dict]:
        """
        Perform inference on a JPEG image.
        Automatically converts non-JPEG formats (PNG, BMP, etc.) to JPEG.

        Args:
            image_path: Path to image file (JPEG, PNG, BMP, etc.)
            image_index: Optional image index for dataset validation

        Returns:
            List of detections, each containing:
                - index: Image index
                - label: Object class name
                - class_id: Numeric class ID
                - confidence: Detection confidence (0.0-1.0)
                - bbox_pixels: Bounding box in pixels {x, y, w, h}
                - bbox_yolo: Normalized bounding box (center format)

        Raises:
            requests.HTTPError: If inference fails
        """
        url = f"{self.base_url}/inference-jpeg"
        if image_index >= 0:
            url += f"?index={image_index}"

        # Check if image needs conversion to JPEG
        img = Image.open(image_path)
        if img.format != 'JPEG':
            # Convert to JPEG in memory
            buffer = io.BytesIO()
            # Convert to RGB if needed (PNG might have alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(buffer, format='JPEG', quality=95)
            image_data = buffer.getvalue()
        else:
            # Already JPEG, read directly
            with open(image_path, 'rb') as f:
                image_data = f.read()

        headers = {'Content-Type': 'image/jpeg'}

        response = self.session.post(
            url, data=image_data, headers=headers, auth=self.auth
        )

        # Handle different status codes
        if response.status_code == 200:
            return response.json()['detections']
        elif response.status_code == 204:
            return []  # No detections
        elif response.status_code == 503:
            raise Exception("Server busy - queue full")
        else:
            response.raise_for_status()

    def infer_tensor(
        self, rgb_array: np.ndarray, image_index: int = -1
    ) -> List[Dict]:
        """
        Perform inference on a preprocessed RGB tensor.

        Args:
            rgb_array: NumPy array with shape (height, width, 3) and dtype uint8
                      Must match model input dimensions (typically 640x640x3)
            image_index: Optional image index for dataset validation

        Returns:
            List of detections (same format as infer_jpeg)

        Raises:
            ValueError: If array dimensions don't match model requirements
            requests.HTTPError: If inference fails
        """
        url = f"{self.base_url}/inference-tensor"
        if image_index >= 0:
            url += f"?index={image_index}"

        # Validate array shape
        if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
            raise ValueError(f"Expected shape (H, W, 3), got {rgb_array.shape}")

        if rgb_array.dtype != np.uint8:
            raise ValueError(f"Expected dtype uint8, got {rgb_array.dtype}")

        # Convert to bytes (RGB interleaved)
        tensor_bytes = rgb_array.tobytes()

        headers = {'Content-Type': 'application/octet-stream'}

        response = self.session.post(
            url, data=tensor_bytes, headers=headers, auth=self.auth
        )

        if response.status_code == 200:
            return response.json()['detections']
        elif response.status_code == 204:
            return []
        elif response.status_code == 503:
            raise Exception("Server busy - queue full")
        else:
            response.raise_for_status()

    def preprocess_image_to_tensor(
        self, image_path: str, target_size: Tuple[int, int] = (640, 640)
    ) -> np.ndarray:
        """
        Preprocess an image to tensor format with letterboxing.

        This applies the same preprocessing the server does for JPEG inputs,
        allowing you to use the faster tensor endpoint.

        Args:
            image_path: Path to image file
            target_size: Target (width, height), default (640, 640)

        Returns:
            NumPy array with shape (height, width, 3) ready for tensor inference
        """
        img = Image.open(image_path).convert('RGB')

        # Calculate scale to maintain aspect ratio
        scale = min(target_size[0] / img.width, target_size[1] / img.height)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        # Resize image
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create black background
        result = Image.new('RGB', target_size, (0, 0, 0))

        # Paste resized image centered
        offset_x = (target_size[0] - new_w) // 2
        offset_y = (target_size[1] - new_h) // 2
        result.paste(img_resized, (offset_x, offset_y))

        # Convert to numpy array
        return np.array(result, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Axis Inference Server client (JPEG and tensor inference)[web:16][web:18]"
    )

    parser.add_argument(
        "image",
        help="Path to input image (JPEG or any format Pillow can open)"
    )

    parser.add_argument(
        "--host", "-H",
        required=True,
        help="Camera IP or hostname, e.g. 192.168.1.100"
    )

    parser.add_argument(
        "--username", "-u",
        default=None,
        help="Username for digest/basic auth (optional)"
    )

    parser.add_argument(
        "--password", "-p",
        default=None,
        help="Password for auth (optional)"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["jpeg", "tensor", "both"],
        default="both",
        help="Inference mode: jpeg, tensor, or both (default: both)"
    )

    parser.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="Image index metadata to send to server (default: 0)"
    )

    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.0,
        help="Minimum confidence threshold (0.0-1.0, default: 0.0 shows all)"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize client
    client = InferenceClient(
        host=args.host,
        username=args.username,
        password=args.password,
    )

    # Get server capabilities
    print("=== Server Capabilities ===")
    capabilities = client.get_capabilities()
    print(f"Model: {capabilities['model']['input_width']}x{capabilities['model']['input_height']}")
    print(f"Classes: {len(capabilities['model']['classes'])}")
    print(f"Formats: {[f['endpoint'] for f in capabilities['model']['input_formats']]}")
    print()

    # Check server health
    print("=== Server Health ===")
    health = client.get_health()
    print(f"Running: {health['running']}")
    print(f"Queue size: {health['queue_size']}")
    print(f"Total requests: {health['statistics']['total_requests']}")
    print()

    image_path = args.image

    if args.mode in ("jpeg", "both"):
        print("=== JPEG Inference ===")
        start_time = time.time()
        detections = client.infer_jpeg(image_path, image_index=args.index)
        inference_time_ms = (time.time() - start_time) * 1000

        # Filter by confidence threshold
        if args.confidence > 0.0:
            detections = [d for d in detections if d['confidence'] >= args.confidence]

        print(f"Inference time: {inference_time_ms:.1f} ms")
        print(f"Found {len(detections)} objects (confidence >= {args.confidence:.0%}):")

        # Show individual detections
        for det in detections:
            print(
                f"  - {det['label']}: {det['confidence']:.2%} at "
                f"({det['bbox_pixels']['x']}, {det['bbox_pixels']['y']}) "
                f"{det['bbox_pixels']['w']}x{det['bbox_pixels']['h']}"
            )

        # Show label summary
        if detections:
            label_counts = Counter(det['label'] for det in detections)
            print("\nLabel Summary:")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}")
        print()

    if args.mode in ("tensor", "both"):
        print("=== Tensor Inference ===")
        preprocess_start = time.time()
        tensor = client.preprocess_image_to_tensor(image_path)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000

        print(f"Preprocessed tensor shape: {tensor.shape}")
        print(f"Preprocessing time: {preprocess_time_ms:.1f} ms")

        inference_start = time.time()
        detections = client.infer_tensor(tensor, image_index=args.index)
        inference_time_ms = (time.time() - inference_start) * 1000

        # Filter by confidence threshold
        if args.confidence > 0.0:
            detections = [d for d in detections if d['confidence'] >= args.confidence]

        print(f"Inference time: {inference_time_ms:.1f} ms")
        print(f"Total time: {preprocess_time_ms + inference_time_ms:.1f} ms")
        print(f"Found {len(detections)} objects (confidence >= {args.confidence:.0%}):")

        # Show individual detections
        for det in detections:
            print(
                f"  - {det['label']}: {det['confidence']:.2%} at "
                f"({det['bbox_pixels']['x']}, {det['bbox_pixels']['y']}) "
                f"{det['bbox_pixels']['w']}x{det['bbox_pixels']['h']}"
            )

        # Show label summary
        if detections:
            label_counts = Counter(det['label'] for det in detections)
            print("\nLabel Summary:")
            for label, count in label_counts.most_common():
                print(f"  {label}: {count}")
        print()


if __name__ == "__main__":
    main()
