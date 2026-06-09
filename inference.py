#!/usr/bin/env python3
"""
YOLO ONNX inference for jewelry detection.

Pipeline:
  1. Letterbox input to 960x960 (preserve aspect ratio, gray padding 114)
  2. Normalize [0,1], convert to NCHW [1,3,960,960]
  3. ONNX Runtime inference → [1, 300, 6]  (x1,y1,x2,y2,conf,cls)
  4. Filter by confidence + NMS
  5. Draw bounding boxes + labels on the 960x960 frame
  6. Save / display

Usage:
    python inference.py --input hand.png --output result.png
    python inference.py --input hand.mp4 --output result.mp4 --conf 0.4
    python inference.py --input hand.png --display
    python inference.py --input 0 --display       # camera 0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
try:
    import onnxruntime
except ImportError:
    print(
        "onnxruntime is required. Install it with:\n"
        "    uv pip install onnxruntime\n"
        "    # or for GPU:\n"
        "    # uv pip install onnxruntime-gpu",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES: dict[int, str] = {0: "jewelry"}
INPUT_SIZE: int = 960  # ONNX model expects [1, 3, 960, 960]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def letterbox(img: np.ndarray) -> np.ndarray:
    """Resize preserving aspect ratio so the largest side equals ``INPUT_SIZE``;
    pad with gray (114) to form a square ``(INPUT_SIZE, INPUT_SIZE)`` image.

    Returns
        (padded, scale, dw, dh) where
            padded: ``(INPUT_SIZE, INPUT_SIZE, 3)`` uint8
            scale:  scale factor applied to original dimensions
            dw, dh: horizontal / vertical padding (pixels, each side)
    """
    h, w = img.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    dw = (INPUT_SIZE - new_w) // 2
    dh = (INPUT_SIZE - new_h) // 2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[dh : dh + new_h, dw : dw + new_w] = resized

    return canvas, scale, dw, dh


def preprocess(img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Convert BGR → RGB, letterbox to ``INPUT_SIZE``, normalise to [0, 1],
    format as NCHW batch.

    Returns
        tensor: ``(1, 3, INPUT_SIZE, INPUT_SIZE)`` float32
        scale, dw, dh: letterbox parameters (for optional postprocessing)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    square_img, scale, dw, dh = letterbox(img_rgb)
    tensor = (
        np.ascontiguousarray(
            np.transpose(square_img.astype(np.float32) / 255.0, (2, 0, 1))
        )[np.newaxis, ...]
    )
    return tensor, scale, dw, dh


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Non-Maximum Suppression via OpenCV.

    Returns
        Array of kept indices (may be empty).
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.intp)

    result = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), 0.0, iou_threshold
    )

    indices = result[0] if isinstance(result, tuple) else result
    if len(indices) == 0:
        return np.array([], dtype=np.intp)

    return indices.flatten().astype(np.intp)


def postprocess(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw ONNX output to (boxes, scores).

    ``output`` shape ``(1, 300, 6)`` — each row:
        ``[x1, y1, x2, y2, confidence, class_id]``  (pixel coords, 960x960)

    Returns
        boxes:  ``(N, 4)`` in ``[x1, y1, x2, y2]`` format (960x960 space)
        scores: ``(N,)``
    """
    dets = np.squeeze(output, axis=0)  # (300, 6)

    mask = dets[:, 4] >= conf_threshold
    dets = dets[mask]
    if dets.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    boxes = dets[:, :4]
    scores = dets[:, 4]

    # Clip to [0, INPUT_SIZE] — model outputs can slightly exceed bounds
    np.clip(boxes[:, [0, 2]], 0, INPUT_SIZE, out=boxes[:, [0, 2]])
    np.clip(boxes[:, [1, 3]], 0, INPUT_SIZE, out=boxes[:, [1, 3]])

    keep = nms(boxes, scores, iou_threshold)
    return boxes[keep], scores[keep]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def draw_boxes(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> None:
    """Draw bounding boxes with labels on ``img`` (in-place)."""
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"jewelry {score:.2f}"
        label_y = max(y1 - 6, 18)
        cv2.putText(img, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# Single-image & video helpers
# ---------------------------------------------------------------------------


def infer_one(
    session: onnxruntime.InferenceSession,
    img: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run full inference pipeline on one BGR image.

    Returns
        (boxes, scores, square_img) where ``square_img`` is the 960x960
        letterboxed frame (BGR) ready for drawing.
    """
    rgb_tensor, _scale, _dw, _dh = preprocess(img)

    # Also produce the 960x960 BGR letterbox for drawing
    square_bgr, _scale, _dw, _dh = letterbox(img)

    outputs = session.run(None, {session.get_inputs()[0].name: rgb_tensor})
    boxes, scores = postprocess(outputs[0], conf_threshold, iou_threshold)
    return boxes, scores, square_bgr


def process_image(
    session: onnxruntime.InferenceSession,
    in_path: str,
    out_path: str | None,
    conf_threshold: float,
    iou_threshold: float,
    display: bool,
) -> None:
    """Run inference on a single image file."""
    img = cv2.imread(in_path)
    if img is None:
        print(f"Error: cannot read image '{in_path}'", file=sys.stderr)
        sys.exit(1)

    boxes, scores, square = infer_one(session, img, conf_threshold, iou_threshold)
    draw_boxes(square, boxes, scores)

    print(f"Detections: {len(boxes)}")
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        print(f"  jewelry {score:.2f}  ({x1}, {y1}) -> ({x2}, {y2})")

    if out_path:
        cv2.imwrite(out_path, square)
        print(f"Saved: {out_path}")

    if display:
        cv2.imshow(f"Detection {Path(in_path).name}", square)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    session: onnxruntime.InferenceSession,
    in_path: str,
    out_path: str | None,
    conf_threshold: float,
    iou_threshold: float,
    display: bool,
) -> None:
    """Run inference on a video file."""
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{in_path}'", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames, output: {INPUT_SIZE}x{INPUT_SIZE}")

    writer: cv2.VideoWriter | None = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (INPUT_SIZE, INPUT_SIZE))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, scores, square = infer_one(session, frame, conf_threshold, iou_threshold)
        draw_boxes(square, boxes, scores)

        if writer:
            writer.write(square)
        if display:
            cv2.imshow("Detection", square)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 30 == 0 and total > 0:
            pct = 100.0 * frame_idx / total
            print(f"\r  Processed {frame_idx}/{total} ({pct:.0f}%)", end="")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {frame_idx}")
    if out_path:
        print(f"Saved: {out_path}")


def process_camera(
    session: onnxruntime.InferenceSession,
    camera_id: int,
    out_path: str | None,
    conf_threshold: float,
    iou_threshold: float,
    display: bool,
) -> None:
    """Run inference from a webcam."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_id}", file=sys.stderr)
        sys.exit(1)

    print(f"Camera {camera_id} opened, output: {INPUT_SIZE}x{INPUT_SIZE}")
    print("Press 'q' in the window to quit.")

    writer: cv2.VideoWriter | None = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30, (INPUT_SIZE, INPUT_SIZE))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        boxes, scores, square = infer_one(session, frame, conf_threshold, iou_threshold)
        draw_boxes(square, boxes, scores)

        if writer:
            writer.write(square)
        if display:
            cv2.imshow(f"Camera {camera_id}", square)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 15 == 0:
            print(f"\r  Frames processed: {frame_idx}", end="")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nTotal frames processed: {frame_idx}")
    if out_path:
        print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO ONNX inference for jewelry detection"
    )
    parser.add_argument(
        "--model",
        default="best.onnx",
        help="Path to ONNX model (default: best.onnx)",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to image/video file, or camera ID (0, 1, ...)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the result",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show result in a window",
    )
    args = parser.parse_args()

    if not args.output and not args.display:
        parser.error("Either --output or --display is required")

    if not os.path.isfile(args.model):
        print(f"Error: model file not found: '{args.model}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    session = onnxruntime.InferenceSession(
        args.model, providers=["CPUExecutionProvider"]
    )
    print(f"  Input  : {session.get_inputs()[0].shape}")
    print(f"  Output : {session.get_outputs()[0].shape}")

    # Camera mode if --input is a number
    if args.input.isdigit():
        process_camera(
            session, int(args.input), args.output,
            args.conf, args.iou, args.display
        )
        return

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: '{args.input}'", file=sys.stderr)
        sys.exit(1)

    ext = Path(args.input).suffix.lower()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if ext in image_exts:
        process_image(
            session, args.input, args.output, args.conf, args.iou, args.display
        )
    elif ext in video_exts:
        process_video(
            session, args.input, args.output, args.conf, args.iou, args.display
        )
    else:
        print(
            f"Unsupported file extension: '{ext}'.\n"
            f"  Images: {', '.join(sorted(image_exts))}\n"
            f"  Videos: {', '.join(sorted(video_exts))}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
