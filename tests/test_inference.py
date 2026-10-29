"""Unit and smoke tests for vision-inference-engine."""
import numpy as np
import pytest
from ultralytics import YOLO
import main


def test_yolo_model_initialization():
    """Verify that YOLOv8 weights load correctly."""
    assert main.yolo_model is not None
    assert isinstance(main.yolo_model, YOLO)


def test_detect_gesture_blank_frame():
    """Verify detect_gesture processes a dummy video frame without errors."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    processed = main.detect_gesture(dummy_frame)
    assert processed is not None
    assert processed.shape == (480, 640, 3)


def test_yolo_inference_blank_frame():
    """Verify YOLOv8 performs inference on synthetic test frame."""
    dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
    results = main.yolo_model(dummy_frame)
    assert len(results) > 0
    assert hasattr(results[0], "boxes")
