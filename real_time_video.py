"""
Real-time Emotion Recognition System

This module provides real-time emotion detection using webcam input.
It detects faces, classifies emotions, and displays results in real-time.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import time
from imutils.video import VideoStream

# Configuration parameters
DETECTION_MODEL_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'
EMOTION_MODEL_PATH = 'models/_mini_XCEPTION.102-0.66.hdf5'
FRAME_WIDTH = 640
CANVAS_HEIGHT = 320
CANVAS_WIDTH = 520
BAR_HEIGHT = 28
BAR_SPACING = 36
LEFT_MARGIN = 20
TOP_MARGIN = 20
DETECTION_TIME_SECONDS = 60  # set to 120 for 2 minutes

# Emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Color theme (B, G, R)
EMOTION_COLORS = {
    "angry": (60, 60, 220),       # Red-ish
    "disgust": (40, 170, 80),     # Green
    "scared": (200, 120, 10),     # Orange
    "happy": (0, 200, 255),       # Yellow
    "sad": (200, 60, 60),         # Blue-ish
    "surprised": (255, 160, 0),   # Cyan-ish
    "neutral": (160, 160, 160)    # Gray
}

def open_camera():
    """Try to open camera using multiple backends and indices (macOS-friendly)."""
    attempts = []
    # Preferred backends on macOS first
    attempts += [(i, cv2.CAP_AVFOUNDATION) for i in range(5)]
    attempts += [(i, cv2.CAP_QT) for i in range(5)]
    # Fallback to default backend
    attempts += [(i, cv2.CAP_ANY) for i in range(5)]

    for index, backend in attempts:
        cam = cv2.VideoCapture(index, backend)
        if not cam.isOpened():
            cam.release()
            continue
        # Set resolution and warm up
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.3)
        ok, _ = cam.read()
        if ok:
            return cam
        cam.release()
    return None

def open_videostream():
    """Fallback: use imutils VideoStream if cv2.VideoCapture fails."""
    try:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        frame = vs.read()
        if frame is not None:
            return vs
        vs.stop()
    except Exception:
        pass
    return None

def initialize_models():
    """Initialize face detection and emotion classification models."""
    face_detection = cv2.CascadeClassifier(DETECTION_MODEL_PATH)
    emotion_classifier = load_model(EMOTION_MODEL_PATH, compile=False)
    return face_detection, emotion_classifier

def detect_faces(gray_image, face_detection):
    """Detect faces in the grayscale image."""
    return face_detection.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

def process_face_roi(gray_image, face_coords):
    """Process the face region of interest for emotion classification."""
    fX, fY, fW, fH = face_coords
    roi = gray_image[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    return roi

def draw_emotion_bars(canvas, predictions):
    """Draw emotion probability bars on a dark-themed canvas with color-coded bars."""
    # Dark background
    canvas[:] = (20, 20, 20)

    # Panel title
    cv2.putText(canvas, "Emotion Probabilities", (LEFT_MARGIN, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    usable_width = CANVAS_WIDTH - LEFT_MARGIN * 2

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, predictions)):
        y_top = TOP_MARGIN + i * BAR_SPACING
        bar_width = int(max(1, prob * usable_width))

        # Background bar track
        cv2.rectangle(
            canvas,
            (LEFT_MARGIN, y_top),
            (LEFT_MARGIN + usable_width, y_top + BAR_HEIGHT),
            (50, 50, 50),
            -1,
        )

        # Filled bar
        color = EMOTION_COLORS.get(emotion, (100, 100, 100))
        cv2.rectangle(
            canvas,
            (LEFT_MARGIN, y_top),
            (LEFT_MARGIN + bar_width, y_top + BAR_HEIGHT),
            color,
            -1,
        )

        # Border
        cv2.rectangle(
            canvas,
            (LEFT_MARGIN, y_top),
            (LEFT_MARGIN + usable_width, y_top + BAR_HEIGHT),
            (90, 90, 90),
            1,
        )

        # Text label
        label = f"{emotion}: {prob * 100:.2f}%"
        cv2.putText(
            canvas,
            label,
            (LEFT_MARGIN + 8, y_top + int(BAR_HEIGHT * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

def draw_face_detection(frame, face_coords, emotion_label):
    """Draw face detection box and emotion label on frame."""
    fX, fY, fW, fH = face_coords

    # Label background for readability
    label_text = emotion_label
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (fX, max(0, fY - th - 10)), (fX + tw + 10, fY), (0, 0, 0), -1)

    # Draw emotion label above face
    cv2.putText(frame, label_text, (fX + 5, fY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw face bounding box (green)
    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 200, 0), 2)

# --- Summary window ---
def show_session_summary(emotion_counts: dict, detections_made: int, duration_s: int) -> None:
    """Display a visual summary window with counts and percentages.

    Press 'q', 'ESC', or 'Enter' to close.
    """
    width, height = 760, 520
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # Title and header
    cv2.putText(canvas, "Session Summary", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Duration: {duration_s}s", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frames with detected face: {detections_made}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Bars
    left = 30
    top = 150
    bar_h = 26
    gap = 34
    max_w = width - left - 60

    if detections_made <= 0:
        cv2.putText(canvas, "No face detections.", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    else:
        for idx, emotion in enumerate(EMOTIONS):
            count = int(emotion_counts.get(emotion, 0))
            pct = (count / detections_made) * 100.0
            y = top + idx * gap

            # Track background
            cv2.rectangle(canvas, (left, y), (left + max_w, y + bar_h), (55, 55, 55), -1)

            # Filled bar by percentage
            fill_w = int(max_w * (pct / 100.0))
            color = EMOTION_COLORS.get(emotion, (160, 160, 160))
            cv2.rectangle(canvas, (left, y), (left + fill_w, y + bar_h), color, -1)
            cv2.rectangle(canvas, (left, y), (left + max_w, y + bar_h), (90, 90, 90), 1)

            label = f"{emotion}: {count} ({pct:.2f}%)"
            cv2.putText(canvas, label, (left + 8, y + int(bar_h*0.75)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)

    # Footer instructions
    cv2.putText(canvas, "Press q / ESC / Enter to close", (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.namedWindow('Session Summary', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Session Summary', width, height)
    cv2.moveWindow('Session Summary', 120, 120)
    while True:
        cv2.imshow('Session Summary', canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord('q'), 27, 13):  # q, ESC, Enter
            break
    cv2.destroyWindow('Session Summary')

def main():
    """Main function for real-time emotion recognition."""
    # Initialize models
    face_detection, emotion_classifier = initialize_models()

    # Initialize video capture (robust)
    camera = open_camera()
    use_videostream = False
    vs = None
    if camera is None:
        vs = open_videostream()
        if vs is None:
            print("Could not open camera. On macOS, grant camera access in System Settings > Privacy & Security > Camera to your terminal/app. Close apps using the camera and try again.")
            return
        use_videostream = True

    # Prepare windows (resizable, larger)
    cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Recognition', 900, 680)
    cv2.moveWindow('Emotion Recognition', 80, 80)

    cv2.namedWindow('Emotion Probabilities', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Probabilities', CANVAS_WIDTH, CANVAS_HEIGHT)
    cv2.moveWindow('Emotion Probabilities', 1000, 80)

    print("Starting emotion recognition system...")
    print(f"Session length: {DETECTION_TIME_SECONDS} seconds (press 'q' to quit early)")

    try:
        # Counters for summary
        emotion_counts = {e: 0 for e in EMOTIONS}
        detections_made = 0  # frames where a face was detected and classified
        start_time = time.time()
        while True:
            # Stop when time elapses
            if time.time() - start_time >= DETECTION_TIME_SECONDS:
                break
            # Capture frame
            if use_videostream:
                frame = vs.read()
                if frame is None:
                    print("Failed to capture frame")
                    break
            else:
                ret, frame = camera.read()
                if not ret or frame is None:
                    print("Failed to capture frame")
                    break

            # Resize frame
            frame = imutils.resize(frame, width=FRAME_WIDTH)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Create canvas for emotion bars
            canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype="uint8")
            frame_clone = frame.copy()

            # Detect faces
            faces = detect_faces(gray, face_detection)

            if len(faces) > 0:
                # Get the largest face
                faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

                # Process face ROI
                roi = process_face_roi(gray, faces)

                # Predict emotions
                predictions = emotion_classifier.predict(roi, verbose=0)[0]
                top_idx = int(np.argmax(predictions))
                emotion_label = EMOTIONS[top_idx]

                # Draw emotion bars
                draw_emotion_bars(canvas, predictions)

                # Draw face detection
                draw_face_detection(frame_clone, faces, emotion_label)

                # Update counters
                detections_made += 1
                emotion_counts[emotion_label] += 1
            else:
                # Still render empty bar panel (zeros) for consistent UI
                draw_emotion_bars(canvas, np.zeros(len(EMOTIONS), dtype=np.float32))

            # Display results
            cv2.imshow('Emotion Recognition', frame_clone)
            cv2.imshow('Emotion Probabilities', canvas)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Cleanup
        if use_videostream and vs is not None:
            vs.stop()
        elif camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")

    # Print session summary
    print("\n========== Session Summary ==========")
    print(f"Duration target: {DETECTION_TIME_SECONDS} seconds")
    print(f"Frames with detected face: {detections_made}")
    if detections_made == 0:
        print("No face detections during the session.")
        # Also show empty visual summary window
        show_session_summary(emotion_counts, detections_made, DETECTION_TIME_SECONDS)
        return
    print("\nEmotion counts and percentages (of detected frames):")
    for emotion in EMOTIONS:
        count = emotion_counts[emotion]
        pct = (count / detections_made) * 100.0
        print(f"- {emotion}: {count} ({pct:.2f}%)")
    print("====================================\n")

    # Visual summary window
    show_session_summary(emotion_counts, detections_made, DETECTION_TIME_SECONDS)

if __name__ == "__main__":
    main()
