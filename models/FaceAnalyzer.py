import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.transform import Rotation
import logging
import os
import urllib.request

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── DNN model auto-download ───────────────────────────────────────────────────
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTOTXT_PATH = os.path.join(_MODEL_DIR, "deploy.prototxt")
_CAFFEMODEL_PATH = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


def _ensure_dnn_models():
    """Download the Caffe DNN face-detection model files if they don't exist."""
    for path, url, label in [
        (_PROTOTXT_PATH, _PROTOTXT_URL, "deploy.prototxt"),
        (_CAFFEMODEL_PATH, _CAFFEMODEL_URL, "res10 caffemodel"),
    ]:
        if not os.path.exists(path):
            logger.info(f"Downloading {label} → {path} …")
            try:
                urllib.request.urlretrieve(url, path)
                logger.info(f"  ✔ Downloaded {label} ({os.path.getsize(path)} bytes)")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {label} from {url}: {e}"
                ) from e


class FaceAnalyzer:
    def __init__(self, image_path):
        """Initialize the FaceAnalyzer with an image path."""
        self.image_path = image_path
        self.original_image = self.load_image()
        self.image_height, self.image_width = self.original_image.shape[:2]
        self.face_data = []
        self.zones = self.define_zones()

        # Ensure DNN model files are available
        _ensure_dnn_models()

        # OpenCV DNN face detector (SSD ResNet-10, much more accurate than Haar)
        self.face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT_PATH, _CAFFEMODEL_PATH)

        # Eye cascade for eye-state detection
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

    def load_image(self):
        """Load and validate the input image."""
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        image = self.enhance_image(image)
        return image

    def enhance_image(self, image):
        """Enhance image brightness and contrast using CLAHE for better face detection."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        logger.info("Image enhanced with CLAHE for better face detection")
        return enhanced_image

    def define_zones(self):
        """Define zones based on the image width."""
        third_width = self.image_width // 3
        return {
            "left": (0, third_width),
            "center": (third_width, 2 * third_width),
            "right": (2 * third_width, self.image_width)
        }

    # ── DNN-based face detection ────────────────────────────────────────────────
    def detect_faces_dnn(self, confidence_threshold=0.5):
        """Detect faces using OpenCV DNN (SSD ResNet-10).

        Returns a dict keyed by face_id with score, facial_area, and landmarks.
        """
        h, w = self.original_image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(self.original_image, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        result = {}
        face_idx = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < confidence_threshold:
                continue

            # Scale bounding box back to image dimensions
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_idx += 1
            face_id = f"face_{face_idx}"

            # Detect eyes within face ROI for landmark approximation
            face_roi_gray = gray[y1:y2, x1:x2]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi_gray, 1.1, 3, minSize=(15, 15)
            )

            landmarks = self._approximate_landmarks(
                x1, y1, x2 - x1, y2 - y1, eyes
            )

            result[face_id] = {
                "score": round(confidence, 4),
                "facial_area": [x1, y1, x2, y2],
                "landmarks": landmarks,
            }

        return result

    def _approximate_landmarks(self, x, y, w, h, eyes):
        """Approximate facial landmarks from face bounding box and detected eyes."""
        landmarks = {
            "left_eye": [x + int(w * 0.3), y + int(h * 0.35)],
            "right_eye": [x + int(w * 0.7), y + int(h * 0.35)],
            "nose": [x + int(w * 0.5), y + int(h * 0.55)],
            "mouth_left": [x + int(w * 0.35), y + int(h * 0.75)],
            "mouth_right": [x + int(w * 0.65), y + int(h * 0.75)]
        }

        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda e: e[0])
            ex1, ey1, ew1, eh1 = sorted_eyes[0]
            ex2, ey2, ew2, eh2 = sorted_eyes[1]
            landmarks["left_eye"] = [x + ex1 + ew1 // 2, y + ey1 + eh1 // 2]
            landmarks["right_eye"] = [x + ex2 + ew2 // 2, y + ey2 + eh2 // 2]

        return landmarks

    def calculate_head_pose(self, landmarks):
        """Calculate head pose angles using PnP."""
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0)
            ])

            image_points = [
                landmarks["nose"],
                landmarks.get("chin", landmarks["nose"]),
                landmarks["left_eye"],
                landmarks["right_eye"],
                landmarks.get("mouth_left", landmarks["left_eye"]),
                landmarks.get("mouth_right", landmarks["right_eye"])
            ]
            image_points = np.array(image_points, dtype="double")

            focal_length = self.image_width
            center = (self.image_width / 2, self.image_height / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                raise ValueError("PnP solution failed.")

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz", degrees=True)
            pitch, yaw, roll = euler_angles

            return {
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "roll": round(roll, 2),
                "confidence": 1.0
            }
        except Exception as e:
            logger.warning(f"PnP head pose calculation failed: {str(e)}")
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0, "confidence": 0.0}

    # ── Eye-state detection (replaces emotion detection) ────────────────────────
    def detect_eye_state(self, face_image, face_x, face_w):
        """Detect eye state (open / partially_closed / closed) and gaze direction.

        Args:
            face_image: BGR image cropped to the face region
            face_x: x-coordinate of face bounding box left edge (in full image)
            face_w: width of face bounding box

        Returns:
            dict with keys: eyes_detected, eye_state, gaze, confidence
        """
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            fh, fw = gray.shape[:2]

            # Only search the upper 60% of the face for eyes
            upper_face = gray[0:int(fh * 0.6), :]

            eyes = self.eye_cascade.detectMultiScale(
                upper_face,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(max(12, fw // 10), max(12, fh // 15)),
                maxSize=(fw // 2, int(fh * 0.35)),
            )

            eyes_detected = min(len(eyes), 2)  # cap at 2

            if eyes_detected == 0:
                return {
                    "eyes_detected": 0,
                    "eye_state": "closed",
                    "gaze": "away",
                    "confidence": 0.40,
                }

            # Sort by x-coordinate to get left / right
            sorted_eyes = sorted(eyes, key=lambda e: e[0])[:2]

            # ── Eye aspect ratio heuristic ──
            # Wider eyes (larger h relative to w) ⇒ more open
            ear_values = []
            eye_centers_x = []
            for (ex, ey, ew, eh) in sorted_eyes:
                ear = eh / ew if ew > 0 else 0
                ear_values.append(ear)
                eye_centers_x.append(ex + ew / 2)

            avg_ear = np.mean(ear_values)
            if avg_ear > 0.35:
                eye_state = "open"
                state_conf = min(0.95, 0.6 + avg_ear)
            elif avg_ear > 0.22:
                eye_state = "partially_closed"
                state_conf = 0.60
            else:
                eye_state = "closed"
                state_conf = 0.50

            # ── Gaze direction from eye centroid offset ──
            avg_eye_x = np.mean(eye_centers_x)
            relative_pos = avg_eye_x / fw  # 0.0 = far left, 1.0 = far right

            if 0.30 <= relative_pos <= 0.70:
                gaze = "center"
            elif relative_pos < 0.30:
                gaze = "left"
            else:
                gaze = "right"

            confidence = round(state_conf, 2)

            return {
                "eyes_detected": eyes_detected,
                "eye_state": eye_state,
                "gaze": gaze,
                "confidence": confidence,
            }

        except Exception as e:
            logger.warning(f"Eye-state detection failed: {e}")
            return {
                "eyes_detected": 0,
                "eye_state": "closed",
                "gaze": "away",
                "confidence": 0.30,
            }

    def determine_zone(self, center_x):
        """Determine which zone the student is located in based on x-coordinate."""
        for zone, (start, end) in self.zones.items():
            if start <= center_x < end:
                return zone
        return "unknown"

    # ── Main analysis pipeline ──────────────────────────────────────────────────
    def analyze_faces(self):
        """Main method to detect and analyze faces in the image."""
        try:
            # Use DNN detector for better accuracy and real confidence scores
            faces = self.detect_faces_dnn(confidence_threshold=0.5)

            if not faces:
                logger.warning(
                    "No faces detected in the image. The image may be too dark, "
                    "blurry, or faces may not be clearly visible."
                )
                return 0

            logger.info(f"Detected {len(faces)} face(s) in the image")

            for face_id, face in faces.items():
                x1, y1, x2, y2 = face["facial_area"]
                face_image = self.original_image[y1:y2, x1:x2]

                pose = self.calculate_head_pose(face["landmarks"])
                eye_data = self.detect_eye_state(face_image, x1, x2 - x1)
                zone = self.determine_zone((x1 + x2) // 2)

                face_data = {
                    "face_id": face_id,
                    "detection_confidence": face["score"],
                    "position": {
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2
                    },
                    "zone": zone,
                    "pose": pose,
                    "eyes_detected": eye_data["eyes_detected"],
                    "eye_state": eye_data["eye_state"],
                    "gaze": eye_data["gaze"],
                    "confidence": eye_data["confidence"],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.face_data.append(face_data)
                self.draw_face_analysis(face_data)

            return len(self.face_data)
        except Exception as e:
            logger.error(f"Face analysis failed: {str(e)}")
            return 0

    # ── Drawing / annotation ────────────────────────────────────────────────────
    def draw_face_analysis(self, face_data):
        """Draw bounding box, confidence, eye-state, and head-pose on each face."""
        pos = face_data["position"]
        x1, y1, x2, y2 = pos["x1"], pos["y1"], pos["x2"], pos["y2"]
        pose = face_data["pose"]
        pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]
        det_conf = face_data["detection_confidence"]

        # Colour based on head-pose deviation
        max_dev = max(abs(yaw), abs(pitch))
        if max_dev < 15:
            box_color = (0, 200, 0)       # Green — attentive
        elif max_dev < 35:
            box_color = (0, 200, 255)     # Yellow/Orange — moderate
        else:
            box_color = (0, 0, 230)       # Red — looking away

        # Bounding box
        cv2.rectangle(self.original_image, (x1, y1), (x2, y2), box_color, 2)

        # Corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thick = 3
        cv2.line(self.original_image, (x1, y1), (x1 + corner_len, y1), box_color, thick)
        cv2.line(self.original_image, (x1, y1), (x1, y1 + corner_len), box_color, thick)
        cv2.line(self.original_image, (x2, y1), (x2 - corner_len, y1), box_color, thick)
        cv2.line(self.original_image, (x2, y1), (x2, y1 + corner_len), box_color, thick)
        cv2.line(self.original_image, (x1, y2), (x1 + corner_len, y2), box_color, thick)
        cv2.line(self.original_image, (x1, y2), (x1, y2 - corner_len), box_color, thick)
        cv2.line(self.original_image, (x2, y2), (x2 - corner_len, y2), box_color, thick)
        cv2.line(self.original_image, (x2, y2), (x2, y2 - corner_len), box_color, thick)

        # Label lines — now shows detection confidence, eye state, gaze
        label_lines = [
            f"Conf: {det_conf:.0%}  P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
            f"Eyes: {face_data['eye_state']} ({face_data['confidence']:.0%})",
            f"Gaze: {face_data['gaze']}  Zone: {face_data['zone']}",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        font_thickness = 1
        line_height = 18
        padding = 4

        total_label_height = len(label_lines) * line_height + padding * 2
        label_top = max(0, y1 - total_label_height)

        overlay = self.original_image.copy()
        cv2.rectangle(overlay, (x1, label_top), (x2, y1), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, self.original_image, 0.3, 0, self.original_image)

        for i, text in enumerate(label_lines):
            ty = label_top + padding + (i + 1) * line_height - 4
            cv2.putText(
                self.original_image, text, (x1 + padding, ty),
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA
            )

    # ── Save results ────────────────────────────────────────────────────────────
    def save_results(self, output_image_path, output_csv_path):
        """Save the annotated image and CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_image_path = output_image_path or f"face_analysis_{timestamp}.jpg"
        output_csv_path = output_csv_path or f"face_data_{timestamp}.csv"

        cv2.imwrite(output_image_path, self.original_image)
        logger.info(f"Annotated image saved to {output_image_path}")

        if self.face_data:
            df = pd.json_normalize(self.face_data)
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Face data saved to {output_csv_path}")
        else:
            logger.warning("No face data available — saving placeholder CSV.")
            placeholder_columns = [
                "face_id", "detection_confidence", "zone",
                "pose.pitch", "pose.yaw", "pose.roll",
                "eyes_detected", "eye_state", "gaze", "confidence",
            ]
            placeholder_data = [{
                "face_id": 0,
                "detection_confidence": "None",
                "zone": "unknown",
                "pose.pitch": "None",
                "pose.yaw": "None",
                "pose.roll": "None",
                "eyes_detected": 0,
                "eye_state": "None",
                "gaze": "None",
                "confidence": "None",
            }]
            pd.DataFrame(placeholder_data, columns=placeholder_columns).to_csv(
                output_csv_path, index=False
            )
            logger.info(f"Placeholder face data saved to {output_csv_path}")