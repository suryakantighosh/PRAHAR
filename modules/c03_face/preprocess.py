"""
prahar/modules/c03_face/preprocess.py
Image preprocessor:
- Resize and normalise
- EXIF orientation fix
- Blur score (reject < 40)
- EXIF metadata extraction (date, GPS, camera)
"""
import io
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image, ExifTags
from loguru import logger


BLUR_THRESHOLD = 40.0
TARGET_SIZE = (640, 640)


def load_image(source: bytes) -> Optional[np.ndarray]:
    """Load image bytes → BGR numpy array (OpenCV format)."""
    try:
        arr = np.frombuffer(source, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"[C-03] Image load failed: {e}")
        return None


def fix_exif_orientation(img_bytes: bytes) -> bytes:
    """Rotate image to correct orientation per EXIF tag."""
    try:
        pil = Image.open(io.BytesIO(img_bytes))
        exif = pil._getexif()
        if not exif:
            return img_bytes

        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if not orientation_key or orientation_key not in exif:
            return img_bytes

        orientation = exif[orientation_key]
        rotation_map = {3: 180, 6: 270, 8: 90}
        if orientation in rotation_map:
            pil = pil.rotate(rotation_map[orientation], expand=True)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG")
        return buf.getvalue()
    except Exception:
        return img_bytes


def blur_score(img: np.ndarray) -> float:
    """
    Laplacian variance blur score.
    Higher = sharper. Reject images scoring below BLUR_THRESHOLD (40).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_exif(img_bytes: bytes) -> Dict[str, Any]:
    """Extract EXIF metadata: datetime, GPS, camera model."""
    meta: Dict[str, Any] = {}
    try:
        pil = Image.open(io.BytesIO(img_bytes))
        exif_raw = pil._getexif()
        if not exif_raw:
            return meta

        tag_map = {v: k for k, v in ExifTags.TAGS.items()}

        # DateTime
        dt_key = tag_map.get("DateTime")
        if dt_key and dt_key in exif_raw:
            meta["datetime"] = exif_raw[dt_key]

        # Camera model
        model_key = tag_map.get("Model")
        if model_key and model_key in exif_raw:
            meta["camera_model"] = exif_raw[model_key]

        # GPS
        gps_key = tag_map.get("GPSInfo")
        if gps_key and gps_key in exif_raw:
            gps_raw = exif_raw[gps_key]
            meta["gps_raw"] = str(gps_raw)

    except Exception:
        pass
    return meta


def preprocess(img_bytes: bytes) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Full preprocessing pipeline.
    Returns (processed_array, metadata_dict).
    Returns (None, meta) if image is rejected (too blurry or unreadable).
    """
    # Fix orientation first
    img_bytes = fix_exif_orientation(img_bytes)
    meta = extract_exif(img_bytes)

    img = load_image(img_bytes)
    if img is None:
        return None, meta

    # Blur check
    score = blur_score(img)
    meta["blur_score"] = round(score, 2)

    if score < BLUR_THRESHOLD:
        logger.warning(f"[C-03] Image rejected — blur score {score:.1f} < {BLUR_THRESHOLD}")
        return None, meta

    # Resize if too large (keeps aspect ratio)
    h, w = img.shape[:2]
    if h > TARGET_SIZE[0] or w > TARGET_SIZE[1]:
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

    return img, meta
