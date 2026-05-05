"""
prahar/modules/c03_face/embedders.py
Three face embedding models:
  Model 1 — DeepFace ArcFace       → 512-dim vector
  Model 2 — InsightFace buffalo_l  → 512-dim vector (ONNX, no C++ needed)
  Model 3 — dlib ResNet            → 128-dim vector (optional, Windows fallback)

Each embedder returns Optional[np.ndarray] — None if face not detected.
"""
import os
import numpy as np
from typing import Optional
from loguru import logger


# ── Model 1: DeepFace ArcFace ────────────────────────────────
def embed_arcface(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    DeepFace ArcFace embedding — 512-dim.
    Most reliable across lighting/pose variations.
    """
    try:
        from deepface import DeepFace
        # DeepFace expects BGR numpy array or file path
        result = DeepFace.represent(
            img_path=img_bgr,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="opencv",  # fastest, good enough for cropped faces
        )
        if result and len(result) > 0:
            vec = np.array(result[0]["embedding"], dtype=np.float32)
            # L2-normalise for cosine similarity via dot product
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec
    except Exception as e:
        logger.debug(f"[C-03/ArcFace] {e}")
    return None


# ── Model 2: InsightFace buffalo_l (ONNX — no C++ needed) ───
_insight_app = None

def _get_insight_app():
    """Lazy-load InsightFace app (downloads buffalo_l on first call ~500 MB)."""
    global _insight_app
    if _insight_app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            _insight_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            _insight_app.prepare(ctx_id=-1, det_size=(640, 640))
            logger.info("[C-03] InsightFace buffalo_l loaded")
        except Exception as e:
            logger.warning(f"[C-03/InsightFace] Could not load: {e}")
            _insight_app = None
    return _insight_app


def embed_insightface(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    InsightFace buffalo_l embedding — 512-dim.
    Runs via ONNX — no MSVC needed on Windows.
    """
    app = _get_insight_app()
    if app is None:
        return None
    try:
        faces = app.get(img_bgr)
        if not faces:
            return None
        # Take largest face by bounding box area
        face = max(faces, key=lambda f: (
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        ))
        vec = np.array(face.embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        logger.debug(f"[C-03/InsightFace] {e}")
    return None


# ── Model 3: dlib ResNet face recognition ───────────────────
_dlib_encoder = None
_dlib_detector = None

def _get_dlib():
    global _dlib_encoder, _dlib_detector
    if _dlib_encoder is None:
        try:
            import dlib
            _dlib_detector = dlib.get_frontal_face_detector()
            model_path = os.path.join(
                os.path.expanduser("~"),
                ".prahar", "models",
                "dlib_face_recognition_resnet_model_v1.dat"
            )
            shape_path = os.path.join(
                os.path.expanduser("~"),
                ".prahar", "models",
                "shape_predictor_68_face_landmarks.dat"
            )
            if os.path.exists(model_path) and os.path.exists(shape_path):
                _dlib_encoder = dlib.face_recognition_model_v1(model_path)
                _dlib_predictor = dlib.shape_predictor(shape_path)
                logger.info("[C-03] dlib ResNet loaded")
            else:
                logger.warning(
                    "[C-03/dlib] Model files not found. "
                    "Download from: http://dlib.net/files/"
                    "dlib_face_recognition_resnet_model_v1.dat.bz2"
                )
        except ImportError:
            logger.warning("[C-03/dlib] Not installed — skipping (Windows OK)")
    return _dlib_encoder, _dlib_detector


def embed_dlib(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    dlib ResNet face descriptor — 128-dim.
    Optional on Windows (requires prebuilt wheel or Build Tools).
    """
    encoder, detector = _get_dlib()
    if encoder is None or detector is None:
        return None
    try:
        import dlib
        rgb = img_bgr[:, :, ::-1]   # BGR → RGB
        dets = detector(rgb, 1)
        if not dets:
            return None

        shape_path = os.path.join(
            os.path.expanduser("~"), ".prahar", "models",
            "shape_predictor_68_face_landmarks.dat"
        )
        predictor = dlib.shape_predictor(shape_path)
        shape = predictor(rgb, dets[0])
        descriptor = encoder.compute_face_descriptor(rgb, shape)
        vec = np.array(descriptor, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
    except Exception as e:
        logger.debug(f"[C-03/dlib] {e}")
    return None
