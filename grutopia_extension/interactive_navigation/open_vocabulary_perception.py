"""Pluggable open-vocabulary perception without import-time model loading."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests

from grutopia_extension.interactive_navigation.mapping import SemanticDetection


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class OpenVocabularyPerceptionConfig:
    grounding_dino_url: str = 'http://localhost:12181/gdino'
    mobile_sam_url: str = 'http://localhost:12183/mobile_sam'
    request_timeout: float = 8.0
    confidence_threshold: float = 0.25
    min_points: int = 1
    query_interval: float = 0.5
    clip_model_name: str = 'openai/clip-vit-base-patch32'
    clip_device: Optional[str] = None
    context_categories: Tuple[str, ...] = (
        'chair',
        'table',
        'sofa',
        'door',
        'cabinet',
        'bed',
        'lamp',
        'television',
        'refrigerator',
        'plant',
        'person',
    )

    def __post_init__(self) -> None:
        if self.request_timeout <= 0:
            raise ValueError('request_timeout must be positive')
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError('confidence_threshold must be between zero and one')
        if self.min_points <= 0:
            raise ValueError('min_points must be positive')
        if self.query_interval < 0:
            raise ValueError('query_interval must be non-negative')


@dataclass
class OpenVocabularyDetection:
    label: str
    confidence: float
    bbox: BBox
    mask: np.ndarray
    embedding: Optional[np.ndarray] = None
    centroid: Optional[Tuple[float, float, float]] = None
    point_count: int = 0
    step: int = 0

    def to_semantic_detection(self) -> SemanticDetection:
        """Convert the spatial portion to the mapping module's current type."""
        if self.centroid is None:
            raise ValueError('detection has no three-dimensional centroid')
        embedding = None
        if self.embedding is not None:
            embedding = tuple(float(value) for value in np.asarray(self.embedding).reshape(-1))
        return SemanticDetection(
            label=self.label,
            position=self.centroid,
            confidence=self.confidence,
            embedding=embedding,
            point_count=max(1, int(self.point_count)),
            step=int(self.step),
            sources=('open_vocabulary',),
        )


class AgentVLMBackend:
    """Short-timeout REST client for the existing Agent VLM services."""

    def __init__(
        self,
        config: Optional[OpenVocabularyPerceptionConfig] = None,
        session: Optional[Any] = None,
    ) -> None:
        self.config = config or OpenVocabularyPerceptionConfig()
        self._session = session or requests

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise ValueError('could not encode image')
        return base64.b64encode(encoded).decode('utf-8')

    def _post(self, url: str, payload: dict) -> dict:
        response = self._session.post(
            url,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=self.config.request_timeout,
            proxies={'http': '', 'https': ''},
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError('VLM service returned a non-object JSON response')
        return result

    def _get(self, url: str) -> dict:
        response = self._session.get(
            url,
            timeout=self.config.request_timeout,
            proxies={'http': '', 'https': ''},
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError('VLM health endpoint returned a non-object JSON response')
        return result

    def check_ready(self) -> dict:
        """Fail fast unless both HTTP perception services report ready."""

        status = {}
        endpoints = {
            'grounding_dino': self.config.grounding_dino_url,
            'mobile_sam': self.config.mobile_sam_url,
        }
        for name, endpoint in endpoints.items():
            health_url = endpoint.rsplit('/', 1)[0] + '/health'
            payload = self._get(health_url)
            if payload.get('ready') is not True:
                raise RuntimeError(f'{name} service is not ready at {health_url}')
            status[name] = health_url
        return status

    def detect(self, image: np.ndarray, caption: str) -> list[dict]:
        result = self._post(
            self.config.grounding_dino_url,
            {'image': self._encode_image(image), 'caption': caption},
        )
        boxes = result.get('boxes', ())
        scores = result.get('logits', result.get('scores', ()))
        labels = result.get('phrases', result.get('labels', ()))
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError('GroundingDINO response arrays have different lengths')
        return [
            {'bbox': box, 'confidence': float(np.asarray(score).max()), 'label': str(label)}
            for box, score, label in zip(boxes, scores, labels)
        ]

    def segment(self, image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
        result = self._post(
            self.config.mobile_sam_url,
            {'image': self._encode_image(image), 'bbox': [float(value) for value in bbox]},
        )
        encoded = result.get('cropped_mask')
        if not isinstance(encoded, str):
            raise ValueError('MobileSAM response is missing cropped_mask')
        raw = base64.b64decode(encoded, validate=True)
        expected_size = int(np.prod(image.shape[:2]))
        mask = np.frombuffer(raw, dtype=np.uint8)
        if mask.size != expected_size:
            raise ValueError(f'MobileSAM mask has {mask.size} pixels; expected {expected_size}')
        return mask.reshape(image.shape[:2]).astype(bool)


class OpenVocabularyPerception:
    """Detect, segment, embed, and associate image objects with world points."""

    def __init__(
        self,
        config: Optional[OpenVocabularyPerceptionConfig] = None,
        detector: Optional[Any] = None,
        segmenter: Optional[Any] = None,
        embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        simulator_fallback: Optional[Callable[..., Iterable[Any]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or OpenVocabularyPerceptionConfig()
        backend = None
        if detector is None or segmenter is None:
            backend = AgentVLMBackend(self.config)
        self.detector = detector if detector is not None else backend
        self.segmenter = segmenter if segmenter is not None else backend
        self.embedder = embedder
        self.simulator_fallback = simulator_fallback
        self._clock = clock
        self._last_query_time = float('-inf')
        self._last_query: Optional[str] = None
        self._last_result: list[Any] = []
        self._clip_processor: Optional[Any] = None
        self._clip_model: Optional[Any] = None
        self._clip_device: Optional[str] = None
        self.last_item_errors: list[dict] = []
        self.last_query_status = 'idle'
        # One small, JSON-serializable record for the camera frame most
        # recently sent to GroundingDINO. Never reuse it for a later step.
        self.last_debug_frame: Optional[dict] = None

    def check_ready(self, load_clip: bool = True) -> dict:
        """Validate remote services and local CLIP before a long run starts."""

        services = {}
        checked = set()
        for component in (self.detector, self.segmenter):
            if id(component) in checked:
                continue
            checked.add(id(component))
            checker = getattr(component, 'check_ready', None)
            if checker is not None:
                result = checker()
                if isinstance(result, dict):
                    services.update(result)
        if load_clip and self.embedder is None:
            self._ensure_clip()
        return {
            'services': services,
            'clip_device': self._clip_device if load_clip else None,
        }

    def build_query(self, target: str, include_context: bool = True) -> str:
        labels = [part.strip() for part in target.replace('|', '.').split('.') if part.strip()]
        if include_context:
            labels.extend(self.config.context_categories)
        unique = list(dict.fromkeys(label.casefold() for label in labels))
        return ' . '.join(unique) + (' .' if unique else '')

    def perceive(
        self,
        rgba: np.ndarray,
        depth: np.ndarray,
        point_image: np.ndarray,
        target: str,
        include_context: bool = True,
        step: int = 0,
    ) -> list[Any]:
        self.last_item_errors = []
        self.last_debug_frame = None
        rgb, depth_array, points = self._validate_inputs(rgba, depth, point_image)
        query = self.build_query(target, include_context=include_context)
        now = self._clock()
        debug_frame = {
            'step': int(step),
            'status': 'requested',
            'query': query,
            'image_size': [int(rgb.shape[1]), int(rgb.shape[0])],
            'boxes': [],
            'mapped': [],
            'item_errors': [],
        }
        self.last_debug_frame = debug_frame
        if query == self._last_query and now - self._last_query_time < self.config.query_interval:
            # A materialized detection contains geometry from a particular
            # RGB/depth frame. Replaying it would submit a stale world-space
            # centroid as a new observation when either the camera or object
            # has moved. Rate limiting therefore skips this frame instead of
            # caching three-dimensional observations.
            self.last_query_status = 'rate_limited'
            debug_frame['status'] = 'rate_limited'
            return []

        try:
            detections = self._run_detector(rgb, query)
            debug_frame['boxes'] = self._debug_detector_boxes(detections, rgb.shape[:2])
            result = self._materialize(detections, rgb, depth_array, points, step=step)
            debug_frame['mapped'] = [
                {
                    'label': detection.label,
                    'bbox_xyxy': [float(value) for value in detection.bbox],
                    'centroid': [float(value) for value in detection.centroid],
                    'point_count': int(detection.point_count),
                }
                for detection in result
                if isinstance(detection, OpenVocabularyDetection)
                and detection.centroid is not None
            ]
            debug_frame['item_errors'] = list(self.last_item_errors)
            debug_frame['status'] = 'ok'
            self.last_query_status = 'ok'
        except Exception as error:
            debug_frame['item_errors'] = list(self.last_item_errors)
            debug_frame['status'] = 'failed'
            debug_frame['error'] = {
                'type': type(error).__name__,
                'message': str(error),
            }
            self.last_query_status = 'failed'
            if self.simulator_fallback is None:
                raise
            else:
                result = list(
                    self.simulator_fallback(
                        rgba=rgba,
                        depth=depth,
                        point_image=point_image,
                        query=query,
                    )
                )
                self.last_query_status = 'fallback'
                debug_frame['status'] = 'fallback'

        self._last_query_time = now
        self._last_query = query
        self._last_result = list(result)
        return result

    def _debug_detector_boxes(self, detections: Iterable[Any], image_shape: tuple[int, int]) -> list[dict]:
        """Retain DINO boxes before thresholding, masking, and map association."""

        boxes = []
        for raw in detections:
            try:
                label, confidence, bbox = self._parse_detection(raw, image_shape)
                boxes.append({
                    'label': label,
                    'confidence': float(confidence),
                    'bbox_xyxy': [float(value) for value in bbox],
                    'above_threshold': bool(confidence >= self.config.confidence_threshold),
                })
            except Exception as error:
                boxes.append({
                    'label': self._raw_label(raw),
                    'error': f'{type(error).__name__}: {error}',
                })
        return boxes

    def as_semantic_detections(self, detections: Iterable[OpenVocabularyDetection]) -> list[SemanticDetection]:
        return [detection.to_semantic_detection() for detection in detections if detection.centroid is not None]

    @staticmethod
    def _validate_inputs(
        rgba: np.ndarray,
        depth: np.ndarray,
        point_image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = np.asarray(rgba)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError('rgba must have shape (height, width, 3 or 4)')
        depth_array = np.asarray(depth)
        points = np.asarray(point_image)
        if depth_array.shape != image.shape[:2]:
            raise ValueError('depth must align pixel-for-pixel with rgba')
        if points.shape != (*image.shape[:2], 3):
            raise ValueError('point_image must have shape (height, width, 3)')
        return np.ascontiguousarray(image[..., :3]), depth_array, points

    def _run_detector(self, image: np.ndarray, query: str) -> list[dict]:
        detector = self.detector
        if hasattr(detector, 'detect'):
            raw = detector.detect(image, query)
        elif hasattr(detector, 'predict'):
            raw = detector.predict(image, caption=query)
        else:
            raw = detector(image, query)
        if isinstance(raw, dict):
            boxes = raw.get('boxes', ())
            scores = raw.get('logits', raw.get('scores', ()))
            labels = raw.get('phrases', raw.get('labels', ()))
            raw = [
                {'bbox': box, 'confidence': float(np.asarray(score).max()), 'label': label}
                for box, score, label in zip(boxes, scores, labels)
            ]
        elif hasattr(raw, 'boxes') and hasattr(raw, 'logits') and hasattr(raw, 'phrases'):
            raw = [
                {'bbox': box, 'confidence': float(np.asarray(score).max()), 'label': label}
                for box, score, label in zip(raw.boxes, raw.logits, raw.phrases)
            ]
        return list(raw)

    def _materialize(
        self,
        raw_detections: Iterable[Any],
        image: np.ndarray,
        depth: np.ndarray,
        point_image: np.ndarray,
        step: int = 0,
    ) -> list[OpenVocabularyDetection]:
        output = []
        for raw in raw_detections:
            label = self._raw_label(raw)
            try:
                label, confidence, bbox = self._parse_detection(raw, image.shape[:2])
                if confidence < self.config.confidence_threshold:
                    continue
                mask = self._run_segmenter(image, bbox)
                if mask.shape != image.shape[:2]:
                    raise ValueError('segmenter mask must align pixel-for-pixel with the input')
                valid = (
                    mask
                    & np.isfinite(depth)
                    & (depth > 0)
                    & np.isfinite(point_image).all(axis=-1)
                )
                world_points = point_image[valid]
                if len(world_points) < self.config.min_points:
                    continue
                embedding = self._embed_masked_crop(image, mask)
                output.append(
                    OpenVocabularyDetection(
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        mask=mask,
                        embedding=embedding,
                        centroid=tuple(float(value) for value in world_points.mean(axis=0)),
                        point_count=int(len(world_points)),
                        step=int(step),
                    )
                )
            except Exception as error:
                self.last_item_errors.append(
                    {
                        'label': label,
                        'type': type(error).__name__,
                        'message': str(error),
                    }
                )
        if self.last_item_errors and not output:
            first = self.last_item_errors[0]
            raise RuntimeError(
                f'all materialized detections failed; first error for '
                f'{first["label"]!r}: {first["type"]}: {first["message"]}'
            )
        return output

    @staticmethod
    def _raw_label(raw: Any) -> str:
        if isinstance(raw, OpenVocabularyDetection):
            return raw.label
        if isinstance(raw, dict):
            return str(raw.get('label', raw.get('phrase', 'object')))
        if isinstance(raw, (tuple, list)) and raw:
            return str(raw[0])
        return 'unknown'

    @staticmethod
    def _parse_detection(raw: Any, image_shape: tuple[int, int]) -> tuple[str, float, BBox]:
        if isinstance(raw, OpenVocabularyDetection):
            return raw.label, raw.confidence, raw.bbox
        if isinstance(raw, dict):
            label = str(raw.get('label', raw.get('phrase', 'object')))
            confidence = float(raw.get('confidence', raw.get('score', raw.get('logit', 1.0))))
            box = np.array(raw.get('bbox', raw.get('box')), dtype=float, copy=True)
        else:
            label, confidence, box = raw
            box = np.array(box, dtype=float, copy=True)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('detector bbox must contain four finite values')
        height, width = image_shape
        if np.max(np.abs(box)) <= 1.5:
            box = box * np.array([width, height, width, height], dtype=float)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        return label, confidence, tuple(float(value) for value in box)

    def _run_segmenter(self, image: np.ndarray, bbox: BBox) -> np.ndarray:
        segmenter = self.segmenter
        if hasattr(segmenter, 'segment'):
            mask = segmenter.segment(image, bbox)
        elif hasattr(segmenter, 'segment_bbox'):
            mask = segmenter.segment_bbox(image, list(bbox))
        else:
            mask = segmenter(image, bbox)
        return np.asarray(mask, dtype=bool)

    def _embed_masked_crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rows, columns = np.nonzero(mask)
        if not len(rows):
            raise ValueError('cannot embed an empty mask')
        y1, y2 = rows.min(), rows.max() + 1
        x1, x2 = columns.min(), columns.max() + 1
        crop = image[y1:y2, x1:x2].copy()
        crop[~mask[y1:y2, x1:x2]] = 0
        vector = self.embedder(crop) if self.embedder is not None else self._clip_embed(crop)
        embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('embedder returned a zero or non-finite vector')
        return embedding / norm

    def _clip_embed(self, crop: np.ndarray) -> np.ndarray:
        self._ensure_clip()
        import torch

        inputs = self._clip_processor(images=crop, return_tensors='pt')
        inputs = {name: value.to(self._clip_device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = self._clip_model.get_image_features(**inputs)
        return features[0].detach().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalized CLIP text embedding for open-vocabulary matching."""

        if not str(text).strip():
            raise ValueError('text cannot be empty')
        self._ensure_clip()
        import torch

        inputs = self._clip_processor(text=[str(text)], return_tensors='pt', padding=True)
        inputs = {name: value.to(self._clip_device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = self._clip_model.get_text_features(**inputs)
        embedding = features[0].detach().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('CLIP returned an invalid text embedding')
        return embedding / norm

    def _ensure_clip(self):
        if self._clip_model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = self.config.clip_device or ('cuda' if torch.cuda.is_available() else 'cpu')
            self._clip_processor = CLIPProcessor.from_pretrained(
                self.config.clip_model_name,
                local_files_only=True,
            )
            self._clip_model = CLIPModel.from_pretrained(
                self.config.clip_model_name,
                local_files_only=True,
            ).to(device)
            self._clip_model.eval()
            self._clip_device = device


__all__ = [
    'AgentVLMBackend',
    'OpenVocabularyDetection',
    'OpenVocabularyPerception',
    'OpenVocabularyPerceptionConfig',
]
