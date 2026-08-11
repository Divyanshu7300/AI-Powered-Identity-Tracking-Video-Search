from __future__ import annotations

from threading import RLock
from typing import Dict, Tuple

from models.yolo import YOLODetector


_cache_lock = RLock()
_detectors: Dict[Tuple[str], Tuple[YOLODetector, RLock]] = {}
_reid_encoders: Dict[Tuple[str | None, str], Tuple[object, RLock]] = {}
_clip_models: Dict[Tuple[str], Tuple[object, object, str, RLock]] = {}
_clip_errors: Dict[Tuple[str], str] = {}
_florence_models: Dict[Tuple[str], Tuple[object, object, str, RLock]] = {}
_florence_errors: Dict[Tuple[str], str] = {}
_blip_models: Dict[Tuple[str], Tuple[object, object, str, RLock]] = {}
_blip_errors: Dict[Tuple[str], str] = {}


def get_detector(model_path: str, conf_threshold: float) -> Tuple[YOLODetector, RLock]:
    key = (model_path,)
    with _cache_lock:
        cached = _detectors.get(key)
        if cached is None:
            cached = (
                YOLODetector(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                ),
                RLock(),
            )
            _detectors[key] = cached
        detector, detector_lock = cached
    return detector, detector_lock


def get_reid_encoder(weights_path: str | None, model_name: str) -> Tuple[object, RLock]:
    key = (weights_path, model_name)
    with _cache_lock:
        cached = _reid_encoders.get(key)
        if cached is None:
            from models.reid_model import ReIDEncoder

            cached = (
                ReIDEncoder(
                    weights_path=weights_path,
                    model_name=model_name,
                ),
                RLock(),
            )
            _reid_encoders[key] = cached
    return cached


def get_clip(model_name: str = "google/siglip-base-patch16-224") -> Tuple[object, object, str, RLock]:
    key = (model_name,)
    with _cache_lock:
        cached = _clip_models.get(key)
        if cached is not None:
            return cached
        if key in _clip_errors:
            raise RuntimeError(f"CLIP model loading failed previously: {_clip_errors[key]}")

    try:
        import torch
        # Auto classes support both legacy CLIP and newer SigLIP models through
        # the same get_image_features / get_text_features interface.
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        cached = (model, processor, device, RLock())
        with _cache_lock:
            _clip_models[key] = cached
            _clip_errors.pop(key, None)
        return cached
    except Exception as exc:
        with _cache_lock:
            _clip_errors[key] = str(exc)
        raise


def get_florence(model_name: str = "microsoft/Florence-2-base") -> Tuple[object, object, str, RLock]:
    key = (model_name,)
    with _cache_lock:
        cached = _florence_models.get(key)
        if cached is not None:
            return cached
        if key in _florence_errors:
            raise RuntimeError(f"Florence model loading failed previously: {_florence_errors[key]}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model.eval()
        cached = (model, processor, device, RLock())
        with _cache_lock:
            _florence_models[key] = cached
            _florence_errors.pop(key, None)
        return cached
    except Exception as exc:
        with _cache_lock:
            _florence_errors[key] = str(exc)
        raise


def get_blip(model_name: str = "Salesforce/blip-image-captioning-base") -> Tuple[object, object, str, RLock]:
    key = (model_name,)
    with _cache_lock:
        cached = _blip_models.get(key)
        if cached is not None:
            return cached
        if key in _blip_errors:
            raise RuntimeError(f"BLIP model loading failed previously: {_blip_errors[key]}")

    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        cached = (model, processor, device, RLock())
        with _cache_lock:
            _blip_models[key] = cached
            _blip_errors.pop(key, None)
        return cached
    except Exception as exc:
        with _cache_lock:
            _blip_errors[key] = str(exc)
        raise


def warmup(
    *,
    detector_model: str | None = None,
    detector_conf_threshold: float = 0.5,
    reid_weights: str | None = None,
    reid_model_name: str | None = None,
    clip_model_name: str | None = None,
    florence_model_name: str | None = None,
    blip_model_name: str | None = None,
) -> Dict[str, object]:
    warmed: Dict[str, object] = {}
    if detector_model:
        get_detector(detector_model, detector_conf_threshold)
        warmed["detector"] = detector_model
    if reid_model_name:
        get_reid_encoder(reid_weights, reid_model_name)
        warmed["reid"] = reid_model_name
    if clip_model_name:
        get_clip(clip_model_name)
        warmed["clip"] = clip_model_name
    if florence_model_name:
        get_florence(florence_model_name)
        warmed["florence"] = florence_model_name
    if blip_model_name:
        get_blip(blip_model_name)
        warmed["blip"] = blip_model_name
    return {"warmed": warmed, "status": status()}


def status() -> Dict[str, object]:
    with _cache_lock:
        return {
            "detectors": len(_detectors),
            "reid_encoders": len(_reid_encoders),
            "clip_models": len(_clip_models),
            "florence_models": len(_florence_models),
            "blip_models": len(_blip_models),
            "detector_keys": [key[0] for key in _detectors.keys()],
            "reid_keys": [
                {
                    "weights_path": weights_path,
                    "model_name": model_name,
                }
                for weights_path, model_name in _reid_encoders.keys()
            ],
            "clip_keys": [key[0] for key in _clip_models.keys()],
            "florence_keys": [key[0] for key in _florence_models.keys()],
            "blip_keys": [key[0] for key in _blip_models.keys()],
            "errors": {
                "clip": {key[0]: error for key, error in _clip_errors.items()},
                "florence": {key[0]: error for key, error in _florence_errors.items()},
                "blip": {key[0]: error for key, error in _blip_errors.items()},
            },
        }
