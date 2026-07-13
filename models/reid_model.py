from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms


@dataclass
class EmbeddingBatch:
    embeddings: np.ndarray
    valid_indices: List[int]


class ReIDEncoder(nn.Module):
    def __init__(
        self,
        device: Optional[str] = None,
        weights_path: Optional[str] = None,
        model_name: str = "osnet_x0_25",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_name = model_name
        self.pretrained = pretrained
        self.backend = "torchreid"
        self.fallback_error: str | None = None

        try:
            from torchreid.reid import models as torchreid_models

            self.model = torchreid_models.build_model(
                name=model_name,
                num_classes=1,
                loss="softmax",
                pretrained=pretrained,
                use_gpu=self.device.type == "cuda",
            )
            self.embedding_dim = int(
                getattr(self.model, "feature_dim", 512)
            )
        except Exception as exc:
            self.model = None
            self.backend = "color_histogram"
            self.fallback_error = str(exc)
            self.embedding_dim = 48

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.to(self.device)
        self.eval()

        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> None:
        if self.model is None:
            raise RuntimeError(
                "Cannot load Re-ID weights because TorchReID is unavailable; "
                f"fallback backend is active: {self.fallback_error}"
            )

        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
        )

        state_dict = checkpoint.get(
            "state_dict",
            checkpoint,
        )

        self.model.load_state_dict(
            state_dict,
            strict=False,
        )

        self.eval()

    def preprocess(
        self,
        crop: np.ndarray,
    ) -> Optional[torch.Tensor]:

        if crop is None or crop.size == 0:
            return None

        rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        return self.transform(rgb)

    def forward(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("TorchReID model is unavailable; use encode_crops for fallback embeddings.")

        embeddings = self.model(tensor)

        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]

        return nn.functional.normalize(
            embeddings,
            dim=1,
        )

    @torch.inference_mode()
    def encode_crops(
        self,
        crops: Iterable[np.ndarray],
        batch_size: int = 32,
    ) -> EmbeddingBatch:

        all_embeddings = []
        all_valid_indices = []

        crop_list = list(crops)
        if self.model is None:
            return self._encode_histogram_crops(crop_list)

        for batch_start in range(
            0,
            len(crop_list),
            batch_size,
        ):

            batch_end = min(
                batch_start + batch_size,
                len(crop_list),
            )

            batch_crops = crop_list[
                batch_start:batch_end
            ]

            tensors = []
            batch_valid_indices = []

            for local_idx, crop in enumerate(batch_crops):

                tensor = self.preprocess(crop)

                if tensor is None:
                    continue

                tensors.append(tensor)

                batch_valid_indices.append(
                    batch_start + local_idx
                )

            if not tensors:
                continue

            batch = torch.stack(tensors).to(
                self.device
            )

            embeddings = (
                self(batch)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_embeddings.append(embeddings)

            all_valid_indices.extend(
                batch_valid_indices
            )

        if not all_embeddings:
            return EmbeddingBatch(
                embeddings=np.empty(
                    (0, self.embedding_dim),
                    dtype=np.float32,
                ),
                valid_indices=[],
            )

        all_embeddings = np.vstack(all_embeddings)

        return EmbeddingBatch(
            embeddings=all_embeddings,
            valid_indices=all_valid_indices,
        )

    def _encode_histogram_crops(
        self,
        crops: List[np.ndarray],
    ) -> EmbeddingBatch:

        embeddings = []
        valid_indices = []
        for index, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            vector = self._color_histogram(crop)
            if vector.size == 0:
                continue
            embeddings.append(vector)
            valid_indices.append(index)

        if not embeddings:
            return EmbeddingBatch(
                embeddings=np.empty((0, self.embedding_dim), dtype=np.float32),
                valid_indices=[],
            )

        return EmbeddingBatch(
            embeddings=np.vstack(embeddings).astype(np.float32),
            valid_indices=valid_indices,
        )

    def _color_histogram(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        channels = cv2.split(hsv)
        histograms = [
            cv2.calcHist([channel], [0], None, [16], [0, 256]).reshape(-1)
            for channel in channels
        ]
        vector = np.concatenate(histograms).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> "ReIDEncoder":

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        return cls(
            device=device,
            weights_path=checkpoint_path,
        )
