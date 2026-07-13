from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2


class TrackClipExporter:
    """Exports the source-video segment where a track is visible."""

    def __init__(self, clips_dir: str = "data/clips") -> None:
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        memory: Dict[str, object],
        padding_frames: int = 0,
    ) -> Dict[str, object]:
        source_path = Path(str(memory.get("source_path") or ""))
        if not source_path.exists():
            raise FileNotFoundError(f"Source video not found for memory: {memory.get('memory_id')}")

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open source video: {source_path}")

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if source_fps <= 0 or source_fps > 120:
                source_fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            first_frame = max(1, int(memory.get("first_frame") or 1))
            last_frame = max(first_frame, int(memory.get("last_frame") or first_frame))
            start_frame = max(1, first_frame - max(0, int(padding_frames)))
            end_frame = last_frame + max(0, int(padding_frames))

            output_path = self._output_path(str(memory["memory_id"]))
            writer = self._open_writer(output_path, source_fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"Unable to write clip: {output_path}")

            timeline = self._timeline_by_frame(memory)
            frames_exported = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
            current_frame = start_frame
            while current_frame <= end_frame:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame = self._draw_track_context(
                    frame,
                    memory,
                    self._timeline_item_for_frame(timeline, current_frame),
                )
                writer.write(frame)
                frames_exported += 1
                current_frame += 1
            writer.release()
        finally:
            cap.release()

        return {
            "memory_id": memory["memory_id"],
            "clip_path": str(output_path),
            "clip_url": f"/clips/{output_path.name}",
            "frames_exported": frames_exported,
            "start_frame": start_frame,
            "end_frame": current_frame - 1,
            "padding_frames": max(0, int(padding_frames)),
        }

    def _output_path(self, memory_id: str) -> Path:
        safe_name = memory_id.replace(":", "_").replace("/", "_")
        return self.clips_dir / f"{safe_name}_visible_segment.mp4"

    def _timeline_by_frame(self, memory: Dict[str, object]) -> Dict[int, Dict[str, object]]:
        timeline = memory.get("timeline") or []
        if not isinstance(timeline, list):
            return {}
        return {
            int(item["frame_index"]): item
            for item in timeline
            if isinstance(item, dict) and item.get("frame_index") is not None
        }

    def _timeline_item_for_frame(
        self,
        timeline: Dict[int, Dict[str, object]],
        frame_index: int,
    ) -> Dict[str, object] | None:
        if not timeline:
            return None
        if frame_index in timeline:
            return timeline[frame_index]
        earlier_frames = [index for index in timeline if index <= frame_index]
        if earlier_frames:
            return timeline[max(earlier_frames)]
        return timeline[min(timeline)]

    def _draw_track_context(
        self,
        frame,
        memory: Dict[str, object],
        timeline_item: Dict[str, object] | None,
    ):
        bbox = None
        if timeline_item:
            bbox = timeline_item.get("bbox")
        if bbox is None:
            bbox = memory.get("latest_bbox")
        if not bbox:
            return frame

        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return frame

        cv2.rectangle(frame, (x1, y1), (x2, y2), (52, 226, 197), 3)
        label = f"Track {memory.get('track_id')}"
        timestamp = None
        if timeline_item:
            timestamp = timeline_item.get("timestamp_seconds")
        if timestamp is not None:
            label = f"{label} {timestamp}s"
        cv2.putText(
            frame,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (52, 226, 197),
            2,
        )
        return frame

    def _open_writer(self, output_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*codec),
                fps,
                size,
            )
            if writer.isOpened():
                return writer
            writer.release()
        return cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
