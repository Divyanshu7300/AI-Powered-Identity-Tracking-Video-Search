import tempfile
import numpy as np
from pathlib import Path

from app.services.semantic_search import SemanticPersonSearchIndex, SemanticObservation
from app.services.evidence_store import EvidenceStore
from app.services.query_planner import PersonSearchPlanner


def test_object_association():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = SemanticPersonSearchIndex(persist_dir=tmp_dir)
        person_bbox = [100, 100, 200, 400]
        frame_objects = [
            {"bbox": [110, 150, 180, 300], "class_name": "backpack"},
            {"bbox": [190, 80, 250, 150], "class_name": "umbrella"},
            {"bbox": [500, 500, 600, 600], "class_name": "car"},
        ]
        associated = index._associate_objects(person_bbox, frame_objects)
        assert "backpack" in associated
        assert "umbrella" in associated
        assert "car" not in associated


def test_vlm_fallback_and_metadata():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = SemanticPersonSearchIndex(persist_dir=tmp_dir)
        index.caption_model_setting = "heuristic"  # Force fast safeguard test

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = [50, 50, 150, 250]
        frame_shape = (480, 640, 3)
        associated = ["umbrella", "backpack"]

        caption, upper, lower, has_bag, has_umbrella, has_phone, loc = index._generate_vlm_caption(
            crop, bbox, frame_shape, associated
        )

        assert "person wearing" in caption
        assert "umbrella" in caption
        assert has_umbrella is True
        assert has_bag is True
        assert has_phone is False


def test_hybrid_search_scoring():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = SemanticPersonSearchIndex(persist_dir=tmp_dir)

        obs = SemanticObservation(
            observation_id="test:1:1",
            memory_id="test:1",
            track_id=1,
            source_name="test",
            source_label="test-video",
            frame_index=1,
            timestamp_seconds=1.0,
            bbox=[100, 100, 200, 400],
            caption="person with umbrella near center top",
            crop_url="/crops/test.jpg",
            frame_url="/evidence/test.jpg",
            embedding=np.zeros((512,), dtype=np.float32),
            objects=["umbrella"],
            has_umbrella=True,
            has_bag=False,
            upper_color="red",
            lower_color="black",
        )
        index.observations[obs.observation_id] = obs

        res = index.search(query="person with umbrella", top_k=5)
        assert len(res["matches"]) > 0
        assert res["matches"][0]["track_id"] == 1


def test_fast_search_filters_unrelated_low_score_results():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = SemanticPersonSearchIndex(persist_dir=tmp_dir)
        relevant = SemanticObservation(
            observation_id="test:1:1", memory_id="test:1", track_id=1, source_name="test",
            frame_index=1, timestamp_seconds=1.0, bbox=[0, 0, 10, 20],
            caption="person wearing blue shirt near left top", crop_url="", frame_url="",
            source_label="test-video",
            embedding=np.empty((0,), dtype=np.float32), upper_color="blue",
        )
        unrelated = SemanticObservation(
            observation_id="test:2:1", memory_id="test:2", track_id=2, source_name="test",
            frame_index=1, timestamp_seconds=1.0, bbox=[20, 0, 30, 20],
            caption="person wearing gray shirt near left top", crop_url="", frame_url="",
            source_label="test-video",
            embedding=np.empty((0,), dtype=np.float32), upper_color="gray",
        )
        index.observations = {relevant.observation_id: relevant, unrelated.observation_id: unrelated}

        res = index.search(query="blue shirt", top_k=8)

        assert [match["track_id"] for match in res["matches"]] == [1]


def test_vlm_failure_fast_fallback():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index = SemanticPersonSearchIndex(persist_dir=tmp_dir)
        index._florence_failed = True
        index._blip_failed = True

        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = [50, 50, 150, 250]
        frame_shape = (480, 640, 3)

        caption, upper, lower, _, _, _, _ = index._generate_vlm_caption(
            crop, bbox, frame_shape, ["backpack"]
        )
        assert "person wearing" in caption
        assert "backpack" in caption


def test_semantic_index_keeps_the_better_nearby_representative():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        store = EvidenceStore(str(root / "crops"), str(root / "evidence"))
        index = SemanticPersonSearchIndex(
            persist_dir=str(root / "semantic"),
            evidence_store=store,
            representatives_per_track=2,
            minimum_frame_gap=30,
        )
        # Keep this deterministic: detector confidence is one of the crop-quality inputs.
        index._crop_quality = lambda crop, track: float(track["confidence"])
        frame = np.full((120, 120, 3), 128, dtype=np.uint8)
        low_quality = {"track_id": 7, "bbox": [20, 10, 100, 110], "confidence": 0.4}
        high_quality = {"track_id": 7, "bbox": [20, 10, 100, 110], "confidence": 0.9}

        index.add_track_observations(frame, [low_quality], "video", 10)
        index.add_track_observations(frame, [high_quality], "video", 20)

        observations = list(index.observations.values())
        assert len(observations) == 1
        assert observations[0].frame_index == 20
        assert observations[0].quality_score == 0.9
        assert observations[0].crop_url.startswith("/media/crops/")


def test_semantic_index_limits_full_evidence_frames_per_track():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        store = EvidenceStore(str(root / "crops"), str(root / "evidence"))
        index = SemanticPersonSearchIndex(
            persist_dir=str(root / "semantic"), evidence_store=store,
            representatives_per_track=8, minimum_frame_gap=0, evidence_frames_per_track=3,
        )
        index._crop_quality = lambda crop, track: float(track["confidence"])
        frame = np.full((120, 120, 3), 128, dtype=np.uint8)
        for frame_index, confidence in enumerate((0.4, 0.5, 0.6, 0.9), 1):
            index.add_track_observations(
                frame, [{"track_id": 7, "bbox": [20, 10, 100, 110], "confidence": confidence}],
                "video", frame_index * 10,
            )

        observations = list(index.observations.values())
        assert len(observations) == 4
        assert len([item for item in observations if item.frame_url]) == 3
        assert len(list((root / "evidence").glob("*.jpg"))) == 3


def test_query_planner_relaxes_detailed_constraints_in_safe_order():
    plan = PersonSearchPlanner().build("blue shirt with backpack on left")
    assert plan.passes[0].name == "exact"
    assert plan.passes[1].name == "without_location"
    assert "horizontal_zone" in plan.passes[1].relaxed_fields
    assert plan.passes[-1].name == "semantic"
    assert {"upper_color", "objects"}.issubset(plan.passes[-1].relaxed_fields)
