import json
import os
import tempfile
from pathlib import Path
import pytest


class TestScanDataset:
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        (tmp_path / "videos").mkdir()
        (tmp_path / "condition" / "pose_depth").mkdir(parents=True)
        (tmp_path / "ref_images").mkdir()

        for vid_id in ["clip_001", "clip_002"]:
            (tmp_path / "videos" / f"{vid_id}.mp4").write_bytes(b"\x00")
            (tmp_path / "condition" / "pose_depth" / f"{vid_id}.mp4").write_bytes(b"\x00")
            (tmp_path / "ref_images" / f"{vid_id}.png").write_bytes(b"\x00")

        prompts = {"clip_001": "A person dancing", "clip_002": "A person walking"}
        (tmp_path / "prompts.json").write_text(json.dumps(prompts))
        return tmp_path

    def test_finds_all_videos(self, mock_dataset):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from prepare_realisdance import scan_dataset
        samples = scan_dataset(str(mock_dataset))
        assert len(samples) == 2

    def test_sample_has_required_keys(self, mock_dataset):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from prepare_realisdance import scan_dataset
        samples = scan_dataset(str(mock_dataset))
        for s in samples:
            assert "video_id" in s
            assert "video" in s
            assert "prompt" in s
            assert "reference_video" in s
            assert "image" in s

    def test_prompts_loaded_from_json(self, mock_dataset):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from prepare_realisdance import scan_dataset
        samples = scan_dataset(str(mock_dataset))
        by_id = {s["video_id"]: s for s in samples}
        assert by_id["clip_001"]["prompt"] == "A person dancing"

    def test_handles_missing_condition(self, mock_dataset):
        import sys
        sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
        from prepare_realisdance import scan_dataset
        (mock_dataset / "condition" / "pose_depth" / "clip_002.mp4").unlink()
        samples = scan_dataset(str(mock_dataset))
        by_id = {s["video_id"]: s for s in samples}
        assert "reference_video" not in by_id["clip_002"]
