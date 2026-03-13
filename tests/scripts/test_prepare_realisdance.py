import json
import os
import tempfile
from pathlib import Path
import pytest


class TestScanDataset:
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        (tmp_path / "gt").mkdir()
        (tmp_path / "vace_conditioning").mkdir()
        (tmp_path / "ref").mkdir()

        for vid_id in ["clip_001", "clip_002"]:
            (tmp_path / "gt" / f"{vid_id}.mp4").write_bytes(b"\x00")
            (tmp_path / "vace_conditioning" / f"{vid_id}.mp4").write_bytes(b"\x00")
            (tmp_path / "ref" / f"{vid_id}.png").write_bytes(b"\x00")

        # Write prompts as CSV (id,prompt)
        csv_content = 'id,prompt\nclip_001,"A person dancing"\nclip_002,"A person walking"\n'
        (tmp_path / "prompts.csv").write_text(csv_content)
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
            assert "video" in s  # condition video path
            assert "prompt" in s
            assert "gt_video" in s
            assert "image" in s

    def test_prompts_loaded_from_csv(self, mock_dataset):
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
        (mock_dataset / "vace_conditioning" / "clip_002.mp4").unlink()
        samples = scan_dataset(str(mock_dataset))
        # clip_002 should be skipped (no condition video)
        assert len(samples) == 1
        assert samples[0]["video_id"] == "clip_001"
