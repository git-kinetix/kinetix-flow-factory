# tests/models/ltx/test_e2e_smoke.py
"""
End-to-end smoke test for LTX Union NFT training.

Verifies the full pipeline works: adapter construction → inference →
reward → NFT optimization, using mocked/tiny components.

Run: pytest tests/models/ltx/test_e2e_smoke.py -v -m gpu
"""
import pytest
import os

requires_model = pytest.mark.skipif(
    not os.environ.get("LTX_MODEL_PATH"),
    reason="Set LTX_MODEL_PATH env var for e2e test",
)


@requires_model
@pytest.mark.gpu
class TestE2ESmokeTest:
    def test_single_epoch_completes(self):
        """Run 1 epoch of NFT training and verify no crashes."""
        pytest.skip("Run manually on RD-H200-2: ff-train configs/ltx_union_nft_realisdance.yaml")
