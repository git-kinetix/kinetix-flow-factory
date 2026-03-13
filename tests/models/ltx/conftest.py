import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    """Free CUDA memory between tests to prevent OOM with the 22B model."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
