import pytest
pytest.importorskip('transformers', reason='Install the pinned HF environment to run actual architecture tests')
pytest.importorskip('peft')
from gamingmech.integration import run_checks

def test_real_hf_tiny_architectures():
    results = run_checks()
    assert set(results) == {'qwen', 'llama'}
    assert all(x['passed'] for x in results.values())
