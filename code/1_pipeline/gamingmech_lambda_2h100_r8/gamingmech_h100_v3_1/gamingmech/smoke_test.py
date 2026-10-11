"""Compatibility entry point for actual HF architecture checks. No experiment is run."""
from .integration import run_checks
if __name__ == "__main__":
    import json
    print(json.dumps(run_checks(), indent=2))
