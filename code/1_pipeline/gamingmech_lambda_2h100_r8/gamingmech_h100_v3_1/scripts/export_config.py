"""Print a complete starting configuration without editing distribution files."""
import argparse
import json
from pathlib import Path
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role',choices=['calibration','confirmatory'],default='calibration')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    name='calibration.json' if a.role=='calibration' else 'h100.json'
    print(json.dumps(json.loads((root/'configs'/name).read_text()),indent=2))
