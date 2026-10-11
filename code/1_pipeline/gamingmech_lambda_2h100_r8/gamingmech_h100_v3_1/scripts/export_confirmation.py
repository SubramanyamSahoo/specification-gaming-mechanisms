"""Export explicitly chosen settings; never tune thresholds to force significance."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamingmech.calibration import export_confirmation

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, help='Completed calibration project; its gate may have failed')
    p.add_argument('--model', choices=['qwen','llama'], default='qwen')
    p.add_argument('--output', required=True)
    p.add_argument('--max-new-tokens', required=True, type=int, help='A cap actually evaluated on calibration dev prompts')
    p.add_argument('--max-updates', required=True,
                   help='One number for every family, or per family such as qwen=192,llama=96; an explicit induction-strength choice')
    p.add_argument('--seeds', required=True, help='Fresh confirmatory seeds, e.g. 23,37,53')
    p.add_argument('--contamination-rates', required=True, help='Explicit rates such as 0.05,0.25; use none to decline the low-dose experiment')
    p.add_argument('--note', required=True, help='Reasoned decision based on pilot measurements')
    p.add_argument('--set', nargs='*', default=[], help='Optional explicit gate/training settings; recorded in the decision')
    a=p.parse_args()
    rates=() if a.contamination_rates.lower()=='none' else tuple(float(x) for x in a.contamination_rates.split(','))
    if '=' in a.max_updates:
        by_model={k.strip(): int(v) for k, v in (x.split('=') for x in a.max_updates.split(','))}
        if a.model not in by_model: p.error(f'--max-updates must name {a.model}')
        updates=by_model[a.model]
    else:
        by_model, updates={}, int(a.max_updates)
    cfg=export_confirmation(a.project, Path(a.output), model_key=a.model, max_new_tokens=a.max_new_tokens,
                            max_updates=updates, seeds=tuple(int(x) for x in a.seeds.split(',')),
                            contamination_rates=rates, note=a.note, overrides=a.set, max_updates_by_model=by_model)
    print('Wrote selected protocol:',Path(a.output).resolve())
    print('Additional primary-family low-dose adapters:',len(rates)*len(cfg.plan.contamination_objectives)*len(cfg.plan.contamination_seeds))
    print('No pilot weights or audit approval were promoted. Initialize a new project from this file.')
