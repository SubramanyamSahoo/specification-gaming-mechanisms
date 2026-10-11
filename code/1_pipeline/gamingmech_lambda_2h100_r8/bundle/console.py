#!/usr/bin/env python3
"""Interactive, logged orchestration of the UNMODIFIED v3.1 + dual-H100 packages.

No automatic scientific tuning. No secrets saved in state. Calibration stops before
confirmation. Each model has its own locked project; both ledgers count the same bill.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'gamingmech_h100_v3_1'
TOOLS = ROOT / 'gamingmech_dual_h100_tools'
WORK = ROOT / 'work'
PYTHON = PACKAGE / '.venv/bin/python'
DEFAULT_CONFIG = ROOT / 'configs/dual_calibration.json'
ACTIONS = ('install', 'calibration', 'status', 'caps', 'freeze', 'confirm-pilot',
           'confirm', 'mechanisms', 'features', 'minimal', 'trajectory')


def stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def ask(label: str, default: str | None = None) -> str:
    value = input(label + (f' [{default}]' if default is not None else '') + ': ').strip()
    if not value and default is not None:
        return default
    if not value:
        raise ValueError(f'{label} must not be empty')
    return value


def checked_billing(rate_text: str, start_text: str) -> dict:
    rate = float(rate_text)
    dt = datetime.fromisoformat(start_text.replace('Z', '+00:00'))
    if not math.isfinite(rate) or rate <= 0 or dt.tzinfo is None:
        raise ValueError('Use a finite positive TOTAL hourly rate and a timezone-aware actual start.')
    if dt > datetime.now(timezone.utc):
        raise ValueError('Billing start must not be in the future.')
    return {'hourly_usd': rate, 'started_utc': dt.isoformat()}


def verify_bundle() -> None:
    sys.path.insert(0, str(ROOT))
    from verify_bundle import verify
    verify(ROOT)


@contextmanager
def workflow_lock():
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / '.workflow.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise RuntimeError('Another console operation is running. Reattach; do not start a duplicate.') from ex
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_state() -> tuple[Path, dict]:
    active = read(WORK / 'active.json')
    workspace = Path(active['workspace']).resolve()
    state = read(workspace / 'state.json')
    if Path(state['bundle_root']) != ROOT:
        raise RuntimeError('The bundle moved after setup. Restore its original path; do not rewrite source locks.')
    cfg = workspace / 'calibration.json'
    if sha(cfg) != state['calibration_config_sha256']:
        raise RuntimeError('Calibration config changed. Create a separate documented trial, not an in-place edit.')
    return workspace, state


def create_state(config: Path, billing: dict) -> tuple[Path, dict]:
    if (WORK / 'active.json').exists():
        raise RuntimeError('An active study already exists; resume it to preserve billing and results.')
    cfg = read(config)
    if cfg.get('runtime', {}).get('study_role') != 'calibration':
        raise ValueError('Initial configuration must have the calibration role.')
    work = WORK / ('study_' + stamp())
    work.mkdir(parents=True, exist_ok=False)
    (work / 'calibration.json').write_bytes(config.read_bytes())
    state = {'bundle_root': str(ROOT), 'created_utc': datetime.now(timezone.utc).isoformat(),
             'calibration_config_sha256': sha(work / 'calibration.json'), 'billing_entered': billing,
             'cal_q': str(work / 'calibration_qwen'), 'cal_l': str(work / 'calibration_llama'),
             'conf_q': str(work / 'confirmation_qwen'), 'conf_l': str(work / 'confirmation_llama')}
    atomic_json(work / 'state.json', state)
    atomic_json(WORK / 'active.json', {'workspace': str(work)})
    return work, state


class Runner:
    def __init__(self, action: str):
        self.action = action
        self.out = WORK / 'console_logs' / (stamp() + '-' + action)
        self.out.mkdir(parents=True, exist_ok=False)
        self.serial = 0
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        self.threads = min(8, max(1, cpus // 2))
        requested = os.environ.get('GAMING_THREADS_PER_JOB')
        if requested:
            self.threads = int(requested)
        if self.threads < 1 or self.threads * 2 > cpus:
            raise ValueError('GAMING_THREADS_PER_JOB oversubscribes available CPU affinity.')
        self.env = os.environ.copy()
        self.env.update(OMP_NUM_THREADS=str(self.threads), MKL_NUM_THREADS=str(self.threads),
                        OPENBLAS_NUM_THREADS=str(self.threads), NUMEXPR_NUM_THREADS=str(self.threads),
                        TOKENIZERS_PARALLELISM='false', USE_TORCH='1', USE_TF='0',
                        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        self.env.setdefault('HF_HOME', str(Path.home() / '.cache/huggingface'))
        self.env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        self.gpus: list[str] = []
        print(f'Console logs: {self.out}', flush=True)

    def call(self, argv, *, env: dict | None = None) -> None:
        argv = [str(x) for x in argv]
        self.serial += 1
        log = self.out / f'{self.serial:03d}.log'
        meta = {'argv': argv, 'cwd': str(ROOT), 'started_utc': datetime.now(timezone.utc).isoformat()}
        # Never serialize process environment or authentication tokens.
        secret = self.env.get('HF_TOKEN', '')
        if secret and any(secret in x for x in argv):
            raise RuntimeError('A credential appeared in command arguments; refusing to log or launch it.')
        atomic_json(log.with_suffix('.json'), meta)
        print('\n$ ' + shlex.join(argv) + f'\nLog: {log}', flush=True)
        proc = None
        t0 = time.monotonic()
        status = None
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, env=(env or self.env), start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8', errors='replace', bufsize=1)
            assert proc.stdout is not None
            with log.open('w', encoding='utf-8') as f:
                for line in proc.stdout:
                    clean = line.replace(secret, '[REDACTED]') if secret else line
                    print(clean, end='', flush=True); f.write(clean); f.flush()
            status = proc.wait()
            if status != 0:
                raise subprocess.CalledProcessError(status, argv)
        finally:
            if proc is not None and proc.poll() is None:
                print('\nStopping the active command and waiting for its cleanup...', flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            if proc is not None:
                status = proc.returncode
                if proc.stdout is not None:
                    proc.stdout.close()
            meta.update(exit_code=status, elapsed_seconds=time.monotonic() - t0)
            atomic_json(log.with_suffix('.json'), meta)

    def authenticate(self) -> None:
        token = os.environ.get('HF_TOKEN')
        if not token:
            token = getpass.getpass('Hugging Face token (hidden; not saved by this console): ').strip()
        if not token:
            raise ValueError('HF_TOKEN is required, with access to the Llama model.')
        self.env['HF_TOKEN'] = token
        # Keep it only for this live console; tmux server environment is not changed.
        os.environ['HF_TOKEN'] = token

    def hardware(self) -> None:
        sys.path.insert(0, str(TOOLS))
        from parallel import select_gpus
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total',
                                       '--format=csv,noheader,nounits'], text=True)
        selected = select_gpus(text, (0, 1))
        self.gpus = [x['uuid'] for x in selected]
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                       '--format=csv,noheader'], text=True)
        if any(uid in busy for uid in self.gpus):
            raise RuntimeError('Selected GPUs already have compute processes. Inspect nvidia-smi; do not run duplicate jobs.\n' + busy)
        print(text, end='', flush=True)

    def pkg(self, project: Path, model: str, *args, phase: str = 'pilot') -> None:
        env = self.env.copy()
        env['CUDA_VISIBLE_DEVICES'] = self.gpus[0 if model == 'qwen' else 1] if self.gpus else '0'
        self.call(['bash', PACKAGE / 'scripts/run.sh', '--project', project, '--model', model,
                   '--phase', phase, *args], env=env)

    def ensure_install(self) -> None:
        marker = WORK / 'installation_passed.json'
        if (marker.exists() and PYTHON.is_file() and
                read(marker).get('core_manifest_sha256') == sha(PACKAGE / 'PACKAGE_SHA256.json')):
            print('Using the previously installed, checked environment.')
            return
        self.call(['bash', PACKAGE / 'scripts/install.sh'])
        self.call([PYTHON, '-m', 'unittest', 'discover', '-s', ROOT / 'tests', '-p', 'test_*.py'])
        self.call([PYTHON, '-m', 'unittest', 'discover', '-s', TOOLS, '-p', 'test_tools.py'])
        atomic_json(marker, {'utc': datetime.now(timezone.utc).isoformat(),
                             'core_manifest_sha256': sha(PACKAGE / 'PACKAGE_SHA256.json')})

    def parallel(self, workspace: Path, state: dict, stage: str, confirmation: bool = False) -> None:
        prefix = 'conf' if confirmation else 'cal'
        self.call([PYTHON, TOOLS / 'parallel.py', '--package', PACKAGE,
                   '--qwen-project', state[prefix + '_q'], '--llama-project', state[prefix + '_l'],
                   '--stage', stage, '--threads-per-job', self.threads, '--logs', workspace / 'launch_logs'])


def ensure_project(r: Runner, project: Path, config: Path, model: str,
                   parent: Path | None = None) -> None:
    if project.exists():
        if not (project / 'protocol.json').exists() or (project / 'HANDOFF_FAILED.json').exists():
            raise RuntimeError(f'Incomplete project needs explicit recovery, not overwrite: {project}')
        return
    args = ['--init', '--config', str(config)]
    if parent is not None:
        args += ['--data-from', str(parent), '--billing-from', str(parent)]
    r.pkg(project, model, *args)


def calibration(r: Runner) -> None:
    r.ensure_install(); r.authenticate(); r.hardware()
    if (WORK / 'active.json').exists():
        workspace, state = load_state()
    else:
        d = read(DEFAULT_CONFIG)
        by_model = d['train'].get('max_updates_by_model') or {}
        updates = ', '.join(f'{k} {v}' for k, v in sorted(by_model.items())) or str(d['train']['max_updates'])
        print(f"\nThe default is a CALIBRATION CANDIDATE: {d['train']['pair_batch_size']} pairs × "
              f"{d['train']['grad_accum']} accumulation; updates {updates}; "
              f"{d['eval']['max_new_tokens']} output tokens.")
        print('It is not certified sufficient or fastest. Data/quality requirements are unchanged.')
        config = Path(ask('Initial calibration JSON', str(DEFAULT_CONFIG))).expanduser().resolve()
        billing = checked_billing(ask('TOTAL two-H100 instance price in USD/hour'),
                                  ask('Actual instance billing start, ISO timestamp WITH timezone'))
        workspace, state = create_state(config, billing)
    print(f'Active study: {workspace}')
    if ask('Type RUN to start/resume data preparation, both profiles and eight one-seed calibration adapters') != 'RUN':
        raise RuntimeError('Not started; saved state can be resumed later.')
    q, l = Path(state['cal_q']), Path(state['cal_l'])
    ensure_project(r, q, workspace / 'calibration.json', 'qwen')
    if not (q / 'billing.json').exists():
        b = state['billing_entered']
        r.pkg(q, 'qwen', '--clock', '--hourly-usd', str(b['hourly_usd']), '--started-utc', b['started_utc'])
    r.pkg(q, 'qwen', '--preflight')
    r.pkg(q, 'qwen', '--stages', 'data,validate')
    ensure_project(r, l, workspace / 'calibration.json', 'llama', q)
    r.pkg(l, 'llama', '--preflight')
    r.parallel(workspace, state, 'profile')
    r.parallel(workspace, state, 'pilot')
    for model, p in [('qwen', q), ('llama', l)]:
        r.pkg(p, model, '--calibration-summary')
    print('\nCALIBRATION FINISHED. Read both families before choosing confirmation settings.')
    print('No confirmation or mechanism jobs start automatically. GPUs are now idle; billing continues.')


def require_calibration_completed(state: dict) -> None:
    for model, key in [('qwen', 'cal_q'), ('llama', 'cal_l')]:
        p = Path(state[key])
        receipt = p / model / 'receipts/pilot/behavior.json'
        gate = p / model / '04_behavior/pilot/dev/artifacts/gate_pilot.json'
        if not receipt.exists() or not read(receipt).get('passed') or not gate.exists():
            raise RuntimeError(f'Complete {model} calibration behavior before selecting the final protocol.')


def freeze(r: Runner) -> None:
    workspace, state = load_state(); require_calibration_completed(state)
    r.ensure_install(); r.authenticate(); r.hardware()
    selected = workspace / 'selected_confirmation.json'
    if not selected.exists():
        print('Choose from measured DEVELOPMENT outputs, not confirmatory outcomes. No automatic threshold lowering.')
        cap = int(ask('Selected output cap actually evaluated in calibration'))
        for model, key in [('qwen', 'cal_q'), ('llama', 'cal_l')]:
            summary = Path(state[key]) / model / '00_calibration/pilot/artifacts/calibration_summary.json'
            if not summary.exists() or cap not in read(summary).get('observed_token_caps', []):
                raise ValueError(f'Measure cap {cap} for {model} with the cap-sweep action before freezing.')
        updates = ask('Selected optimizer updates per adapter: one number, or per family such as qwen=192,llama=96')
        parts = [x.split('=') for x in updates.split(',')] if '=' in updates else [['all', updates]]
        if not all(len(x) == 2 and x[1].strip().isdigit() and int(x[1]) > 0 for x in parts):
            raise ValueError('Give a positive number, or qwen=N,llama=M')
        seeds = ask('Fresh confirmation seeds, comma separated; example 23,37,53')
        rates = ask('Explicit eligible-pair contamination rates, comma separated, or none')
        note = ask('Decision rationale based on BOTH model-family calibration reports')
        overrides = shlex.split(input('Optional justified --set key=value overrides (Enter for none): ').strip())
        cmd = [PYTHON, PACKAGE / 'scripts/export_confirmation.py', '--project', state['cal_q'],
               '--model', 'qwen', '--output', selected, '--max-new-tokens', str(cap), '--max-updates', str(updates),
               '--seeds', seeds, '--contamination-rates', rates, '--note', note]
        if overrides:
            cmd += ['--set', *overrides]
        r.call(cmd)
        atomic_json(workspace / 'confirmation_selection_hash.json', {'sha256': sha(selected)})
    else:
        mark = workspace / 'confirmation_selection_hash.json'
        if not mark.exists() or read(mark)['sha256'] != sha(selected):
            raise RuntimeError('Existing selection is unregistered or changed; do not silently resume.')
    for model, key in [('qwen', 'conf_q'), ('llama', 'conf_l')]:
        p = Path(state[key])
        ensure_project(r, p, selected, model, Path(state['cal_q']))
        r.pkg(p, model, '--preflight')
    sample = Path(state['conf_q']) / '01_data/tables/semantic_audit_to_review.csv'
    print('\nReview the actual rows and fill the audit fields before proceeding:\n' + str(sample))
    print('Leave the next entry empty to return to the menu; select this action again after reviewing.')
    audit_text = input('Absolute path to your REVIEWED semantic-audit CSV: ').strip()
    if not audit_text:
        print('Confirmation initialized but blocked from training pending human audit.')
        return
    audit = Path(audit_text).expanduser().resolve()
    if not audit.is_file():
        raise FileNotFoundError(audit)
    for model, key in [('qwen', 'conf_q'), ('llama', 'conf_l')]:
        r.pkg(Path(state[key]), model, '--audit-csv', str(audit))
    print('Both confirmation projects prepared. Choose the first-seed confirmation action next.')


def status() -> None:
    if not (WORK / 'active.json').exists():
        print('No study initialized yet.'); return
    workspace, state = load_state()
    print('Workspace:', workspace)
    for model, key in [('qwen', 'cal_q'), ('llama', 'cal_l'), ('qwen', 'conf_q'), ('llama', 'conf_l')]:
        project = Path(state[key]); print(f'\n{key}: {project}')
        for phase in ('pilot', 'full'):
            gate = project / model / '04_behavior' / phase / 'dev/artifacts' / f'gate_{phase}.json'
            if gate.exists():
                print(f'  {phase} gate passed: {read(gate).get("passed")}')
        if (project / 'billing.json').exists():
            d = read(project / 'billing.json'); spent = 0.0
            if d.get('active', True):
                spent += max(0.0, time.time() - datetime.fromisoformat(d['started_utc']).timestamp()) / 3600 * d['hourly_usd']
            for seg in d.get('segments', []):
                spent += (datetime.fromisoformat(seg['ended_utc']) - datetime.fromisoformat(seg['started_utc'])).total_seconds() / 3600 * seg['hourly_usd']
            print(f'  Estimated TOTAL instance spend: ${spent:.2f}; duplicated ledger, do not sum across projects.')
    print('\nRaw logs:', workspace / 'launch_logs')
    print('Detach is not shutdown. Provider billing continues until you stop the instance.')


def do_action(action: str) -> None:
    verify_bundle()
    if action == 'status':
        status(); return
    with workflow_lock():
        r = Runner(action)
        if action == 'install':
            r.ensure_install(); return
        if action == 'calibration':
            calibration(r); return
        if action == 'freeze':
            freeze(r); return
        workspace, state = load_state()
        r.ensure_install(); r.authenticate(); r.hardware()
        if action == 'caps':
            require_calibration_completed(state)
            caps = ask('Measured cap candidates, comma separated; example 256,512,1024')
            for model, key in [('qwen', 'cal_q'), ('llama', 'cal_l')]:
                r.pkg(Path(state[key]), model, '--calibration-caps', caps)
            return
        # These remain separate intentional launches, not automatic consumption of all optional sweeps.
        stage = 'pilot' if action == 'confirm-pilot' else action
        if stage not in ('pilot', 'confirm', 'mechanisms', 'features', 'minimal', 'trajectory'):
            raise ValueError(action)
        r.parallel(workspace, state, stage, confirmation=True)


MENU = '''
GAMINGMECH — one model process per H100; no DDP
  1  Install dependencies and run host tests (no 8B training)
  2  Start/resume calibration: common data, BOTH GPU profiles, BOTH one-seed pilots
  3  Status, paths, gates and estimated instance spend
  4  Calibration token-cap sweep (optional; additional generation cost)
  5  Select final protocol, initialize both confirmation projects, register human audit
  6  First confirmatory seed on Qwen and Llama in parallel
  7  Remaining confirmation seeds + behavior/extraction in parallel
  8  Mechanisms: steering, patching, dissection and detection in parallel
  9  Sparse features in parallel (optional)
 10  Restricted adapters in parallel (optional additional training)
 11  Checkpoint trajectories in parallel (optional)
  0  Exit console; DOES NOT stop instance billing
Detach safely: Ctrl-B, then D. Ctrl-C requests cancellation, not detachment.
'''


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=ACTIONS, help='Run one action and exit; default is interactive menu')
    a = p.parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit('Python >=3.10 is required.')
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    if a.action:
        do_action(a.action); return
    while True:
        print(MENU)
        try:
            choice = input('Action: ').strip()
            if choice == '0':
                print('No automatic shutdown: stop the rented instance yourself when finished.'); return
            n = int(choice)
            if not 1 <= n <= len(ACTIONS):
                raise ValueError('Choose one of the displayed actions.')
            do_action(ACTIONS[n - 1])
            print('\nAction complete. No next stage starts automatically.')
        except (KeyboardInterrupt, EOFError):
            print('\nInterrupted. Completed checkpoints/logs remain. No automatic restart; billing continues.')
            return
        except Exception as ex:
            print(f'\nSTOPPED: {type(ex).__name__}: {ex}\nInspect the printed logs. Nothing was silently marked successful.')
            print('Fix the operational cause, then repeat the same action for package-level resume. Do not edit locks.')


if __name__ == '__main__':
    main()
