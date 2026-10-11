"""Single-GPU, resumable Llama follow-up coordinator; reuses the locked r8 package."""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import getpass
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from r9_support import atomic_json, checked_outputs, fingerprint, load_json, sha256

ROOT = Path(__file__).resolve().parent
STAGES = ('attention', 'seeds', 'public', 'joint_heads', 'steering', 'dose')


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]


def check_bundle():
    manifest = load_json(ROOT / 'SHA256.json')
    for rel, digest in manifest.items():
        path = ROOT / rel
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError('Add-on file checksum mismatch: ' + rel)


def check_core(package):
    expected = load_json(ROOT / 'expected_core.json')
    actual = {p.name: sha256(p) for p in sorted((package / 'gamingmech').glob('*.py'))}
    if expected != actual:
        raise RuntimeError('The installed core is not the checked r8 code. Do not edit locks or reinstall; use the matching r8 directory.')
    sys.path.insert(0, str(package))


@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another process owns this run/project lock: ' + str(path)) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def gpu_uuid(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if not text.startswith('GPU-') or '\n' in text:
        raise RuntimeError('Expected one physical GPU UUID, got an invalid response')
    return text


def require_idle(gpu):
    text = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).strip()
    if text:
        raise RuntimeError(f'GPU {gpu} already has compute processes ({text}). Nothing was killed. Wait for that job.')


def parse_time(text):
    value = datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Billing times must include a timezone')
    return value.timestamp()


def spend_at(billing, timestamp):
    rate = float(billing['hourly_usd'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Invalid hourly billing rate')
    total = 0.0
    for segment in billing.get('segments', []):
        start, end = parse_time(segment['started_utc']), parse_time(segment['ended_utc'])
        sr = float(segment['hourly_usd'])
        if end < start or end > timestamp + 5 or sr <= 0 or not math.isfinite(sr):
            raise ValueError('Invalid historical billing segment')
        total += (end - start) * sr / 3600
    start = parse_time(billing['started_utc'])
    if start > timestamp:
        raise ValueError('Billing start is in the future')
    if billing.get('active', True):
        total += (timestamp - start) * rate / 3600
    return total


def budget_remaining(run, timestamp=None):
    timestamp = time.time() if timestamp is None else timestamp
    billing = load_json(Path(run['source']) / 'billing.json')
    if fingerprint(billing) != run['billing_identity']:
        raise RuntimeError('Source billing ledger changed after extras initialization. Do not reset its clock.')
    if not billing.get('active', True):
        raise RuntimeError('Provider billing clock is marked paused; GPU execution is not authorized')
    spent = spend_at(billing, timestamp)
    total_remaining = run['budget_usd'] - run['reserve_usd'] - spent
    extra = run['settings']['max_additional_usd']
    if extra is not None:
        total_remaining = min(total_remaining, extra - (spent - run['spent_at_initialization']))
    return total_remaining, spent


def source_contract(cfg):
    from gamingmech.runtime import data_identity
    paths = ['protocol.json', 'revisions.json',
             'llama/04_behavior/full/dev/artifacts/gate_full.json',
             'llama/05_activations/full/artifacts/directions.pt',
             'llama/05_activations/full/artifacts/caa_directions.pt',
             'llama/06_patching/full/tables/head_patching_verification.csv',
             'llama/07_dissect/full/artifacts/minimal_adapter_specs.json',
             'llama/08_steering/full/artifacts/development_selections.json']
    for condition in cfg.conditions():
        paths.append(str((cfg.adapter_dir(condition) / 'final' / 'COMPLETE.json').relative_to(cfg.exp_dir)))
    records = {}
    for rel in paths:
        p = cfg.exp_dir / rel
        if not p.is_file():
            raise RuntimeError('Required completed-study artifact is missing: ' + str(p))
        records[rel] = sha256(p)
    return {'files': records, 'data_identity': data_identity(cfg), 'fingerprint': cfg.fingerprint(),
            'seeds': list(cfg.seeds), 'reference_seed': cfg.reference_seed}


def mirror(src, dst):
    """Copy mutable data/metadata; only immutable original weight directories are linked."""
    dst.mkdir(parents=True, exist_ok=True)
    root_files = ('protocol.json', 'revisions.json', 'billing.json', 'billing_origin.json', 'data_origin.json',
                  'environment.json', 'preflight.json', 'architecture_checks.json', 'dataset_schema_checks.json')
    for name in root_files:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    shutil.copytree(src / '01_data', dst / '01_data', symlinks=False, dirs_exist_ok=False,
                    ignore=shutil.ignore_patterns('operation_cache', '__pycache__'))
    train_source = src / 'llama' / '03_train' / 'artifacts'
    train_dest = dst / 'llama' / '03_train' / 'artifacts'; train_dest.mkdir(parents=True)
    links = []
    for item in sorted(train_source.iterdir()):
        if item.is_file() and item.name in ('reference_logprobs.pt', 'retained_synthetic_roster.json'):
            shutil.copy2(item, train_dest / item.name)
        elif item.is_dir() and (item / 'final' / 'COMPLETE.json').exists():
            folder = train_dest / item.name; folder.mkdir()
            for record in item.iterdir():
                if record.is_file() and record.suffix in ('.csv', '.json'):
                    shutil.copy2(record, folder / record.name)
                elif record.is_dir() and (record.name == 'final' or record.name.startswith('checkpoint_')):
                    (folder / record.name).symlink_to(record.resolve(), target_is_directory=True)
                    links.append({'link': str((folder / record.name).relative_to(dst)), 'source': str(record.resolve())})
    for phase in ('full',):
        origin = src / 'llama' / '04_behavior' / phase / 'dev' / 'artifacts'
        target = dst / 'llama' / '04_behavior' / phase / 'dev' / 'artifacts'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('gate.json', 'gate_full.json'):
            if (origin / name).exists(): shutil.copy2(origin / name, target / name)
    atomic_json(dst / 'SOURCE_LINKS.json', links)


def ensure_auth(cfg):
    # Prompt before loading HF libraries that read environment variables on import.
    entered = getpass.getpass('Valid Hugging Face token (hidden; Enter uses your existing token): ').strip()
    candidate = entered or os.environ.get('HF_TOKEN', '')
    if not candidate:
        home = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache' / 'huggingface')))
        path = Path(os.environ.get('HF_TOKEN_PATH', str(home / 'token')))
        candidate = path.read_text().strip() if path.is_file() else ''
    if not candidate or any(c.isspace() for c in candidate):
        raise RuntimeError('No valid-form token supplied. No models were loaded.')
    os.environ['HF_TOKEN'] = candidate
    os.environ[cfg.runtime.hf_token_env] = candidate
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    try:
        HfApi(token=candidate).whoami()
        revisions = load_json(cfg.exp_dir / 'revisions.json')
        for repo in (cfg.spec.hf_id, cfg.eval.judge_hf_id):
            revision = revisions.get('model:' + repo)
            if not revision: raise ValueError('Missing locked model revision')
            get_hf_file_metadata(hf_hub_url(repo, 'config.json', revision=revision), token=candidate, timeout=20)
            print('Model-file access verified:', repo, flush=True)
    except Exception as error:
        status = getattr(getattr(error, 'response', None), 'status_code', 'unknown')
        raise RuntimeError(f'Credential/repository check failed ({type(error).__name__}, HTTP {status}). No GPU model was loaded.') from None


def initialize_run(base, source, settings, contract, cfg, new=False):
    work = ROOT / 'work'; work.mkdir(exist_ok=True)
    pointer = work / 'active.json'
    bundle_digest = sha256(ROOT / 'SHA256.json')
    if pointer.exists() and not new:
        folder = Path(load_json(pointer)['run']); run = load_json(folder / 'run.json')
        if any((run['source'] != str(source), run['base'] != str(base), run['source_contract'] != contract,
                run['settings'] != settings, run['bundle_manifest_sha256'] != bundle_digest)):
            raise RuntimeError('Active extras identity differs from these inputs. Do not reuse its caches. Existing outputs were left unchanged.')
        return folder, run
    folder = work / ('llama_r9_' + stamp()); folder.mkdir()
    if shutil.disk_usage(folder).free < settings['minimum_free_disk_gib'] * 2**30:
        raise RuntimeError('Insufficient free disk for additional checkpoints; original study was not changed')
    project = folder / 'project'
    mirror(source, project)
    billing = load_json(source / 'billing.json')
    run = {'release': settings['release'], 'run_id': folder.name, 'created_utc': now(),
           'base': str(base), 'source': str(source), 'project': str(project), 'settings': settings,
           'source_contract': contract, 'bundle_manifest_sha256': bundle_digest,
           'budget_usd': cfg.runtime.budget_usd, 'reserve_usd': cfg.runtime.reserve_usd,
           'billing_identity': fingerprint(billing), 'spent_at_initialization': spend_at(billing, time.time()),
           'scientific_status': 'post-main-study follow-ups; completion is not hypothesis confirmation',
           'measurement': 'original r8 metrics retained; judge validation and generalization not certified'}
    shutil.copytree(ROOT / 'tests', folder / 'source_snapshot' / 'tests')
    for name in load_json(ROOT / 'SHA256.json'):
        p = ROOT / name
        target = folder / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): shutil.copy2(p, target)
    atomic_json(folder / 'run.json', run)
    atomic_json(pointer, {'run': str(folder)})
    return folder, run


def manifest_outputs(project, stage):
    folders = [project / 'RESULTS' / stage]
    if stage == 'seeds':
        folders += [project / 'llama' / '06_patching', project / 'llama' / '07_dissect']
    result = {}
    for directory in folders:
        for p in sorted(directory.rglob('*')) if directory.exists() else []:
            if p.is_file() and not p.is_symlink() and 'operation_cache' not in p.parts and p.suffix in ('.csv','.json','.jsonl','.png','.pdf','.html'):
                result[str(p.relative_to(project))] = sha256(p)
    if not result:
        raise RuntimeError('Experiment has no recorded outputs: ' + stage)
    return result


def done_receipt(folder, stage):
    path = folder / 'receipts' / (stage + '.json')
    if not path.exists(): return None
    result = load_json(path)
    if result.get('status') != 'completed': return None
    checked_outputs(folder / 'project', result)
    from gamingmech.dpo import verify_complete
    for rel in result.get('new_adapter_finals', []):
        verify_complete(folder / 'project' / rel)
    return result


def stop_child(child, grace):
    if child.poll() is not None: return
    os.killpg(child.pid, signal.SIGINT)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()


def execute_stage(folder, run, stage):
    attempt = folder / 'logs' / (stamp() + '_' + stage); attempt.mkdir(parents=True)
    path = attempt / 'run.log'
    command = [sys.executable, '-u', str(ROOT / 'worker.py'), str(folder), stage]
    state = {'status': 'running', 'stage': stage, 'started_utc': now(), 'log': str(path),
             'supervisor_pid': os.getpid()}
    stop_file = folder / 'STOP_REQUESTED.json'
    with path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        state['worker_pid'] = child.pid; atomic_json(folder / 'status.json', state)
        def reader():
            for line in child.stdout:
                log.write(line); log.flush()
                sys.stdout.write(line); sys.stdout.flush()
        thread = threading.Thread(target=reader, daemon=True); thread.start()
        failure = None
        try:
            while child.poll() is None:
                if stop_file.exists():
                    failure = 'Stop requested by the user'; break
                remaining, spent = budget_remaining(run)
                state.update(remaining_work_usd=remaining, estimated_instance_spent_usd=spent, checked_utc=now())
                atomic_json(folder / 'status.json', state)
                if remaining <= 0:
                    failure = 'Instance budget/reserve or extra-spending authorization reached'; break
                time.sleep(run['settings']['budget_poll_seconds'])
        except BaseException as error:
            failure = type(error).__name__ + ': ' + str(error)
        finally:
            if failure: stop_child(child, run['settings']['shutdown_grace_seconds'])
            code = child.wait(); thread.join(timeout=10)
            child.stdout.close()
    atomic_json(attempt / 'exit.json', {'returncode': code, 'stop_reason': failure, 'ended_utc': now()})
    if failure or code:
        raise RuntimeError(f'{stage} stopped: {failure or ("exit " + str(code))}. Log: {path}')
    if not done_receipt(folder, stage):
        raise RuntimeError('Worker exited without a verified completion receipt: ' + stage)


def pack(folder):
    archive = folder / ('llama_r9_results_' + stamp() + '.zip')
    excluded = []
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for base, dirs, names in os.walk(folder, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in ('operation_cache','__pycache__','.pytest_cache')
                             and not (Path(base) / d).is_symlink())
            for name in sorted(names):
                p = Path(base) / name
                if p == archive or p.suffix == '.zip': continue
                if p.is_symlink() or p.suffix not in ('.csv','.json','.jsonl','.png','.pdf','.html','.md','.log','.txt','.sh','.py'):
                    excluded.append(str(p.relative_to(folder))); continue
                # Avoid duplication of pretrained/source datasets and stored tokens.
                if '01_data' in p.relative_to(folder).parts:
                    continue
                z.write(p, arcname=str(p.relative_to(folder)))
        z.writestr('ARCHIVE_SCOPE.json', json.dumps({'raw_jsonl_included': True,
                    'weights_and_tensor_bytes_included': False, 'excluded_binary_or_link_paths': excluded,
                    'source_project_required': load_json(folder / 'run.json')['source'],
                    'scope': 'results export, NOT a full backup of models/activation tensors'}, indent=2))
    atomic_json(folder / 'latest_export.json', {'path': str(archive), 'sha256': sha256(archive)})
    print('RESULTS ZIP:', archive, flush=True)
    return archive


def validate_settings(settings):
    if tuple(settings['stage_order']) != STAGES:
        raise ValueError('Keep the declared six-experiment ordering in this release')
    for key in ('physical_gpu', 'cpu_threads', 'attention_batch_max', 'budget_poll_seconds', 'shutdown_grace_seconds'):
        value = settings[key]
        if not isinstance(value, int) or value < (0 if key == 'physical_gpu' else 1):
            raise ValueError('Invalid setting: ' + key)
    cap = settings['max_additional_usd']
    if cap is not None and (not math.isfinite(float(cap)) or cap <= 0):
        raise ValueError('max_additional_usd must be null or positive')
    rates = settings['dose_flip_rates']
    if len(set(rates)) != len(rates) or any(not 0 < r < 1 for r in rates):
        raise ValueError('New dose rates must be distinct and strictly between zero and one')


def run_all(args):
    check_bundle()
    settings = load_json(ROOT / 'settings.json')
    if args.max_additional_usd is not None:
        settings['max_additional_usd'] = args.max_additional_usd
    elif not args.new_run and (ROOT / 'work' / 'active.json').exists():
        prior = Path(load_json(ROOT / 'work' / 'active.json')['run'])
        settings['max_additional_usd'] = load_json(prior / 'run.json')['settings']['max_additional_usd']
    validate_settings(settings)
    base = Path(args.base or os.environ.get('R8_ROOT', str(Path.home() / 'gamingmech_lambda_2h100_r8'))).expanduser().resolve()
    source = Path(args.source or os.environ.get('LLAMA_SOURCE', str(base / 'work' / 'v3' / 'confirmation_llama'))).expanduser().resolve()
    package = base / 'gamingmech_h100_v3_1'
    check_core(package)
    with locked(base / 'work' / 'llama_r9_launch.lock'), locked(source / '.execution.lock'):
        from gamingmech.runtime import load_project, require_gate, tick
        cfg = load_project(str(source), 'llama', 'full')
        require_gate(cfg); tick()
        if cfg.spec.hf_id != 'meta-llama/Llama-3.1-8B-Instruct':
            raise RuntimeError('This add-on is specifically for the completed Llama-3.1-8B-Instruct study')
        contract = source_contract(cfg)
        device = gpu_uuid(settings['physical_gpu']); require_idle(device)
        os.environ.update(CUDA_VISIBLE_DEVICES=device, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                          OMP_NUM_THREADS=str(settings['cpu_threads']), MKL_NUM_THREADS=str(settings['cpu_threads']),
                          TOKENIZERS_PARALLELISM='false', USE_TORCH='1', USE_TF='0', PYTHONUNBUFFERED='1',
                          MPLBACKEND='Agg', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                          R9_ATTENTION_BATCH_MAX=str(settings['attention_batch_max']),
                          R9_DOSE_RATES=json.dumps(settings['dose_flip_rates']))
        ensure_auth(cfg)
        folder, run = initialize_run(base, source, settings, contract, cfg, args.new_run)
        pending = [stage for stage in STAGES if not done_receipt(folder, stage)]
        if pending:
            from gamingmech.behavior import _public_file
            mirror_cfg = load_project(run['project'], 'llama', 'full')
            public_hashes = {}
            for filename in ('are_you_sure.jsonl', 'answer.jsonl'):
                records = _public_file(mirror_cfg, filename)
                if not records or any(not all(k in row for k in ('prompt', 'base', 'metadata')) for row in records):
                    raise RuntimeError('Invalid public benchmark input schema: ' + filename)
                public_hashes[filename] = sha256(mirror_cfg.data_dir / 'artifacts' / 'public' / filename)
            pub_identity = folder / 'public_sources.json'
            if pub_identity.exists() and load_json(pub_identity) != public_hashes:
                raise RuntimeError('Public benchmark bytes changed; do not reuse completed caches')
            atomic_json(pub_identity, public_hashes)
            versions = {}
            for dist in ('torch','transformers','peft','huggingface_hub','numpy'):
                versions[dist] = importlib.metadata.version(dist)
            env_path = folder / 'library_versions.json'
            if env_path.exists() and load_json(env_path) != versions:
                raise RuntimeError('Library versions changed since this extras workspace was initialized')
            atomic_json(env_path, versions)
        print('EXTRAS WORKSPACE:', folder, flush=True)
        print('PHYSICAL GPU:', settings['physical_gpu'], device, flush=True)
        print('PENDING:', ', '.join(pending) or 'none', flush=True)
        remaining, spent = budget_remaining(run)
        print(f'Original-instance accounting: spent ${spent:.2f}; work allowance remaining ${remaining:.2f}', flush=True)
        print('All six extras are requested. seeds adds up to four restricted adapters; dose adds three sycophancy adapters.', flush=True)
        print('No calibration, original full-adapter training, Qwen work or failed SAE retry is scheduled.', flush=True)
        # A repeated command explicitly requests resume after a prior stop.
        stop_file = folder / 'STOP_REQUESTED.json'
        if stop_file.exists():
            stop_file.rename(folder / ('STOP_REQUESTED_' + stamp() + '.json'))
        try:
            for stage in pending:
                if (folder / 'STOP_REQUESTED.json').exists():
                    raise RuntimeError('Stop requested before the next experiment')
                if stage == 'joint_heads' and not done_receipt(folder, 'seeds'):
                    raise RuntimeError('Joint-head replication needs the completed seed subexperiments')
                require_idle(device)
                remaining, _ = budget_remaining(run)
                if remaining <= 0: raise RuntimeError('No authorized compute allowance remains')
                if shutil.disk_usage(folder).free < settings['minimum_free_disk_gib'] * 2**30:
                    raise RuntimeError('Free disk fell below the configured checkpoint reserve')
                execute_stage(folder, run, stage)
            atomic_json(folder / 'status.json', {'status': 'completed', 'ended_utc': now(),
                        'stages': list(STAGES), 'meaning': 'execution completed; not a hypothesis pass'})
        except BaseException as error:
            atomic_json(folder / 'status.json', {'status': 'stopped', 'ended_utc': now(), 'error': str(error)})
            with contextlib.suppress(Exception): pack(folder)
            raise
        pack(folder)
        print('ALL SIX LLAMA EXTRAS COMPLETED. Provider billing continues until you terminate the instance.', flush=True)


def active_folder():
    pointer = ROOT / 'work' / 'active.json'
    if not pointer.exists(): raise RuntimeError('No extras workspace has been initialized yet')
    return Path(load_json(pointer)['run'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run','status','stop','pack'))
    parser.add_argument('--base'); parser.add_argument('--source')
    parser.add_argument('--max-additional-usd', type=float, help='Optional additional spending authorization; never overrides the original total ceiling')
    parser.add_argument('--new-run', action='store_true', help='Explicitly start a separate output workspace, not a resume')
    args = parser.parse_args()
    try:
        if args.command == 'run':
            if not os.environ.get('TMUX'):
                raise RuntimeError('Run START_ALL.sh so work is attached to tmux, not an unprotected SSH process')
            run_all(args)
        else:
            folder = active_folder()
            if args.command == 'pack': pack(folder)
            elif args.command == 'stop':
                atomic_json(folder / 'STOP_REQUESTED.json', {'requested_utc': now()})
                print('Stop requested for THIS extras supervisor only. Wait for status=stopped. No Qwen signal sent.')
            else:
                print('WORKSPACE:', folder)
                if (folder / 'status.json').exists():
                    status = load_json(folder / 'status.json'); print(json.dumps(status, indent=2))
                    log = Path(status.get('log', '/nonexistent'))
                    if log.is_file():
                        print('\nLATEST LOG:'); print(''.join(log.read_text(errors='replace').splitlines(True)[-12:]))
                for stage in STAGES:
                    p = folder / 'receipts' / (stage + '.json')
                    print(stage + ':', load_json(p).get('status') if p.exists() else 'pending')
                if (folder / 'latest_export.json').exists(): print('EXPORT:', load_json(folder / 'latest_export.json')['path'])
    except (Exception, KeyboardInterrupt) as error:
        print('STOPPED:', str(error), file=sys.stderr)
        print('No later experiment was started and no failed scientific gate was changed.', file=sys.stderr)
        raise SystemExit(1)

if __name__ == '__main__': main()
