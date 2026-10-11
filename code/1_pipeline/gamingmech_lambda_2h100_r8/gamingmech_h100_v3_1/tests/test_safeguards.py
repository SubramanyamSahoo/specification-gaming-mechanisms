from pathlib import Path
import dataclasses
import pytest
import torch
from gamingmech.config import Config, validate_config, apply_overrides
from gamingmech.runtime import activate, initialize, load_project, partition, cached_json
from gamingmech.common import save_json, load_json


def test_gate_failure_aborts_actual_runner(cfg, monkeypatch):
    from gamingmech import run as runner
    from gamingmech import runtime
    called=[]
    monkeypatch.setattr(runtime,'data_identity',lambda c:'fixture-data')
    def fail(c,cache):
        called.append('behavior')
        raise RuntimeError('BEHAVIOR GATE FAILED')
    monkeypatch.setitem(runner.RUNNERS,'behavior',fail)
    monkeypatch.setitem(runner.RUNNERS,'activations',lambda c,cache:called.append('activations'))
    try:
        with pytest.raises(RuntimeError,match='GATE FAILED'):runner.run(cfg,['behavior','activations'])
        assert called == ['behavior']
        assert not load_json(cfg.exp_dir/cfg.model_key/'receipts'/'pilot'/'behavior.json')['passed']
    finally:activate(None)


def test_operation_cache_corruption_fails(cfg):
    called=[]
    try:
        a=cached_json(cfg,'99_fixture',{'v':1},lambda:called.append(1) or {'x':1})
        b=cached_json(cfg,'99_fixture',{'v':1},lambda:called.append(2) or {'x':2})
        assert a==b=={'x':1} and called==[1]
        out=next(p for p in (cfg.stage_dir('99_fixture')/'artifacts'/'operation_cache').glob('*.json') if not p.name.endswith('.complete.json'))
        out.write_text('{"x":3}')
        with pytest.raises(RuntimeError,match='Corrupt'):cached_json(cfg,'99_fixture',{'v':1},lambda:None)
    finally:activate(None)


def test_partition_restores_after_exception(cfg):
    old=cfg.runtime.partition
    with pytest.raises(ValueError),partition(cfg,'dev'):
        assert cfg.runtime.partition=='dev'
        raise ValueError('test')
    assert cfg.runtime.partition==old


def test_config_bad_split_rejected():
    cfg=Config();cfg.runtime.fit_fraction=.8;cfg.runtime.dev_fraction=.4
    with pytest.raises(ValueError,match='disjoint'):validate_config(cfg)


def test_empty_tuple_cli_rates_are_numeric():
    cfg=apply_overrides(Config(),['plan.contamination_rates=0.101,0.104'])
    assert cfg.plan.contamination_rates==(.101,.104)
    assert len(set(c.name for c in cfg.conditions()))==len(cfg.conditions())


def test_config_roundtrip_and_phase_identity(tmp_path):
    cfg=Config();cfg.runtime.device='cpu';cfg.runtime.dtype='float32';cfg.runtime.project=str(tmp_path/'new')
    try:
        initialize(cfg)
        q=load_project(str(cfg.exp_dir),'qwen','pilot');l=load_project(str(cfg.exp_dir),'llama','full')
        assert q.fingerprint()==l.fingerprint()==cfg.fingerprint()
        assert len(q.conditions())==4 and len(l.conditions())==12
        assert q.data_dir==l.data_dir
    finally:activate(None)


def test_new_training_logs_plot(cfg):
    from gamingmech.common import save_csv
    from gamingmech.plots import plot_training
    cfg.runtime.fig_formats=('png',)
    path=cfg.stage_dir('03_train')/'artifacts'/'control_f000_s11'
    save_csv([{'step':1,'val_loss':.6,'val_accuracy':.5,'val_margin':.1}],path/'validation_1.csv')
    save_csv([{'step':2,'val_loss':.5,'val_accuracy':.6,'val_margin':.2}],path/'validation_2.csv')
    plot_training(cfg)
    assert (cfg.stage_dir('03_train')/'figures'/'validation_curves.png').is_file()


def test_report_indexes_only_existing_files(cfg):
    from gamingmech.report import build_report
    from gamingmech.common import save_csv
    save_csv([{'condition':'fixture','metric':1.}],cfg.stage_dir('04_behavior')/'tables'/'fixture_means.csv')
    path=build_report(cfg)
    assert path.is_file() and 'fixture_means.csv' in path.read_text()
    assert list(path.parent.glob('*.tex'))


def test_balanced_post_semantic_selection(cfg):
    from gamingmech.data import exact_bucket_counts,select_validated_roster
    from gamingmech.config import FAMILIES
    cfg.data.n_train_pairs=200
    candidates=[]
    for (fam,bucket) in exact_bucket_counts(cfg,150):
        for i in range(100):
            candidates.append({'item_id':f'{fam}/{bucket}/{i}','group_id':f'{fam}/{bucket}/{i}',
                               'family':fam,'bucket':bucket,'split':'train'})
    a=select_validated_roster(cfg,candidates)
    b=select_validated_roster(cfg,list(reversed(candidates)))
    assert a==b
    assert sum(x['split']=='train' for x in a)==150
    assert sum(x['split']=='val' for x in a)==8
    assert not ({x['group_id'] for x in a if x['split']=='train'} & {x['group_id'] for x in a if x['split']=='val'})
    assert sum(exact_bucket_counts(cfg,157).values())==157


def test_segmented_clock_preserves_cost_and_blocks_paused_work(cfg):
    from gamingmech.runtime import billing_start,billing_pause,billing_resume,budget,tick
    from datetime import datetime,timezone,timedelta
    now=datetime.now(timezone.utc)
    billing_start(cfg,2.,(now-timedelta(hours=3)).isoformat())
    billing_pause(cfg,(now-timedelta(hours=2)).isoformat())
    assert budget(cfg)['spent_usd']==pytest.approx(2.,abs=.001)
    activate(cfg)
    try:
        with pytest.raises(RuntimeError,match='paused'):tick()
        billing_resume(cfg,4.,(now-timedelta(hours=1)).isoformat())
        assert budget(cfg)['spent_usd']==pytest.approx(6.,abs=.01)
    finally:activate(None)
