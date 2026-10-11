"""Plots read real saved CSVs only. No hypothetical or hardcoded scientific results."""
from __future__ import annotations
import csv
import html
import math
import re
from pathlib import Path


def rows(path):
    if not path.is_file(): return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def slug(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_')


def effect_plot(data, path, title, label_key='condition', value='diff_mean', low='diff_lo', high='diff_hi'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good = [r for r in data if value in r and math.isfinite(float(r[value]))]
    if not good: return False
    labels = [str(r[label_key]) for r in good]
    y = [float(r[value]) for r in good]
    fig, axis = plt.subplots(figsize=(max(7., min(15., .48*len(good))), 4.8))
    if all(low in r and high in r and math.isfinite(float(r[low])) and math.isfinite(float(r[high])) for r in good):
        axis.vlines(range(len(good)), [float(r[low]) for r in good], [float(r[high]) for r in good])
        axis.plot(range(len(good)), y, 'o')
    else:
        axis.plot(range(len(good)), y, 'o')
    axis.axhline(0, linestyle='--', linewidth=.8)
    axis.set_xticks(range(len(good)), labels, rotation=45, ha='right')
    axis.set_title(title); axis.set_ylabel(value); axis.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches='tight'); fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return True


def plot_experiment(project, stage):
    directory = project / 'RESULTS' / stage
    figures = directory / 'figures'; figures.mkdir(parents=True, exist_ok=True)
    made = []
    if stage == 'attention':
        data = rows(directory / 'attention_effects.csv')
        for family in sorted({r['family'] for r in data}):
            for measure in ('to_claim', 'to_user_message', 'to_first_token'):
                sub = [dict(r, label=f"{r['group']} L{r['layer']}H{r['head']}") for r in data
                       if r['family']==family and r['measure']==measure and r['reference']=='control']
                name = slug(f'{family}_{measure}') + '.png'
                if effect_plot(sub, figures/name, f'{family}: attention shift versus matched control', 'label'): made.append(name)
    elif stage == 'joint_heads':
        data = rows(directory/'joint_head_patching.csv')
        for family in sorted({r['family'] for r in data}):
            sub = [dict(r, label=f"seed {r['seed']} / {r['group']}") for r in data if r['family']==family]
            name = slug(family) + '.png'
            if effect_plot(sub, figures/name, f'{family}: simultaneous donor-head intervention', 'label'): made.append(name)
    elif stage in ('public','steering','dose'):
        paths = {'public':['public_effects.csv'],
                 'steering':['generation_effects_vs_unsteered.csv','logit_effects_vs_unsteered.csv'],
                 'dose':['generation_effects.csv','logit_effects.csv']}[stage]
        for name in paths:
            data = rows(directory/name)
            if stage in ('public','dose'): data = [r for r in data if r.get('reference')=='control']
            for metric in sorted({r['metric'] for r in data}):
                sub=[r for r in data if r['metric']==metric]
                outfile=slug(Path(name).stem+'_'+metric)+'.png'
                title=f'{stage}: {metric} (paired difference; fixed-run intervals)'
                if effect_plot(sub, figures/outfile, title): made.append(outfile)
        if stage=='dose':
            data=rows(directory/'monitor_dose.csv')
            if effect_plot(data, figures/'monitor_shift.png', 'Within-seed monitor shift; not a calibrated alarm',
                           value='monitor_shift_vs_control', low='lo', high='hi'): made.append('monitor_shift.png')
    with (directory/'figure_index.csv').open('w', newline='', encoding='utf-8') as out:
        writer=csv.DictWriter(out, fieldnames=['png','pdf','source_directory']); writer.writeheader()
        for name in made: writer.writerow({'png':'figures/'+name,'pdf':'figures/'+str(Path(name).with_suffix('.pdf')),
                                          'source_directory':str(directory)})


def write_index(project):
    sections=[]
    for stage in ('attention','seeds','public','joint_heads','steering','dose'):
        directory=project/'RESULTS'/stage
        links=[]
        if directory.exists():
            for p in sorted(directory.rglob('*')):
                if p.is_file() and p.suffix in ('.csv','.png','.pdf','.jsonl') and 'judge_cache' not in p.parts:
                    rel=str(p.relative_to(project)); links.append(f'<li><a href="{html.escape(rel,quote=True)}">{html.escape(rel)}</a></li>')
        sections.append('<h2>'+stage+'</h2><ul>'+''.join(links)+'</ul>')
    # Core plots for the two additional seeds remain in their original stage layout.
    seedlinks=[]
    for p in sorted((project/'llama').rglob('seed*')):
        if p.is_dir() and p.parent.name in ('06_patching','07_dissect'):
            for f in sorted(p.rglob('*.png')):
                rel=str(f.relative_to(project)); seedlinks.append(f'<li><a href="{html.escape(rel,quote=True)}">{html.escape(rel)}</a></li>')
    text='''<!doctype html><meta charset="utf-8"><title>Llama r9 results</title>
<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}li{margin:8px 0}</style>
<h1>Llama r9 follow-up results</h1><p>Execution completion is not scientific confirmation.
These follow-ups reuse the original r8 evaluator. Judge-score means are not human-labeled rates;
legacy switching also includes neither/both/no-answer. Inspect raw responses and controls.</p>'''
    text+=''.join(sections)+'<h2>Additional seed plots</h2><ul>'+''.join(seedlinks)+'</ul>'
    (project/'RESULTS.html').write_text(text,encoding='utf-8')
