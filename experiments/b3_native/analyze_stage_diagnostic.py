"""Offline S2 parity prediction and S3 same-state/sensitivity analysis."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(stage,old_native):
    flat=[dict(time=record['time'],**s) for record in stage['stage_comparisons'] for s in record['stages']]
    failed=[dict(time=s['time'],stage=s['stage'],worst_active_relative_difference=s['worst_active_relative_difference']) for s in flat if not s['active_slots_pass_relative_1e_minus12']]
    late=[s for s in stage['samples'] if 5<=s['time']<=18 and 'max_slot_difference' in s]
    current={s['time']:s['baseline']['error'] for s in stage['samples'] if 'baseline' in s}
    common=[s for s in old_native['rows'] if s['time'] in current]
    last=stage['samples'][-1]
    return dict(status=stage['status'],complete_sampled_stage_set=stage['complete_sampled_stage_set'],
        sampled_stages=len(flat),failed_stages=failed,all_stages_pass_registered_relative_target=not failed and stage['complete_sampled_stage_set'],
        largest_absolute_rhs_difference=max(s['global_max_absolute_difference'] for s in flat),
        baseline_saved_error_max_difference=max(abs(current[s['time']]-s['error']) for s in common),
        perturbation_log_growth_t5_to_t18=float(np.polyfit([s['time'] for s in late],np.log([s['max_slot_difference'] for s in late]),1)[0]),
        last_time=last['time'],last_max_slot_difference=last.get('max_slot_difference'),last_amplification=last.get('amplification_vs_initial_max'),
        last_analytic_error={name:last[name]['error'] for name in ('baseline','one_ulp') if name in last},
        historical_discrepancy_proven_solely_roundoff=False,external_independent_reproduction=False)


def plot(stage,s2,nulls,python_saved,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    rows=stage['samples']
    for name in ('baseline','one_ulp'):
        r=[s for s in rows if name in s and s['time']>0]
        axes[0,0].semilogy([s['time'] for s in r],[s[name]['error'] for s in r],label=name)
    axes[0,0].semilogy(*np.array(python_saved['samples']).T,'--',label='saved Python')
    axes[0,0].set(title='S3: A=0.1 analytic errors',ylabel='max field error / A')
    r=[s for s in rows if 'max_slot_difference' in s]
    axes[0,1].semilogy([s['time'] for s in r],[s['max_slot_difference'] for s in r])
    axes[0,1].set(title='S3: amplification of one initial ULP',ylabel='max slot difference')
    records=stage['stage_comparisons']
    axes[1,0].semilogy([r['time'] for r in records],[max(s['global_max_absolute_difference'] for s in r['stages']) for r in records],marker='o')
    axes[1,0].set(title='S3: same-state RHS absolute differences',ylabel='max absolute difference')
    rows=s2['rows']; t=np.array([r['time'] for r in rows]); prediction=nulls['seed42_prediction']
    axes[1,1].plot(t,[r['rms']['lapse'] for r in rows],label='measured native RMS')
    axes[1,1].plot(t,t*prediction['predicted_rms_slope'],'--',label='8 parity modes, no fitted parameter')
    axes[1,1].plot(t,t*prediction['constant_only_slope'],':',label='constant mode only')
    axes[1,1].set(title='S2-c: parity-null-mode prediction',ylabel='lapse RMS')
    for ax in axes.flat:
        ax.set_xlabel('crossing time'); ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8); axes[1,1].legend(fontsize=8)
    fig.savefig(destination,dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage_directory',type=Path); ap.add_argument('--null-result',type=Path,required=True)
    ap.add_argument('--plot',action='store_true'); args=ap.parse_args()
    stage=json.loads((args.stage_directory/'result.json').read_text())
    old_native=json.loads(Path('docs/long-diagnostics/b1-a01-native.json').read_text())
    result=analyze(stage,old_native)
    (args.stage_directory/'analysis.json').write_text(json.dumps(result,indent=1,allow_nan=False)+'\n',encoding='utf-8')
    if args.plot:
        plot(stage,json.loads(Path('docs/long-diagnostics/s2-fd4-n8-t500.json').read_text()),json.loads(args.null_result.read_text()),
             json.loads(Path('docs/b1-gauge-wave/long-a0.1.json').read_text()),args.stage_directory/'stage-and-null-modes.png')
    print(json.dumps(result,indent=1))


if __name__=='__main__': main()
