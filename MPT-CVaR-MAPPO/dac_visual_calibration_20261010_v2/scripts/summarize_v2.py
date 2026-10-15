"""Post-measurement report only; never fits models or changes gates/timings."""
from __future__ import annotations
import argparse
from v2_common import BASE, PROFILES, read, csv_write, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--complete',action='store_true');opts=p.parse_args()
    from runner import audit, check
    check()
    reports=[read(BASE/'analysis/VALIDATION_REPORT.json')]
    _, confirmation=audit(False)
    rows=confirmation
    if opts.complete:
        if read(BASE/'protocol/STATE.json')['status']!='complete':
            raise RuntimeError('Final state not complete')
        _,test=audit(True);rows+=test
        reports.append(read(BASE/'analysis/TEST_REPORT.json'))
    for report in reports:
        if report['plan_sha256']!=sha(BASE/'protocol/PLAN.json') or report['model_sha256']!=sha(BASE/'model/CONDITIONAL_MODEL.json'):
            raise RuntimeError('Report fingerprint mismatch')
    csv_write(BASE/'analysis/v2_service_requests.csv',rows,exclusive=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(len(reports),2,figsize=(9,3.4*len(reports)),squeeze=False)
    for r,report in enumerate(reports):
        for c,profile in enumerate(PROFILES):
            ax=axes[r,c];summary=report['summary'][profile]
            x=np.arange(3);metrics=['mean','p95','p99']
            ax.bar(x-.18,[summary['predicted_ms'][k] for k in metrics],.36,label='Frozen conditional prediction',color='#0072B2')
            ax.bar(x+.18,[summary['measured_ms'][k] for k in metrics],.36,label='Measured full service',color='#D55E00')
            ax.set_xticks(x,['Mean','P95','P99']);ax.set_ylabel('Service time (ms)')
            ax.set_title(f"{report['role'].capitalize()}: {profile}")
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
            ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    folder=BASE/'figures';folder.mkdir(exist_ok=True)
    if any((folder/f'fig_visual_service_v2.{ext}').exists() for ext in ('png','pdf')):
        raise FileExistsError('Refuse to overwrite prior V2 report figure')
    fig.savefig(folder/'fig_visual_service_v2.png',dpi=180)
    fig.savefig(folder/'fig_visual_service_v2.pdf')
    plt.close(fig)
    print('V2_POSTPROCESS_OK',len(rows),flush=True)

if __name__=='__main__':main()
