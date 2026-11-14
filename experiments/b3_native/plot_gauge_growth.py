"""Plot only measured S3 refinement samples and retained failure intervals."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def plot(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report,out=Path(report),Path(out)
    data=json.loads(report.read_text(encoding="utf-8"))
    if data["settings"]["points"] != [32,16,64] or data["scientific_acceptance_established"]:
        raise ValueError("registered diagnostic report required")
    out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained")
    colors={16:"#b45309",32:"#2563eb",64:"#047857"}
    for row in sorted(data["trajectories"],key=lambda r:r["nx"]):
        nx=row["nx"]
        times=np.array([s["time"] for s in row["samples"]])
        series=[
            [max(v["l_inf_over_amplitude"] for v in s["fields"].values()) for s in row["samples"]],
            [s["constraints"]["hamiltonian"]["difference"]["l_inf"] for s in row["samples"]],
            [np.sqrt(s["fields"]["h"]["error_spectrum"]["x_power"][-1]) for s in row["samples"]],
            [s["auxiliary_consistency"]["metric_gradient"]["difference"]["l_inf"] for s in row["samples"]],
        ]
        for axis,values in zip(axes.flat,series):
            values=np.asarray(values)
            selected=values>0
            axis.semilogy(times[selected],values[selected],".-",color=colors[nx],label=f"nx={nx}, dt=1/{4*nx}")
            if "failed_interval" in row.get("failure",{}):
                lo,hi=row["failure"]["failed_interval"]
                axis.axvspan(lo,hi,color=colors[nx],alpha=.25)
                axis.axvline(lo,color=colors[nx],linestyle=":",linewidth=1.)
    titles=["Maximum base-field error / A", "ADM Hamiltonian difference from analytic slice",
            "Nyquist RMS of conformal-metric error", "Metric-gradient auxiliary inconsistency"]
    for axis,title in zip(axes.flat,titles):
        axis.set_title(title,fontsize=10)
        axis.set_xlabel("Time (domain crossing units)")
        axis.set_xlim(0,24)
        axis.grid(True,alpha=.2)
    axes[0,0].legend(fontsize=8,loc="upper left")
    axes[0,0].set_ylabel("Max norm / amplitude")
    axes[0,1].set_ylabel("L_inf (evolved - analytic discrete)")
    axes[1,0].set_ylabel("Component-averaged RMS")
    axes[1,1].set_ylabel("L_inf (evolved D - rebuilt D)")
    fig.suptitle("Gauge wave A=0.1: early accuracy does not establish long-time stability",fontsize=12)
    fig.supxlabel("Integer-time samples only; dotted lines: last-valid times. All grids fail before T=24.\n"
                    +f"Native additive-slot RK4; FD4; source {data['source_commit'][:8]}",fontsize=9)
    fig.savefig(out/"refinement.png",dpi=160)
    fig.savefig(out/"refinement.svg")
    plt.close(fig)
    return out/"refinement.png"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(plot(args.report,args.out))
