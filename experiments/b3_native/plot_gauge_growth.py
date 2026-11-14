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
    dissipation = (data["settings"]["points"] == [32]
                   and data["settings"].get("dissipation_sigmas") == [0., .02, .1])
    refinement = (data["settings"]["points"] == [32,16,64]
                  and data["settings"].get("dissipation_sigmas", [0.]) == [0.])
    dissipative_refinement = (data["settings"]["points"] == [16,32,64]
                              and data["settings"].get("fixed_dissipation_sigma") == .1)
    if not (dissipation or refinement or dissipative_refinement) or data["scientific_acceptance_established"]:
        raise ValueError("registered diagnostic report required")
    out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout="constrained")
    colors={16:"#b45309",32:"#2563eb",64:"#047857"}
    if dissipation:
        colors={0.:"#b45309", .02:"#2563eb", .1:"#047857"}
    for row in sorted(data["trajectories"], key=lambda r:r["dissipation_sigma"] if dissipation else r["nx"]):
        nx=row["nx"]
        key=row["dissipation_sigma"] if dissipation else nx
        label=f"sigma={key:g}, nx={nx}" if dissipation else f"nx={nx}, dt=1/{4*nx}"
        if dissipative_refinement:
            label+=", sigma=.1"
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
            axis.semilogy(times[selected],values[selected],".-",color=colors[key],label=label)
            if "failed_interval" in row.get("failure",{}):
                lo,hi=row["failure"]["failed_interval"]
                axis.axvspan(lo,hi,color=colors[key],alpha=.25)
                axis.axvline(lo,color=colors[key],linestyle=":",linewidth=1.)
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
    title = ("Gauge wave A=0.1: x-only dissipation suppresses grid-scale growth; errors remain"
             if dissipation else "Gauge wave A=0.1: early accuracy does not establish long-time stability")
    if dissipative_refinement:
        title="Gauge wave A=0.1: joint space/time refinement at fixed x-dissipation sigma=.1"
    fig.suptitle(title,fontsize=12)
    outcome = ("Baseline fails; sigma=.02/.1 reach T24 only. No stability qualification."
               if dissipation else "All grids fail before T=24.")
    if dissipative_refinement:
        outcome="; ".join(f"nx{r['nx']}: {r['status']}" for r in data["trajectories"])+". No stability qualification."
    fig.supxlabel("Integer-time samples only; dotted lines: last-valid times. "+outcome+"\n"
                    +f"Native additive-slot RK4; FD4; source {data['source_commit'][:8]}",fontsize=9)
    stem="dissipation" if dissipation else "refinement"
    if dissipative_refinement:
        stem="dissipation-refinement"
    fig.savefig(out/f"{stem}.png",dpi=160)
    fig.savefig(out/f"{stem}.svg")
    svg=out/f"{stem}.svg"
    svg.write_bytes(("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n").encode())
    plt.close(fig)
    return out/f"{stem}.png"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    print(plot(args.report,args.out))
