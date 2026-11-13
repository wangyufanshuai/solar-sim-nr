"""Summarize and plot saved native diagnostics without rerunning evolution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.s2_periodic_evolution.robust_v2 import total_norm


def summarize(s2, b1, reference):
    rows = s2["rows"]
    t = np.array([r["time"] for r in rows])
    n = np.array([total_norm(r["rms"]) for r in rows])
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.all(np.isfinite(n)):
        raise ValueError("S2 samples must be finite and strictly ordered")
    common = {r["time"]: r for r in reference["rows"]}
    differences = [abs(r["rms"][k]-v)/abs(v) for r in rows if r["time"] in common
                   for k,v in common[r["time"]]["rms"].items() if abs(v) > 1e-20]
    result = dict(s2=dict(status=s2["status"], target=s2["target"],
        target_reached=rows[-1]["time"] >= s2["target"], last_sample_time=float(t[-1]),
        last_integrated_time=s2["last_completed_time"], seconds=s2["seconds"], peak_rss_bytes=s2["peak_rss_bytes"],
        registered_windows=s2["windows"], growth_from_t0=float(n[-1]/n[0]),
        max_common_rms_relative_difference_vs_python=max(differences),
        hamiltonian_growth_from_t1=rows[-1]["constraints"]["hamiltonian_l2"]/rows[1]["constraints"]["hamiltonian_l2"],
        boundedness_proven=False))
    if t[-1] > 200:
        late = t >= 200
        result["s2"]["available_window_posthoc"] = dict(start=200, end=float(t[-1]),
            log_slope=float(np.polyfit(t[late], np.log(n[late]), 1)[0]),
            norm_ratio=float(n[-1]/n[t==200][0]))
    late = t >= 100
    if sum(late) >= 2:
        result["s2"]["lapse_rms_linear_slope_t100_to_last"] = float(np.polyfit(t[late], [r["rms"]["lapse"] for r in rows if r["time"] >= 100], 1)[0])
    comparisons = [r for r in b1["rows"] if "relative_difference" in r]
    first = next((r["time"] for r in comparisons if r["relative_difference"] > .01), None)
    result["b1"] = dict(status=b1["status"], target=b1["target"],
        last_integrated_time=b1["last_completed_time"], failed_interval=b1.get("failed_interval"),
        failure=b1.get("failure"), seconds=b1["seconds"], peak_rss_bytes=b1["peak_rss_bytes"],
        common_samples=len(comparisons), relative_difference_target=.01,
        worst_common_relative_difference=max(r["relative_difference"] for r in comparisons),
        first_sample_outside_target=first, common_samples_meet_target=first is None,
        target_reached=b1["last_completed_time"] >= b1["target"], external_independent_reproduction=False)
    return result


def plot(s2, b1, reference, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for data, label, style in ((s2,"native","-"), (reference,"saved Python","--")):
        rows=data["rows"]; t=[r["time"] for r in rows]
        ax[0,0].plot(t, [total_norm(r["rms"]) for r in rows], style, label=label)
        ax[0,1].plot(t, [r["rms"]["lapse"] for r in rows], style, label=label)
        h=[r for r in rows if "constraints" in r]
        ax[1,0].plot([r["time"] for r in h], [r["constraints"]["hamiltonian_l2"] for r in h], style, label=label)
    rows=b1["rows"]; t=[r["time"] for r in rows]
    ax[1,1].semilogy(t,[r["error"] for r in rows],label="native A=0.1")
    c=[r for r in rows if "python_error" in r]
    ax[1,1].semilogy([r["time"] for r in c],[r["python_error"] for r in c],"--",label="saved Python A=0.1")
    ax[1,1].axvspan(*b1["failed_interval"], color="red", alpha=.3, label="native failure interval")
    for a,title,y in zip(ax.flat,("S2-c: perturbation norm", "S2-c: lapse secular growth", "S2-c: Hamiltonian constraint", "S3: large-amplitude gauge wave"), ("L2 perturbation", "lapse RMS", "Hamiltonian L2", "max error / A")):
        a.set(title=title, xlabel="crossing time", ylabel=y)
        a.grid(alpha=.25); a.legend(fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--plot", action="store_true")
    args=ap.parse_args()
    s2=json.loads((args.directory/"s2-fd4-n8-t500.json").read_text())
    b1=json.loads((args.directory/"b1-a01-native.json").read_text())
    ref=json.loads(args.reference.read_text())
    result=summarize(s2,b1,ref)
    (args.directory/"summary.json").write_text(json.dumps(result,indent=1,allow_nan=False)+"\n",encoding="utf-8")
    if args.plot:
        plot(s2,b1,ref,args.directory/"long-diagnostics.png")
    print(json.dumps({case:{k:v for k,v in data.items() if k!="registered_windows"} for case,data in result.items()},indent=1))


if __name__=="__main__":
    main()
