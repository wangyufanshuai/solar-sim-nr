"""Plot B1 convergence and time history from saved run files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(filename):
        return json.loads((args.input / filename).read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, (small, large, history) = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained")
    for ax, amplitude in zip((small, large), (0.01, 0.1)):
        for order, color, marker in ((2, "#0F4D92", "o"), (4, "#B64342", "s")):
            rows = read(f"converge-a{amplitude}-fd{order}.json")["rows"]
            ax.loglog([row["nx"] for row in rows], [row["error"] for row in rows],
                      marker=marker, color=color, label=f"FD{order}")
        ax.set_title(f"T = 1, A = {amplitude:g}")
        ax.set_xticks([16, 32, 64], ["16", "32", "64"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(5e-5, 2)
        ax.set_xlabel("Grid points along x")
        ax.legend()
    small.set_ylabel("Max field error / A")
    for amplitude, color, linestyle in ((0.01, "#0F4D92", "-"), (0.1, "#B64342", "--")):
        times, errors = zip(*read(f"long-a{amplitude}.json")["samples"])
        history.semilogy(times, errors, color=color, linestyle=linestyle, label=f"A = {amplitude:g}")
        history.plot(times[-1], errors[-1], "o", color=color)
        if amplitude == 0.1 and times[-1] < 100:
            history.annotate(f"Last sample: {times[-1]:g}", (times[-1], errors[-1]),
                             xytext=(10, -3), textcoords="offset points", color=color)
    history.set_title("FD4, 32 points along x")
    history.set_xlabel("Crossing time")
    history.set_ylabel("Max field error / A")
    history.set_xlim(0, 103)
    history.legend(loc="lower right")
    fig.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
