"""Stream 20k events (level shift at 10k) through the pipeline with and without drift handling; write
results/report.md, results/report.json and results/timeline.png.   python run.py   (~3 s)"""
import argparse
import json
from pathlib import Path

from streaming.evaluate import summarise
from streaming.model import Detector
from streaming.pipeline import run
from streaming.stream import events

N, SHIFT_AT, SEED = 20_000, 10_000, 0


def main(out="results"):
    out = Path(out)
    out.mkdir(exist_ok=True)
    report, timelines = {}, {}
    for name, dh in (("drift handling ON", True), ("drift handling OFF", False)):
        det = Detector(2, drift_handling=dh)
        results, metrics, secs = run(events(N, SEED, shift_at=SHIFT_AT), det)
        s = summarise(results, SHIFT_AT)
        fa_after = [t for t, _, f, lab in results if f and not lab and t >= SHIFT_AT]
        s.update(events_per_s=round(N / secs), drifts_detected_at=det.drifts, model_updates=det.updates,
                 last_false_alarm_after_shift=(fa_after[-1] - SHIFT_AT) if fa_after else None,
                 prometheus=metrics.prometheus())
        report[name] = s
        timelines[name] = results
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [f"{N} events, level shift (+60 % latency) at event {SHIFT_AT}, 0.5 % injected anomalies, seed {SEED}.", "",
             "| run | segment | precision | recall | false alarms | events |", "|---|---|---|---|---|---|"]
    for name, s in report.items():
        for seg in ("before_drift", "first_after_drift", "later_after_drift"):
            r = s[seg]
            lines.append(f"| {name} | {seg.replace('_', ' ')} | {r['precision']:.3f} | {r['recall']:.3f} | {r['false_alarms']} | {r['events']} |")
    for name, s in report.items():
        lines.append(f"\n{name}: drift detections at events {s['drifts_detected_at']}; last false alarm "
                     f"{s['last_false_alarm_after_shift']} events after the shift; {s['model_updates']} per-event model updates; "
                     f"{s['events_per_s']:,} events/s through the threaded queue pipeline.")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    plot(timelines, out / "timeline.png")
    print("\n".join(lines))
    return report


def plot(timelines, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, (name, res) in zip(axes, timelines.items()):
        t = [r[0] for r in res]
        ax.plot(t, [min(r[1], 30) for r in res], lw=0.4, color="0.5", label="anomaly score (|z|, capped at 30)")
        ax.scatter([r[0] for r in res if r[2]], [min(r[1], 30) for r in res if r[2]], s=6, color="tab:red", label="flagged")
        ax.scatter([r[0] for r in res if r[3]], [31] * sum(r[3] for r in res), marker="|", color="tab:blue", label="injected anomaly")
        ax.axvline(SHIFT_AT, color="k", ls="--", lw=1)
        ax.axhline(5, color="tab:red", lw=0.6)
        ax.set(title=name, ylabel="score")
    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    axes[1].set_xlabel("event (dashed line = level shift)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    main(p.parse_args().out)
