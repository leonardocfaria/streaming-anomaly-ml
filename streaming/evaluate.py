"""Offline scoring of a run against the injected labels, split before/after the drift."""


def summarise(results, shift_at: int | None = None, adapt_window: int = 500) -> dict:
    def prf(rows):
        tp = sum(f and lab for _, _, f, lab in rows)
        fp = sum(f and not lab for _, _, f, lab in rows)
        fn = sum(lab and not f for _, _, f, lab in rows)
        return {"precision": tp / (tp + fp) if tp + fp else 1.0, "recall": tp / (tp + fn) if tp + fn else 1.0,
                "false_alarms": fp, "events": len(rows)}

    out = {"overall": prf(results)}
    if shift_at is not None:
        out["before_drift"] = prf([r for r in results if r[0] < shift_at])
        out["first_after_drift"] = prf([r for r in results if shift_at <= r[0] < shift_at + adapt_window])
        out["later_after_drift"] = prf([r for r in results if r[0] >= shift_at + adapt_window])
    return out
