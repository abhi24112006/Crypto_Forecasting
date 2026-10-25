import pandas as pd
from .config import ROOT, MODEL_SOURCES

def load_all():
    frames, metadata, quality = {}, {}, {}
    for name, spec in MODEL_SOURCES.items():
        path = ROOT / spec["path"]
        meta = {"name": name, "color": spec["color"], "rows": 0, "status": "error", "error": None, "path": spec["path"], "duplicates_removed": 0, "rows_dropped": 0}
        if not path.exists():
            meta["error"] = f"Missing prediction file: {spec['path']}"
            metadata[name] = meta
            continue
        try:
            raw = pd.read_csv(path)
            required = [spec["date"], spec["actual"], spec["prediction"]]
            absent = [c for c in required if c not in raw.columns]
            if absent:
                meta["error"] = f"Missing required columns: {', '.join(absent)}"
                metadata[name] = meta
                continue
            frame = pd.DataFrame({"Date": pd.to_datetime(raw[spec["date"]], errors="coerce").dt.normalize(), "Actual": pd.to_numeric(raw[spec["actual"]], errors="coerce"), "Prediction": pd.to_numeric(raw[spec["prediction"]], errors="coerce")})
            before = len(frame)
            frame = frame.dropna(subset=["Date", "Actual", "Prediction"])
            meta["rows_dropped"] = before - len(frame)
            meta["duplicates_removed"] = int(frame.duplicated("Date", keep="last").sum())
            frame = frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
            frame["error"] = frame.Prediction - frame.Actual
            frame["abs_error"] = frame.error.abs()
            frame["pct_error"] = frame.error / frame.Actual.replace(0, pd.NA) * 100
            frame["abs_pct_error"] = frame.pct_error.abs()
            actual_delta, pred_delta = frame.Actual.diff(), frame.Prediction.diff()
            frame["direction_correct"] = (actual_delta.gt(0) == pred_delta.gt(0)).where(actual_delta.notna(), pd.NA)
            meta.update({"rows": len(frame), "status": "ok" if len(frame) else "error", "error": None if len(frame) else "No valid rows after cleaning", "first_date": frame.Date.min().date().isoformat() if len(frame) else None, "last_date": frame.Date.max().date().isoformat() if len(frame) else None, "columns": list(raw.columns)})
            if len(frame): frames[name] = frame
        except Exception as exc:
            meta["error"] = f"Could not read CSV: {exc}"
        metadata[name] = meta
    return frames, metadata
