import numpy as np
import pandas as pd
from .metrics import calculate_metrics

def make_ensemble(frames, models, method="mean", n=3):
    selected=[m for m in models if m in frames]
    if not selected: return pd.DataFrame(columns=["Date","Actual","Prediction"]), {}
    wide=None
    for name in selected:
        part=frames[name][["Date","Actual","Prediction"]].rename(columns={"Actual":f"a_{name}","Prediction":name})
        wide=part if wide is None else wide.merge(part,on="Date",how="inner")
    if wide.empty: return pd.DataFrame(columns=["Date","Actual","Prediction"]), {}
    wide["Actual"]=wide[[f"a_{m}" for m in selected]].mean(axis=1)
    pred=wide[selected]
    if method=="median": out=pred.median(axis=1)
    elif method=="top_n": out=pred[selected[:max(1,n)]].mean(axis=1)
    elif method=="inverse_mae":
        errs={m:np.mean(np.abs(frames[m].set_index("Date").loc[wide.Date,"Prediction"].to_numpy()-frames[m].set_index("Date").loc[wide.Date,"Actual"].to_numpy())) for m in selected}
        w={m:1/max(v,1e-9) for m,v in errs.items()}; out=pred.mul(pd.Series(w)/sum(w.values()),axis=1).sum(axis=1)
    else: out=pred.mean(axis=1)
    result=wide[["Date","Actual"]].copy(); result["Prediction"]=out; result["error"]=out-result.Actual; result["abs_error"]=result.error.abs(); result["direction_correct"]=(result.Actual.diff().gt(0)==result.Prediction.diff().gt(0)).where(result.Actual.diff().notna(),pd.NA)
    return result, calculate_metrics(result)
