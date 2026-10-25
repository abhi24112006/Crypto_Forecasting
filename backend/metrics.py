import numpy as np
import pandas as pd

def calculate_metrics(frame: pd.DataFrame):
    if frame.empty: return {k: None for k in ("mae","rmse","mape","smape","mase","r2","directional_accuracy","bias","max_error","median_abs_error","p90_abs_error","rows")}
    a, p = frame.Actual.to_numpy(float), frame.Prediction.to_numpy(float)
    e = p-a; ae=np.abs(e); denom=np.abs(a)+np.abs(p)
    naive=np.abs(pd.Series(a).diff().dropna().to_numpy())
    d=frame.get("direction_correct", pd.Series(dtype=float)).dropna()
    return {"mae":float(ae.mean()),"rmse":float(np.sqrt(np.mean(e**2))),"mape":float(np.mean(ae[np.abs(a)>1e-12]/np.abs(a[np.abs(a)>1e-12]))*100) if np.any(np.abs(a)>1e-12) else None,"smape":float(np.mean(np.divide(2*ae,denom,out=np.zeros_like(ae),where=denom!=0))*100),"mase":float(ae.mean()/naive.mean()) if len(naive) and naive.mean()!=0 else None,"r2":float(1-np.sum(e**2)/np.sum((a-a.mean())**2)) if len(a)>1 and np.sum((a-a.mean())**2)>0 else None,"directional_accuracy":float(d.mean()*100) if len(d) else None,"bias":float(e.mean()),"max_error":float(ae.max()),"median_abs_error":float(np.median(ae)),"p90_abs_error":float(np.percentile(ae,90)),"rows":int(len(a))}

def composite_score(metrics, weights=None):
    weights=weights or {"mae":.35,"rmse":.25,"mape":.20,"directional_accuracy":.20}
    keys=list(weights); valid={k:[m.get(k) for m in metrics.values() if m.get(k) is not None and np.isfinite(m.get(k))] for k in keys}
    ranges={k:(min(v),max(v)) if v else (0,0) for k,v in valid.items()}; scores={}
    total=sum(max(0,float(x)) for x in weights.values()) or 1
    for name,m in metrics.items():
        vals=[]
        for k,w in weights.items():
            x=m.get(k); lo,hi=ranges[k]
            if x is None: continue
            norm=0.5 if hi==lo else (x-lo)/(hi-lo)
            if k=="directional_accuracy": norm=1-norm
            vals.append((norm,max(0,float(w))))
        scores[name]=sum(v*w for v,w in vals)/sum(w for _,w in vals) if vals and sum(w for _,w in vals) else None
    return scores
