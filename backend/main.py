from contextlib import asynccontextmanager
from datetime import date
from io import StringIO
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .config import MODEL_SOURCES, DEFAULT_WEIGHTS
from .data_loader import load_all
from .metrics import calculate_metrics, composite_score
from .ensemble import make_ensemble
from .analytics import comparison_frames

STATE={"frames":{},"metadata":{}}
def reload_data(): STATE["frames"],STATE["metadata"]=load_all(); return STATE
@asynccontextmanager
async def lifespan(app):
    reload_data(); yield
app=FastAPI(title="Crypto Forecasting Analytics API",version="1.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])

def names_arg(models): return [x.strip() for x in models.split(",") if x.strip()] if models else list(STATE["frames"])
def subset(models=None,start=None,end=None):
    out={}
    for n in names_arg(models):
        if n not in STATE["frames"]: continue
        f=STATE["frames"][n]
        if start:f=f[f.Date>=pd.Timestamp(start)]
        if end:f=f[f.Date<=pd.Timestamp(end)]
        out[n]=f
    return out
def rows(df):
    x=df.copy()
    if "Date" in x:x["Date"]=x.Date.dt.strftime("%Y-%m-%d")
    return x.replace({np.nan:None}).to_dict("records")
def metrics_for(frames,common=True):
    names,wide=comparison_frames(frames,common=common)
    ans={}
    if common and len(names) and len(wide):
        shared={}
        for n in names:
            aligned=pd.DataFrame({"Date":wide.Date,"Actual":wide.Actual,"Prediction":wide[n]})
            aligned=aligned.dropna(subset=["Actual","Prediction"])
            aligned["error"]=aligned.Prediction-aligned.Actual
            aligned["abs_error"]=aligned.error.abs()
            aligned["direction_correct"]=(aligned.Actual.diff().gt(0)==aligned.Prediction.diff().gt(0)).where(aligned.Actual.diff().notna(),pd.NA)
            shared[n]=aligned
    else: shared=frames
    for n,f in shared.items(): ans[n]=calculate_metrics(f)
    return ans

def actual_mismatches(frames=None,tolerance=0.01):
    fs=frames or STATE["frames"]
    joined=None
    for name,f in fs.items():
        p=f[["Date","Actual"]].rename(columns={"Actual":name})
        joined=p if joined is None else joined.merge(p,on="Date",how="inner")
    if joined is None or len(joined)<1:return []
    cols=[n for n in fs if n in joined]
    spread=joined[cols].max(axis=1)-joined[cols].min(axis=1)
    bad=joined[spread>tolerance].copy();bad["difference"]=spread[spread>tolerance]
    return [{"date":r["Date"].date().isoformat(),"difference":float(r["difference"]),"actuals":{n:float(r[n]) for n in cols}} for _,r in bad.iterrows()]

@app.get("/api/health")
def health(): return {"status":"ok","models_loaded":len(STATE["frames"]),"models_total":len(MODEL_SOURCES)}
@app.post("/api/reload")
def reload_endpoint():
    reload_data();return {"status":"reloaded","models_loaded":len(STATE["frames"]),"models":list(STATE["metadata"].values())}
@app.get("/api/models")
def models(): return list(STATE["metadata"].values())
@app.get("/api/data-quality")
def data_quality():
    return {"actual_mismatches":actual_mismatches(),"cleaning": [{"model":n,"duplicates_removed":m.get("duplicates_removed",0),"rows_dropped":m.get("rows_dropped",0),"rows":m.get("rows",0),"error":m.get("error")} for n,m in STATE["metadata"].items()]}
@app.get("/api/overview")
def overview(models:str|None=None,start:str|None=None,end:str|None=None,common_dates_only:bool=True):
    fs=subset(models,start,end); ms=metrics_for(fs,common_dates_only); scores=composite_score(ms)
    best=min((n for n in scores if scores[n] is not None),key=lambda n:scores[n],default=None)
    _,w=comparison_frames(fs,common=common_dates_only)
    dates=sorted(set.union(*(set(f.Date) for f in fs.values()))) if fs else []
    latest={n:{"date":f.Date.max().date().isoformat(),"prediction":float(f.iloc[-1].Prediction)} for n,f in fs.items() if len(f)}
    return {"best_model":best,"best_mae":ms.get(best,{}).get("mae") if best else None,"models_loaded":len(fs),"date_span":{"start":min(dates).date().isoformat() if dates else None,"end":max(dates).date().isoformat() if dates else None},"common_dates":len(w),"latest_actual":float(fs[next(iter(fs))].iloc[-1].Actual) if fs else None,"latest_prediction":latest,"metrics":ms}
@app.get("/api/predictions")
def predictions(models:str|None=None,start:str|None=None,end:str|None=None,common_dates_only:bool=True):
    fs=subset(models,start,end); names,wide=comparison_frames(fs,common=common_dates_only)
    if len(wide)>2000:wide=wide.iloc[np.linspace(0,len(wide)-1,2000).astype(int)]
    return {"models":names,"common_dates_only":common_dates_only,"data":rows(wide[["Date","Actual"]+names]) if len(wide) else []}
@app.get("/api/predictions/{day}/neighbors")
def neighbors(day:date):
    ds=sorted(set.union(*(set(f.Date) for f in STATE["frames"].values()))) if STATE["frames"] else []
    cur=pd.Timestamp(day); before=[x for x in ds if x<cur]; after=[x for x in ds if x>cur]
    return {"date":day.isoformat(),"previous":before[-1].date().isoformat() if before else None,"next":after[0].date().isoformat() if after else None}
@app.get("/api/predictions/{day}")
def prediction_day(day:date):
    d=pd.Timestamp(day); vals=[]
    for n,f in STATE["frames"].items():
        hit=f[f.Date==d]
        if hit.empty: vals.append({"model":n,"color":MODEL_SOURCES[n]["color"],"status":"no saved prediction"});continue
        r=hit.iloc[0]; vals.append({"model":n,"color":MODEL_SOURCES[n]["color"],"status":"ok","actual":float(r.Actual),"predicted":float(r.Prediction),"error":float(r.error),"abs_error":float(r.abs_error),"pct_error":float(r.pct_error) if pd.notna(r.pct_error) else None,"over_under":"over" if r.error>0 else "under" if r.error<0 else "exact"})
    rank={v["model"]:i+1 for i,v in enumerate(sorted((x for x in vals if x["status"]=="ok"),key=lambda x:x["abs_error"]))}
    for x in vals:x["rank"]=rank.get(x["model"])
    return {"date":day.isoformat(),"models":vals}
@app.get("/api/metrics")
def get_metrics(models:str|None=None,start:str|None=None,end:str|None=None,common_dates_only:bool=True):return metrics_for(subset(models,start,end),common_dates_only)
@app.get("/api/best")
def best(models:str|None=None,start:str|None=None,end:str|None=None,common_dates_only:bool=True,mae_weight:float=.35,rmse_weight:float=.25,mape_weight:float=.20,directional_accuracy_weight:float=.20):
    fs=subset(models,start,end); ms=metrics_for(fs,common_dates_only); weights={"mae":mae_weight,"rmse":rmse_weight,"mape":mape_weight,"directional_accuracy":directional_accuracy_weight}; scores=composite_score(ms,weights)
    rank=sorted(({"model":n,"score":s,**ms[n]} for n,s in scores.items() if s is not None),key=lambda x:x["score"])
    winner=rank[0] if rank else None; caveats=[]
    if not common_dates_only:caveats.append("Models are evaluated over their own date ranges; this is not a like-for-like comparison.")
    if winner and winner["rows"]<30:caveats.append("Small sample: fewer than 30 comparable dates.")
    if len(rank)>1 and rank[0]["score"] and (rank[1]["score"]-rank[0]["score"])/rank[0]["score"]<=.02:caveats.append("The runner-up is within 2% of the winner's composite score (statistically close).")
    if winner and winner.get("mase") is not None and winner["mase"]>1:caveats.append("The recommended model is worse than the naive previous-day baseline (MASE > 1).")
    if actual_mismatches(fs):caveats.append("Actual prices differ across model files on shared dates; the chart and common-date metrics use their mean. See Data & Coverage for details.")
    gap=rank[1]["score"]-rank[0]["score"] if len(rank)>1 else 1
    confidence="High" if winner and winner["rows"]>=100 and gap>.05 else "Medium" if winner and winner["rows"]>=30 else "Low"
    return {"winner":winner["model"] if winner else None,"ranking":rank,"scores":scores,"weights":weights,"reason":f"{winner['model']} has the strongest weighted combination of error and direction metrics across {winner['rows']} dates." if winner else "No model has usable data.","caveats":caveats,"confidence":confidence,"like_for_like":common_dates_only}
@app.get("/api/rankings/history")
def rankings_history(models:str|None=None,window:int=30):
    fs=subset(models); all_dates=sorted(set.union(*(set(f.Date) for f in fs.values()))) if fs else []; out=[]
    for i,d in enumerate(all_dates):
        start=all_dates[max(0,i-window+1)]; ms={n:calculate_metrics(f[(f.Date>=start)&(f.Date<=d)]) for n,f in fs.items()}; sc=composite_score(ms); win=min((n for n,v in sc.items() if v is not None),key=lambda n:sc[n],default=None); out.append({"date":d.date().isoformat(),"winner":win,"score":sc.get(win) if win else None})
    return {"window":window,"data":out}
@app.get("/api/winrate")
def winrate(models:str|None=None):
    fs=subset(models); names,w=comparison_frames(fs,common=True); wins={n:0 for n in names}; h2h=[]
    for _,r in w.iterrows():
        es={n:abs(r[n]-r.Actual) for n in names if pd.notna(r[n])};
        if es:
            m=min(es.values());
            for n,e in es.items():
                if e==m:wins[n]+=1
    for a in names:
        for b in names:
            if a!=b:
                both=w[[a,b,"Actual"]].dropna(); h2h.append({"a":a,"b":b,"share_a_wins":float((abs(both[a]-both.Actual)<abs(both[b]-both.Actual)).mean()) if len(both) else None})
    count=len(w); return {"days":count,"models":[{"model":n,"wins":c,"share":c/count if count else 0} for n,c in wins.items()],"head_to_head":h2h}
@app.get("/api/residuals")
def residuals(model:str):
    if model not in STATE["frames"]:raise HTTPException(404,"Model unavailable")
    f=STATE["frames"][model]; e=f.error.dropna(); hist,edges=np.histogram(e,bins=20); ac=[float(e.autocorr(lag=i)) if len(e)>i else None for i in range(1,31)]
    qq=np.quantile(e,np.linspace(.01,.99,min(100,len(e)))).tolist() if len(e) else []
    return {"model":model,"series":rows(f[["Date","error"]]),"histogram":[{"left":float(edges[i]),"right":float(edges[i+1]),"count":int(hist[i])} for i in range(len(hist))],"qq":qq,"autocorrelation":ac}
@app.get("/api/rolling")
def rolling(window:int=30,models:str|None=None):
    fs=subset(models); out={}
    for n,f in fs.items():
        x=f.copy();x["rolling_mae"]=x.abs_error.rolling(window,min_periods=1).mean();x["rolling_rmse"]=(x.error.pow(2).rolling(window,min_periods=1).mean())**.5;x["rolling_directional_accuracy"]=x.direction_correct.astype("boolean").astype("Float64").rolling(window,min_periods=1).mean()*100;out[n]=rows(x[["Date","rolling_mae","rolling_rmse","rolling_directional_accuracy"]])
    return {"window":window,"data":out}
@app.get("/api/calendar")
def calendar(model:str,year:int|None=None):
    if model not in STATE["frames"]:raise HTTPException(404,"Model unavailable")
    f=STATE["frames"][model].copy();
    if year:f=f[f.Date.dt.year==year]
    return {"model":model,"data":rows(f[["Date","abs_pct_error"]])}
@app.get("/api/breakdown")
def breakdown(model:str,by:str="weekday"):
    if model not in STATE["frames"]:raise HTTPException(404,"Model unavailable")
    f=STATE["frames"][model].copy()
    if by=="weekday":f["group"]=f.Date.dt.day_name()
    elif by=="month":f["group"]=f.Date.dt.month_name()
    elif by=="price_level":f["group"]=pd.qcut(f.Actual,4,duplicates="drop").astype(str)
    elif by=="volatility":f["group"]=pd.qcut(f.Actual.pct_change().rolling(14,min_periods=3).std(),3,labels=["low","medium","high"],duplicates="drop").astype(str)
    else:raise HTTPException(400,"by must be weekday, month, volatility, or price_level")
    return {"model":model,"by":by,"data":[{"group":str(k),"mae":float(g.abs_error.mean()),"rmse":float((g.error.pow(2).mean())**.5),"rows":int(len(g))} for k,g in f.groupby("group",observed=True)]}
@app.get("/api/scatter")
def scatter(model:str):
    if model not in STATE["frames"]:raise HTTPException(404,"Model unavailable")
    f=STATE["frames"][model];return {"model":model,"data":rows(f[["Date","Actual","Prediction"]]),"reference":{"min":float(min(f.Actual.min(),f.Prediction.min())),"max":float(max(f.Actual.max(),f.Prediction.max()))}}
@app.get("/api/ensemble")
def ensemble(models:str|None=None,method:str="mean",n:int=3):
    fs=subset(models); selected=list(fs); e,em=make_ensemble(fs,selected,method,n); sm=metrics_for(fs,True);best=min((v.get("mae") for v in sm.values() if v.get("mae") is not None),default=None)
    return {"method":method,"models":selected,"rows":len(e),"series":rows(e[["Date","Actual","Prediction"]]) if len(e) else [],"metrics":em,"best_single_mae":best,"beats_best_single":bool(best is not None and em.get("mae",float("inf"))<best),"derived_from_saved_predictions":True}
@app.get("/api/backtest")
def backtest(model:str,threshold:float=0):
    if model not in STATE["frames"]:raise HTTPException(404,"Model unavailable")
    f=STATE["frames"][model].copy(); f["prev"]=f.Actual.shift(1);f=f.dropna(subset=["prev"]); signal=f.Prediction-f.prev>threshold; daily=f.Actual.pct_change().fillna(0); strat=(daily*signal).fillna(0); equity=(1+strat).cumprod(); hold=(1+daily).cumprod()
    return {"model":model,"educational_only":True,"disclaimer":"Educational only; uses saved predictions and ignores fees, slippage, and look-ahead concerns.","data":[{"date":d.date().isoformat(),"strategy":float(s),"buy_and_hold":float(h)} for d,s,h in zip(f.Date,equity,hold)],"total_return":float(equity.iloc[-1]-1) if len(equity) else None,"buy_and_hold_return":float(hold.iloc[-1]-1) if len(hold) else None,"trades":int(signal.astype(int).diff().fillna(signal.iloc[0] if len(signal) else 0).abs().sum()),"hit_rate":float((strat[signal]>0).mean()) if signal.any() else None,"derived_from_saved_predictions":True}
def csv_response(df,filename):
    s=StringIO();df.to_csv(s,index=False);return StreamingResponse(iter([s.getvalue()]),media_type="text/csv",headers={"Content-Disposition":f"attachment; filename={filename}"})
@app.get("/api/export/metrics.csv")
def export_metrics(models:str|None=None):return csv_response(pd.DataFrame([{"model":n,**m} for n,m in metrics_for(subset(models)).items()]),"metrics.csv")
@app.get("/api/export/predictions.csv")
def export_predictions(models:str|None=None,common_dates_only:bool=True):
    fs=subset(models); names,w=comparison_frames(fs,common=common_dates_only);return csv_response(w[["Date","Actual"]+names] if len(w) else pd.DataFrame(),"predictions.csv")
@app.get("/api/export/date/{day}.csv")
def export_day(day:date):return csv_response(pd.DataFrame(prediction_day(day)["models"]),f"predictions-{day}.csv")
