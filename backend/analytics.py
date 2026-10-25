import pandas as pd

def comparison_frames(frames, models=None, common=True, start=None, end=None):
    names=[m for m in (models or list(frames)) if m in frames]
    if not names:return names, pd.DataFrame()
    sets=[set(frames[m].Date) for m in names]
    dates=set.intersection(*sets) if common else set.union(*sets)
    if start: dates={d for d in dates if d>=pd.Timestamp(start)}
    if end: dates={d for d in dates if d<=pd.Timestamp(end)}
    wide=pd.DataFrame({"Date":sorted(dates)})
    for m in names:
        x=frames[m].set_index("Date")
        wide["Actual "+m]=wide.Date.map(x.Actual)
        wide[m]=wide.Date.map(x.Prediction)
    actual_cols=["Actual "+m for m in names]
    wide["Actual"]=wide[actual_cols].mean(axis=1,skipna=True)
    return names,wide
