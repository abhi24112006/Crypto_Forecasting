import pandas as pd
from backend.metrics import calculate_metrics, composite_score
from backend.ensemble import make_ensemble
from backend.analytics import comparison_frames

def sample(actual, pred):
    f=pd.DataFrame({"Date":pd.date_range("2024-01-01",periods=len(actual)),"Actual":actual,"Prediction":pred})
    f["error"]=f.Prediction-f.Actual;f["abs_error"]=f.error.abs();f["direction_correct"]=(f.Actual.diff().gt(0)==f.Prediction.diff().gt(0)).where(f.Actual.diff().notna(),pd.NA)
    return f

def test_metrics_known_values():
    m=calculate_metrics(sample([10,12,14],[11,11,15]));assert m["mae"]==1;assert round(m["rmse"],4)==1;assert m["rows"]==3

def test_composite_prefers_lower_error():
    m={"good":{"mae":1,"rmse":1,"mape":1,"directional_accuracy":90},"bad":{"mae":3,"rmse":3,"mape":3,"directional_accuracy":50}}
    assert composite_score(m)["good"] < composite_score(m)["bad"]

def test_ensemble_common_dates():
    a=sample([10,11,12],[10,12,12]);b=sample([11,12,13],[12,11,14]);b.Date+=pd.Timedelta(days=1)
    e,m=make_ensemble({"a":a,"b":b},["a","b"]);assert len(e)==2;assert m["rows"]==2
    weighted,wm=make_ensemble({"a":a,"b":b},["a","b"],method="inverse_mae");assert len(weighted)==2;assert wm["mae"] is not None

def test_intersection_logic():
    a=sample([1,2,3],[1,2,3]);b=sample([1,2],[2,2]);b.Date+=pd.Timedelta(days=1)
    _,w=comparison_frames({"a":a,"b":b},common=True);assert len(w)==2
