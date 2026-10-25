from pydantic import BaseModel
from typing import Any

class Health(BaseModel):
    status: str
    models_loaded: int
    models_total: int

class ModelStatus(BaseModel):
    name: str
    color: str
    rows: int
    status: str
    error: str | None = None
    first_date: str | None = None
    last_date: str | None = None
    path: str | None = None

class DataResponse(BaseModel):
    data: list[dict[str, Any]]
