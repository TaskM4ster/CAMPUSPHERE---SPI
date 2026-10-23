import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import load_model, predict_students  # noqa: E402

YesNo = Literal["yes", "no"]

NOTE = (
    "This output is a decision-support flag for staff review. "
    "It is not a verdict about the student."
)


class StudentFeatures(BaseModel):
    """Model A inputs: information available after period 1 (no G2)."""

    model_config = ConfigDict(extra="forbid")

    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    age: int = Field(ge=15, le=22)
    address: Literal["U", "R"]
    famsize: Literal["GT3", "LE3"]
    Pstatus: Literal["T", "A"]
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    reason: Literal["home", "reputation", "course", "other"]
    guardian: Literal["mother", "father", "other"]
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    schoolsup: YesNo
    famsup: YesNo
    paid: YesNo
    activities: YesNo
    nursery: YesNo
    higher: YesNo
    internet: YesNo
    romantic: YesNo
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=93)
    G1: int = Field(ge=0, le=20)


class PredictionResponse(BaseModel):
    risk_label: str
    flagged: bool
    probabilities: dict[str, float]
    note: str = NOTE


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    app.state.pipeline, app.state.features = load_model()
    yield


app = FastAPI(
    title="CAMPUSPHERE SPI",
    description="Student performance risk flagging (Model A, early flag).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": hasattr(app.state, "pipeline"),
        "feature_count": len(app.state.features),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentFeatures):
    row = pd.DataFrame([student.model_dump()])
    result = predict_students(
        row, pipeline=app.state.pipeline, features=app.state.features
    )
    r = result.iloc[0]
    return PredictionResponse(
        risk_label=r["risk_label"],
        flagged=bool(r["flagged"]),
        probabilities={
            key.removeprefix("p_"): float(r[key])
            for key in result.columns
            if key.startswith("p_")
        },
    )