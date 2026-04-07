from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_sentiment

app = FastAPI(title="Sentiment API")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    sentiment = predict_sentiment(req.text)
    return {"sentiment": sentiment}
