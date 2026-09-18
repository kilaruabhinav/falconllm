from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from services.agent_adapter import run_task

app = FastAPI(title="FalconLLM API")

app.add_middleware(
    CORSMiddleware,
    # Vite uses 5173 by default and advances through nearby ports when occupied.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517[3-9]$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
)


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)


@app.get("/")
def root():
    return {"message": "FalconLLM Backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/runs")
def create_run(request: RunRequest):
    """Run through the replaceable agent adapter until the real agent is wired in."""
    return run_task(request.prompt)
