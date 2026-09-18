"""FastAPI routes for agent runs and trace retrieval."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .service import AgentService, create_default_service

router = APIRouter()
_service: AgentService | None = None


class RunRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10_000)


def get_service() -> AgentService:
    global _service
    if _service is None:
        _service = create_default_service()
    return _service


@router.get("/")
def root() -> dict[str, str]:
    return {"message": "FalconLLM Backend is running"}


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/api/runs")
async def create_run(
    request: RunRequest, service: AgentService = Depends(get_service)
) -> dict:
    try:
        return await service.create_run(request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/api/runs")
def list_runs(
    limit: int = Query(50, ge=1, le=200),
    service: AgentService = Depends(get_service),
) -> dict:
    return {"runs": service.list_runs(limit)}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str, service: AgentService = Depends(get_service)) -> dict:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/api/runs/{run_id}/trace")
def get_trace(run_id: str, service: AgentService = Depends(get_service)) -> dict:
    trace = service.get_trace(run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "trace": trace}
