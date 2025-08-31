from fastapi import FastAPI
from schema import ExportRequest, ExportResponse

app = FastAPI(
    title="Gemma Export Service",
    version="1.0.0",
    description="Export service for exporting models",
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "export"}


@app.post("/export", response_model=ExportResponse)
async def export(request: ExportRequest):
    export = ExportResponse(
        success=True,
        job_id=request.job_id,
        export_type=request.export_type,
    )
    return export
