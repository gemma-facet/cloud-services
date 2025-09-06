from pydantic import BaseModel
from typing import Literal, Optional

export_type = Literal["adapter", "merged", "gguf"]


class ExportRequest(BaseModel):
    job_id: str
    export_type: export_type
    hf_token: Optional[str] = None


class ExportInfo(BaseModel):
    type: export_type
    path: str


class ExportResponse(BaseModel):
    success: bool
    job_id: str
    export: ExportInfo


class ExportPaths(BaseModel):
    adapter: Optional[str] = None
    merged: Optional[str] = None
    gguf: Optional[str] = None


class JobResponse(BaseModel):
    base_model_id: str
    created_at: str
    export: ExportPaths = ExportPaths()
    export_status: Optional[str]
    job_id: str
    job_name: str


class JobsResponse(BaseModel):
    jobs: list[JobResponse]
