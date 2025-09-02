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
