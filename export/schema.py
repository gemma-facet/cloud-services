from pydantic import BaseModel
from typing import Literal, Optional


class ExportRequest(BaseModel):
    job_id: str
    export_type: Literal["adapter", "merged", "gguf"]
    hf_token: Optional[str] = None


class ExportResponse(BaseModel):
    success: bool
    job_id: str
    export_type: Literal["adapter", "merged", "gguf"]
    adapter_path: Optional[str] = None
    merged_path: Optional[str] = None
    gguf_path: Optional[str] = None
