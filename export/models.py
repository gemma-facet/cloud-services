from sqlmodel import Field, SQLModel
from datetime import datetime
from typing import Literal, Optional


class Export(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    job_id: str
    job_name: str
    base_model_id: str
    provider: Literal["unsloth", "huggingface"]
    adapter_path: str
    gguf_path: Optional[str] = None
    merged_path: Optional[str] = None
    created_at: datetime = Field(default=datetime.now())
    updated_at: datetime = Field(default=datetime.now())


class ExportRequest(SQLModel):
    job_id: str
    export_type = Literal["adapter", "merged", "gguf"]


class ExportResponse(SQLModel):
    success: bool
    job_id: str
    export_type: Literal["adapter", "merged", "gguf"]
    adapter_path: Optional[str] = None
    merged_path: Optional[str] = None
    gguf_path: Optional[str] = None
