from pydantic import BaseModel, Field
from typing import Literal, Optional, List
from datetime import datetime

export_type = Literal["adapter", "merged", "gguf"]
export_status = Literal["running", "completed", "failed"]
export_variant = Literal["raw", "file", "hf"]
export_destination = Literal["gcs", "hf_hub"]


class ExportArtifact(BaseModel):
    type: export_type
    path: str
    variant: export_variant


class ExportSchema(BaseModel):
    export_id: str
    job_id: str
    type: export_type
    destination: List[export_destination] = ["gcs"]
    hf_repo_id: Optional[str] = None
    status: export_status
    message: Optional[str] = None
    artifacts: List[ExportArtifact] = Field(default_factory=list)
    started_at: datetime
    finished_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class JobArtifactsFiles(BaseModel):
    adapter: Optional[str] = None
    merged: Optional[str] = None
    gguf: Optional[str] = None


class JobArtifactsRaw(BaseModel):
    adapter: Optional[str] = None
    merged: Optional[str] = None


class JobArtifactsHF(BaseModel):
    adapter: Optional[str] = None
    merged: Optional[str] = None
    gguf: Optional[str] = None


class JobArtifacts(BaseModel):
    file: JobArtifactsFiles = Field(default_factory=JobArtifactsFiles)
    raw: JobArtifactsRaw = Field(default_factory=JobArtifactsRaw)
    hf: JobArtifactsHF = Field(default_factory=JobArtifactsHF)


# NOTE: This struct is shared between the API and the backend service
class JobSchema(BaseModel):
    job_id: str
    user_id: str
    adapter_path: str
    base_model_id: str
    modality: Optional[Literal["text", "vision"]] = "text"
    artifacts: Optional[JobArtifacts] = Field(default_factory=JobArtifacts)
