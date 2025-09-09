# =============================================================================
# Gemma Fine-tuning Infrastructure Variables
# =============================================================================

variable "project_id" {
  description = "Google Cloud Project ID where resources will be created"
  type        = string
  validation {
    condition     = length(var.project_id) > 0
    error_message = "Project ID cannot be empty."
  }
}

variable "region" {
  description = "Primary region for resources (e.g., us-central1, us-east1)"
  type        = string
  default     = "us-central1"
  validation {
    condition = can(regex("^[a-z]+-[a-z]+[0-9]$", var.region))
    error_message = "Region must be a valid GCP region format (e.g., us-central1)."
  }
}

variable "labels" {
  description = "Labels to apply to all resources for organization and cost tracking"
  type        = map(string)
  default = {
    project     = "gemma-fine-tuning"
    managed-by  = "terraform"
  }
  validation {
    condition     = can(var.labels["project"]) && can(var.labels["managed-by"])
    error_message = "Labels must include 'project' and 'managed-by' keys."
  }
}
