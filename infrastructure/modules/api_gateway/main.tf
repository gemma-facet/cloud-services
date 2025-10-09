variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Primary region for resources"
  type        = string
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
}

variable "service_account_email" {
  description = "Service account email for API Gateway authentication"
  type        = string
}

variable "preprocessing_service_url" {
  description = "URL of the preprocessing service"
  type        = string
}

variable "training_service_url" {
  description = "URL of the training service"
  type        = string
}

variable "inference_service_url" {
  description = "URL of the inference service"
  type        = string
}

# API Gateway
resource "google_api_gateway_api" "gemma_api" {
  provider = google-beta
  api_id   = terraform.workspace == "default" ? "gemma-api" : "gemma-api-${terraform.workspace}"
  project  = var.project_id

  labels = var.labels
}

# API Config
# Note that GCP does NOT allow updating API config in-place,
# so we use a version-based ID that changes only when the config content changes.
locals {
  api_config_content = templatefile("${path.module}/api-config.yaml", {
    project_id               = var.project_id
    service_account_email    = var.service_account_email
    preprocessing_service_url = var.preprocessing_service_url
    training_service_url     = var.training_service_url
    inference_service_url    = var.inference_service_url
  })
  
  # Create a deterministic config ID based on content hash
  # Format: gemma-api-<hash8>
  config_hash = substr(sha256(local.api_config_content), 0, 8)
  api_config_id = "gemma-api-${local.config_hash}"
}

resource "google_api_gateway_api_config" "gemma_api_config" {
  provider      = google-beta
  api           = google_api_gateway_api.gemma_api.api_id
  api_config_id = local.api_config_id
  project       = var.project_id

  openapi_documents {
    document {
      path = "spec.yaml"
      contents = base64encode(local.api_config_content)
    }
  }

  lifecycle {
    create_before_destroy = true
  }

  labels = var.labels
}

# API Gateway
resource "google_api_gateway_gateway" "gemma_gateway" {
  provider   = google-beta
  api_config = google_api_gateway_api_config.gemma_api_config.id
  gateway_id = "gemma-gateway"
  project    = var.project_id
  region     = var.region

  labels = var.labels
}
