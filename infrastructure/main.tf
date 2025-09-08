# Shared infrastructure for Gemma fine-tuning services
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.44"
    }
  }
}

# Variables
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Region for resources"
  type        = string
  default     = "us-central1"
}

variable "data_bucket_name" {
  description = "GCS bucket for datasets"
  type        = string
  default     = "gemma-dataset-bucket"
}

variable "export_bucket_name" {
  description = "GCS bucket for model exports"
  type        = string
  default     = "gemma-export-bucket"
}

variable "config_bucket_name" {
  description = "GCS bucket for training configs"
  type        = string
  default     = "gemma-train-config"
}

variable "training_image_tag" {
  description = "Docker image tag for training service"
  type        = string
  default     = "latest"
}

variable "preprocessing_image_tag" {
  description = "Docker image tag for preprocessing service"
  type        = string
  default     = "latest"
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value

  disable_on_destroy = false
}

# Shared Service Account for both services
resource "google_service_account" "gemma_services" {
  account_id   = "gemma-services"
  display_name = "Gemma Services Account"
  description  = "Shared service account for preprocessing and training services"
}

# Grant necessary permissions
resource "google_project_iam_member" "gemma_services_storage" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.gemma_services.email}"
}

resource "google_project_iam_member" "gemma_services_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gemma_services.email}"
}

resource "google_project_iam_member" "gemma_services_run" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.gemma_services.email}"
}

resource "google_project_iam_member" "gemma_services_firestore" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.gemma_services.email}"
}


# Artifact Registry Repository (shared)
resource "google_artifact_registry_repository" "gemma_services" {
  location      = var.region
  repository_id = "gemma-fine-tuning"
  description   = "Docker repository for Gemma fine-tuning services"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# GCS Buckets (shared)
resource "google_storage_bucket" "data_bucket" {
  name     = var.data_bucket_name
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "export_bucket" {
  name     = var.export_bucket_name
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

# Add config bucket resource
resource "google_storage_bucket" "config_bucket" {
  name     = var.config_bucket_name
  location = var.region

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

# Preprocessing Service
resource "google_cloud_run_v2_service" "preprocessing_service" {
  name     = "preprocessing-service"
  location = var.region

  template {
    service_account = google_service_account.gemma_services.email
    
    scaling {
      min_instance_count = 0
      max_instance_count = 5  # Can handle multiple preprocessing jobs
    }

    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/gemma-fine-tuning/preprocessing-service:${var.preprocessing_image_tag}"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "GCS_DATA_BUCKET_NAME"
        value = var.data_bucket_name
      }
      
      env {
        name  = "STORAGE_TYPE"
        value = "gcs"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        timeout_seconds = 5
        period_seconds = 10
        failure_threshold = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds = 5
        period_seconds = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.gemma_services
  ]
}

# Training Service
resource "google_cloud_run_v2_service" "training_service" {
  name     = "training-service"
  location = var.region

  template {
    service_account = google_service_account.gemma_services.email
    
    scaling {
      min_instance_count = 0
    }

    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/gemma-fine-tuning/training-service:${var.training_image_tag}"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle = false
      }

      env {
        name  = "GCS_DATA_BUCKET_NAME"
        value = var.data_bucket_name
      }
      
      env {
        name  = "GCS_EXPORT_BUCKET_NAME"
        value = var.export_bucket_name
      }

      env {
        name  = "GCS_CONFIG_BUCKET_NAME"
        value = var.config_bucket_name
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds = 5
        period_seconds = 10
        failure_threshold = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds = 5
        period_seconds = 30
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.gemma_services
  ]
}

# Add Inference Service
resource "google_cloud_run_v2_service" "inference_service" {
  name     = "inference-service"
  location = var.region
  provider = google-beta

  template {
    gpu_zonal_redundancy_disabled = true
    service_account = google_service_account.gemma_services.email
    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }
    containers {
      image = "us-central1-docker.pkg.dev/${var.project_id}/gemma-fine-tuning/inference-service:latest"
      resources {
        limits = {
          cpu    = "4"
          memory = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle = false
      }
      env {
        name  = "GCS_DATA_BUCKET_NAME"
        value = var.data_bucket_name
      }
      env {
        name  = "GCS_EXPORT_BUCKET_NAME"
        value = var.export_bucket_name
      }
      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }
      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 30
        timeout_seconds = 5
        period_seconds = 10
        failure_threshold = 3
      }
      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds = 5
        period_seconds = 30
        failure_threshold = 3
      }
    }
    node_selector {
      accelerator = "nvidia-l4"
    }
  }
  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.gemma_services
  ]
}

# Allow unauthenticated access to all services
resource "google_cloud_run_service_iam_binding" "preprocessing_service_public" {
  location = google_cloud_run_v2_service.preprocessing_service.location
  service  = google_cloud_run_v2_service.preprocessing_service.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

resource "google_cloud_run_service_iam_binding" "training_service_public" {
  location = google_cloud_run_v2_service.training_service.location
  service  = google_cloud_run_v2_service.training_service.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

resource "google_cloud_run_service_iam_binding" "inference_service_public" {
  location = google_cloud_run_v2_service.inference_service.location
  service  = google_cloud_run_v2_service.inference_service.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

# Training Job (Cloud Run Job)
resource "google_cloud_run_v2_job" "training_job" {
  name     = "training-job"
  location = var.region
  provider = google-beta
  
  template {
    template {
      service_account = google_service_account.gemma_services.email
      containers {
        image = "us-central1-docker.pkg.dev/${var.project_id}/gemma-fine-tuning/training-job:latest"
        resources {
          limits = {
            cpu    = "8"
            memory = "32Gi"
            "nvidia.com/gpu" = "1"
          }
        }
        env {
          name  = "GCS_DATA_BUCKET_NAME"
          value = var.data_bucket_name
        }
        env {
          name  = "GCS_EXPORT_BUCKET_NAME"
          value = var.export_bucket_name
        }
        env {
          name  = "GCS_CONFIG_BUCKET_NAME"
          value = var.config_bucket_name
        }
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
      }
      max_retries    = 0
      timeout        = "3600s"
    }
  }
  depends_on = [
    google_project_service.required_apis,
    google_artifact_registry_repository.gemma_services
  ]
}

# Outputs
output "preprocessing_service_url" {
  description = "URL of the preprocessing service"
  value       = google_cloud_run_v2_service.preprocessing_service.uri
}

output "training_service_url" {
  description = "URL of the training service"
  value       = google_cloud_run_v2_service.training_service.uri
}

output "artifact_registry_repo" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.gemma_services.name
}

output "data_bucket_name" {
  description = "Data bucket name"
  value       = google_storage_bucket.data_bucket.name
}

output "export_bucket_name" {
  description = "Export bucket name"
  value       = google_storage_bucket.export_bucket.name
}

output "config_bucket_name" {
  description = "Config bucket name"
  value       = google_storage_bucket.config_bucket.name
}
output "inference_service_url" {
  description = "URL of the inference service"
  value       = google_cloud_run_v2_service.inference_service.uri
}
output "training_job_name" {
  description = "Cloud Run training job name"
  value       = google_cloud_run_v2_job.training_job.name
}