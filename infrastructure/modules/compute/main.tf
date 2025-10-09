# Compute module: Cloud Run services and jobs

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
  description = "Service account email for Cloud Run services"
  type        = string
}

variable "artifact_registry_repo_url" {
  description = "Artifact Registry repository URL"
  type        = string
}

variable "data_bucket_name" {
  description = "Data bucket name"
  type        = string
}

variable "export_bucket_name" {
  description = "Export bucket name"
  type        = string
}

variable "config_bucket_name" {
  description = "Config bucket name"
  type        = string
}

variable "files_bucket_name" {
  description = "Files bucket name"
  type        = string
}

variable "firestore_database_name" {
  description = "Firestore database name"
  type        = string
}

variable "training_image_tag" {
  description = "Training service image tag"
  type        = string
  default     = "latest"
}

variable "preprocessing_image_tag" {
  description = "Preprocessing service image tag"
  type        = string
  default     = "latest"
}

variable "inference_image_tag" {
  description = "Inference service image tag"
  type        = string
  default     = "latest"
}

variable "preprocessing_max_instances" {
  description = "Maximum instances for preprocessing service"
  type        = number
  default     = 10
}

variable "training_max_instances" {
  description = "Maximum instances for training service"
  type        = number
  default     = 10
}

variable "inference_max_instances" {
  description = "Maximum instances for inference service"
  type        = number
  default     = 3
}

# Preprocessing Service
resource "google_cloud_run_v2_service" "preprocessing_service" {
  name     = terraform.workspace == "default" ? "preprocessing-service" : "preprocessing-service-${terraform.workspace}"
  location = var.region

  template {
    service_account = var.service_account_email
    
    labels = merge(var.labels, {
      service = "preprocessing"
    })
    
    scaling {
      min_instance_count = 0
      max_instance_count = var.preprocessing_max_instances
    }

    containers {
      image = "${var.artifact_registry_repo_url}/preprocessing-service:${var.preprocessing_image_tag}"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "8Gi"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      env {
        name  = "GCS_DATA_BUCKET_NAME"
        value = var.data_bucket_name
      }
      
      env {
        name  = "STORAGE_TYPE"
        value = "gcs"
      }

      env {
        name  = "PROJECT_ID"
        value = var.project_id
      }

      env {
        name  = "FIRESTORE_DB"
        value = var.firestore_database_name
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds   = 5
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    max_instance_request_concurrency = 30
  }
}

# Training Service
resource "google_cloud_run_v2_service" "training_service" {
  name     = terraform.workspace == "default" ? "training-service" : "training-service-${terraform.workspace}"
  location = var.region

  template {
    service_account = var.service_account_email
    
    labels = merge(var.labels, {
      service = "training"
    })
    
    scaling {
      min_instance_count = 0
      max_instance_count = var.training_max_instances
    }

    containers {
      image = "${var.artifact_registry_repo_url}/training-service:${var.training_image_tag}"
      
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
        cpu_idle          = false
        startup_cpu_boost = true
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

      env {
        name  = "FIRESTORE_DB"
        value = var.firestore_database_name
      }

    // these are required for gRPC client to call jobs
      env {
        name = "REGION"
        value = var.region
      }

      env {
        name = "TRAINING_JOB_NAME"
        value = terraform.workspace == "default" ? "training-job" : "training-job-${terraform.workspace}"
      }

      env {
        name = "EXPORT_JOB_NAME"
        value = terraform.workspace == "default" ? "export-job" : "export-job-${terraform.workspace}"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds   = 5
        period_seconds    = 10
        failure_threshold = 5
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds   = 5
        period_seconds    = 30
        failure_threshold = 3
      }
    }
  }
}

# Inference Service
resource "google_cloud_run_v2_service" "inference_service" {
  name     = terraform.workspace == "default" ? "inference-service" : "inference-service-${terraform.workspace}"
  location = var.region

  template {
    service_account = var.service_account_email
    gpu_zonal_redundancy_disabled = "true"
    
    labels = merge(var.labels, {
      service = "inference"
    })
    
    scaling {
      min_instance_count = 0
      max_instance_count = var.inference_max_instances
    }

    containers {
      image = "${var.artifact_registry_repo_url}/inference-service:${var.inference_image_tag}"
      
      resources {
        limits = {
          cpu              = "4"
          memory           = "16Gi"
          "nvidia.com/gpu" = "1"
        }
        cpu_idle          = false
        startup_cpu_boost = true
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

      env {
        name  = "FIRESTORE_DB"
        value = var.firestore_database_name
      }

      ports {
        container_port = 8080
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        timeout_seconds   = 5
        period_seconds    = 30
        failure_threshold = 3
      }
    }

    node_selector {
      accelerator = "nvidia-l4"
    }
  }
}

# Training Job (Cloud Run Job)
resource "google_cloud_run_v2_job" "training_job" {
  name     = terraform.workspace == "default" ? "training-job" : "training-job-${terraform.workspace}"
  location = var.region
  launch_stage = "BETA"
  
  template {
    labels = merge(var.labels, {
      job-type = "training"
    })

    template {
      service_account = var.service_account_email
      
      containers {
        image = "${var.artifact_registry_repo_url}/training-job:latest"
        
        resources {
          limits = {
            cpu              = "8"
            memory           = "32Gi"
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
          name  = "GCS_EXPORT_FILES_BUCKET_NAME"
          value = var.files_bucket_name
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "FIRESTORE_DB"
          value = var.firestore_database_name
        }
      }

      node_selector {
        accelerator = "nvidia-l4"
      }

      gpu_zonal_redundancy_disabled = "true"
      
      max_retries = 0
      timeout     = "3600s"
    }
  }
}

# Export job
resource "google_cloud_run_v2_job" "export_job" {
  name     = terraform.workspace == "default" ? "export-job" : "export-job-${terraform.workspace}"
  location = var.region
  launch_stage = "BETA"
  
  template {
    labels = merge(var.labels, {
      job-type = "export"
    })

    template {
      service_account = var.service_account_email
      
      containers {
        image = "${var.artifact_registry_repo_url}/export-job:latest"
        
        resources {
          limits = {
            cpu              = "8"
            memory           = "32Gi"
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
          name = "GCS_EXPORT_FILES_BUCKET_NAME"
          value = var.files_bucket_name
        }

        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "FIRESTORE_DB"
          value = var.firestore_database_name
        }
      }

      node_selector {
        accelerator = "nvidia-l4"
      }

      gpu_zonal_redundancy_disabled = "true"
      
      max_retries = 0
      timeout     = "3600s"
    }
  }
}

# IAM bindings for public access
resource "google_cloud_run_service_iam_binding" "preprocessing_service_public" {
  location = google_cloud_run_v2_service.preprocessing_service.location
  service  = google_cloud_run_v2_service.preprocessing_service.name
  role     = "roles/run.invoker"
  members  = ["serviceAccount:${var.service_account_email}"]
}

resource "google_cloud_run_service_iam_binding" "training_service_public" {
  location = google_cloud_run_v2_service.training_service.location
  service  = google_cloud_run_v2_service.training_service.name
  role     = "roles/run.invoker"
  members  = ["serviceAccount:${var.service_account_email}"]
}

# NOTE: WE DID NOT MAKE A MISTAKE TO HAVE ALLUSERS HERE!!
# To avoid gateway timeout the frontend is no longer using inference through gateway.
# The service itself still does authentication so it's not a problem

resource "google_cloud_run_service_iam_binding" "inference_service_public" {
  location = google_cloud_run_v2_service.inference_service.location
  service  = google_cloud_run_v2_service.inference_service.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]  # Consider restricting this in production
}
