# Core infrastructure: APIs, IAM, Artifact Registry

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

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "firestore.googleapis.com",
    "firebase.googleapis.com",
    "apigateway.googleapis.com",
    "servicecontrol.googleapis.com",
    "servicemanagement.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value

  disable_on_destroy = false
  
  timeouts {
    create = "30m"
    update = "40m"
  }
}

# Service Account for all Gemma services
resource "google_service_account" "gemma_services" {
  account_id   = terraform.workspace == "default" ? "gemma-services" : "gemma-services-${terraform.workspace}"
  display_name = "Gemma Services Account (${terraform.workspace})"
  description  = "Service account for Gemma fine-tuning services (${terraform.workspace})"

  depends_on = [google_project_service.required_apis]
}

# IAM roles for the service account
resource "google_project_iam_member" "gemma_services_permissions" {
  for_each = toset([
    "roles/storage.admin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/datastore.user",
    "roles/firebase.admin",
    "roles/run.developer",
    "roles/run.invoker",
    "roles/artifactregistry.reader"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gemma_services.email}"
}

# Artifact Registry Repository
resource "google_artifact_registry_repository" "gemma_services" {
  location      = var.region
  repository_id = terraform.workspace == "default" ? "gemma-fine-tuning" : "gemma-fine-tuning-${terraform.workspace}"
  description   = "Docker repository for Gemma fine-tuning services (${terraform.workspace})"
  format        = "DOCKER"
  
  labels = var.labels

  depends_on = [google_project_service.required_apis]
}
