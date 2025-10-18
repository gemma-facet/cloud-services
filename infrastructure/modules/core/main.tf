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
    "cloudresourcemanager.googleapis.com",
    "vpcaccess.googleapis.com",
    "compute.googleapis.com"
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

# VPC Network
resource "google_compute_network" "vpc_network" {
  name                    = terraform.workspace == "default" ? "gemma-vpc" : "gemma-vpc-${terraform.workspace}"
  project                 = var.project_id
  auto_create_subnetworks = false

  depends_on = [google_project_service.required_apis]
}

# VPC Subnet
resource "google_compute_subnetwork" "vpc_subnet" {
  name          = terraform.workspace == "default" ? "gemma-subnet" : "gemma-subnet-${terraform.workspace}"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc_network.id
  ip_cidr_range = "10.0.0.0/28"
}

# VPC Access Connector
resource "google_vpc_access_connector" "vpc_connector" {
  name          = terraform.workspace == "default" ? "gemma-connector" : "gemma-connector-${terraform.workspace}"
  project       = var.project_id
  region        = var.region
  subnet {
    name = google_compute_subnetwork.vpc_subnet.name
  }
  machine_type = "e2-micro"
  min_instances = 2
  max_instances = 3
  depends_on = [google_project_service.required_apis]
}

# Cloud Router
resource "google_compute_router" "router" {
  name    = terraform.workspace == "default" ? "gemma-router" : "gemma-router-${terraform.workspace}"
  project = var.project_id
  region  = var.region
  network = google_compute_network.vpc_network.id

  depends_on = [google_project_service.required_apis]
}

# Static IP for NAT
resource "google_compute_address" "static_ip" {
  name    = terraform.workspace == "default" ? "gemma-nat-ip" : "gemma-nat-ip-${terraform.workspace}"
  project = var.project_id
  region  = var.region

  depends_on = [google_project_service.required_apis]
}

# Cloud NAT
resource "google_compute_router_nat" "nat" {
  name                               = terraform.workspace == "default" ? "gemma-nat" : "gemma-nat-${terraform.workspace}"
  project                            = var.project_id
  router                             = google_compute_router.router.name
  region                             = var.region
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"
  subnetwork {
    name                    = google_compute_subnetwork.vpc_subnet.id
    source_ip_ranges_to_nat = ["ALL_IP_RANGES"]
  }
  nat_ip_allocate_option             = "MANUAL_ONLY"
  nat_ips                            = [google_compute_address.static_ip.self_link]
}
