terraform {
  required_version = ">= 1.5"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.1.1"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 7.1.1"
    }
  }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Core Module
module "core" {
  source = "./modules/core"
  
  project_id  = var.project_id
  region      = var.region
  labels      = var.labels
}

# Storage Module
module "storage" {
  source = "./modules/storage"
  
  project_id         = var.project_id
  region             = var.region
  labels             = var.labels
  data_bucket_name   = terraform.workspace == "default" ? "${var.project_id}-datasets" : "${var.project_id}-datasets-${terraform.workspace}"
  export_bucket_name = terraform.workspace == "default" ? "${var.project_id}-models" : "${var.project_id}-models-${terraform.workspace}"
  config_bucket_name = terraform.workspace == "default" ? "${var.project_id}-configs" : "${var.project_id}-configs-${terraform.workspace}"
  files_bucket_name  = terraform.workspace == "default" ? "${var.project_id}-files" : "${var.project_id}-files-${terraform.workspace}"

  depends_on = [module.core]
}

# Firebase Module
# module "firebase" {
#   source = "./modules/firebase"
  
#   project_id        = var.project_id
#   labels            = var.labels
#   enable_firebase   = true
#   firebase_location = "us-central"
  
#   depends_on = [module.core]
# }

# Cloud Run Services Module
module "compute" {
  source = "./modules/compute"
  
  project_id                   = var.project_id
  region                      = var.region
  labels                      = var.labels
  service_account_email       = module.core.service_account_email
  artifact_registry_repo_url  = module.core.artifact_registry_repo_url
  data_bucket_name           = module.storage.data_bucket_name
  export_bucket_name         = module.storage.export_bucket_name
  config_bucket_name         = module.storage.config_bucket_name
  files_bucket_name          = module.storage.files_bucket_name
  training_image_tag         = "latest"
  preprocessing_image_tag    = "latest"
  inference_image_tag        = "latest"
  preprocessing_max_instances = 10
  training_max_instances      = 10
  inference_max_instances     = 3
  
  depends_on = [module.core, module.storage]
}

# API Gateway Module
module "api_gateway" {
  source = "./modules/api_gateway"
  
  project_id               = var.project_id
  region                  = var.region
  labels                  = var.labels
  service_account_email   = module.core.service_account_email
  api_config_id           = "gemma-api"
  preprocessing_service_url = module.compute.preprocessing_service_url
  training_service_url     = module.compute.training_service_url
  inference_service_url    = module.compute.inference_service_url
  
  depends_on = [module.compute]
}