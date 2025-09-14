# Storage module: GCS buckets and Firestore

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

variable "data_bucket_name" {
  description = "Name for the data bucket"
  type        = string
}

variable "export_bucket_name" {
  description = "Name for the export bucket"
  type        = string
}

variable "files_bucket_name" {
  description = "Name for the files bucket"
  type        = string
}

variable "config_bucket_name" {
  description = "Name for the config bucket"
  type        = string
}

# Data bucket for datasets
resource "google_storage_bucket" "data_bucket" {
  name     = var.data_bucket_name
  location = var.region
  
  labels = merge(var.labels, {
    bucket-type = "data"
  })

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90  # Keep data longer for training purposes
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                   = 30
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  uniform_bucket_level_access = true
  
  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

# Export bucket for trained models
resource "google_storage_bucket" "export_bucket" {
  name     = var.export_bucket_name
  location = var.region
  
  labels = merge(var.labels, {
    bucket-type = "export"
  })

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 180  # Keep models longer
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                   = 60
      matches_storage_class = ["STANDARD"]
    }
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  uniform_bucket_level_access = true
}

# NOTE: This is imported directly by terraformer, it could be wrong and might need update??
resource "google_storage_bucket" "files_bucket" {
  default_event_based_hold = "false"
  enable_object_retention  = "false"
  force_destroy            = "false"

  hierarchical_namespace {
    enabled = "false"
  }

  location                 = "US"
  name                     = var.files_bucket_name
  public_access_prevention = "inherited"
  requester_pays           = "false"
  rpo                      = "DEFAULT"

  soft_delete_policy {
    retention_duration_seconds = "604800"
  }

  storage_class               = "STANDARD"
  uniform_bucket_level_access = "true"
}

# Config bucket for training configurations
resource "google_storage_bucket" "config_bucket" {
  name     = var.config_bucket_name
  location = var.region
  
  labels = merge(var.labels, {
    bucket-type = "config"
  })

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

  uniform_bucket_level_access = true
}

# Firestore database
resource "google_firestore_database" "gemma_database" {
  project                           = var.project_id
  name                             = "(default)"
  location_id                      = var.region
  type                            = "FIRESTORE_NATIVE"
  concurrency_mode                = "OPTIMISTIC"
  app_engine_integration_mode     = "DISABLED"
  point_in_time_recovery_enablement = "POINT_IN_TIME_RECOVERY_ENABLED"
  delete_protection_state          = "DELETE_PROTECTION_ENABLED"
}

# NOTE: We did not enable indexing in the original project so we will not turn this on for now
# I do NOT believe we require composite indexes for our current queries, firestore automatically create simple single-field indexes already

# Firestore indexes for performance
# resource "google_firestore_index" "training_jobs_index" {
#   project    = var.project_id
#   database   = google_firestore_database.gemma_database.name
#   collection = "training_jobs"

#   fields {
#     field_path = "job_id"
#     order      = "ASCENDING"
#   }

#   fields {
#     field_path = "created_at"
#     order      = "DESCENDING"
#   }
# }

# resource "google_firestore_index" "datasets_index" {
#   project    = var.project_id
#   database   = google_firestore_database.gemma_database.name
#   collection = "processed_datasets"

#   fields {
#     field_path = "processed_dataset_id"
#     order      = "ASCENDING"
#   }

#   fields {
#     field_path = "created_at"
#     order      = "DESCENDING"
#   }
# }

resource "google_firestore_document" "training_job_example" {
  provider    = google-beta
  project     = google_firestore_database.gemma_database.project
  collection  = "training_jobs"
  document_id = "training_job_example"
  fields      = "{\"job_id\":{\"stringValue\":\"training_job_example\"},\"job_name\":{\"stringValue\":\"SAMPLE FAKE JOB\"}}"
}

resource "google_firestore_document" "processed_dataset_example" {
  provider    = google-beta
  project     = google_firestore_database.gemma_database.project
  collection  = "processed_datasets"
  document_id = "processed_dataset_example"
  fields      = "{\"processed_dataset_id\":{\"stringValue\":\"processed_dataset_example\"},\"dataset_name\":{\"stringValue\":\"SAMPLE FAKE DATASET\"}}"
}

resource "google_firestore_document" "dataset_example" {
  provider    = google-beta
  project     = google_firestore_database.gemma_database.project
  collection  = "datasets"
  document_id = "uploaded_dataset_example"
  fields      = "{\"uploaded_dataset_id\":{\"stringValue\":\"uploaded_dataset_example\"},\"dataset_name\":{\"stringValue\":\"SAMPLE FAKE UPLOADED DATASET\"}}"
}