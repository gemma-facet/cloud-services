output "data_bucket_name" {
  description = "Name of the data bucket"
  value       = google_storage_bucket.data_bucket.name
}

output "data_bucket_url" {
  description = "URL of the data bucket"
  value       = google_storage_bucket.data_bucket.url
}

output "export_bucket_name" {
  description = "Name of the export bucket"
  value       = google_storage_bucket.export_bucket.name
}

output "export_bucket_url" {
  description = "URL of the export bucket"
  value       = google_storage_bucket.export_bucket.url
}

output "config_bucket_name" {
  description = "Name of the config bucket"
  value       = google_storage_bucket.config_bucket.name
}

output "config_bucket_url" {
  description = "URL of the config bucket"
  value       = google_storage_bucket.config_bucket.url
}

output "firestore_database_name" {
  description = "Name of the Firestore database"
  value       = google_firestore_database.gemma_database.name
}
