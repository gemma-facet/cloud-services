output "service_account_email" {
  description = "Email of the service account"
  value       = google_service_account.gemma_services.email
}

output "service_account_id" {
  description = "ID of the service account"
  value       = google_service_account.gemma_services.id
}

output "artifact_registry_repo_url" {
  description = "Full URL of the Artifact Registry repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.gemma_services.repository_id}"
}

output "artifact_registry_repo_name" {
  description = "Name of the Artifact Registry repository"
  value       = google_artifact_registry_repository.gemma_services.name
}
