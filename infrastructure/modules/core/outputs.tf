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

output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.vpc_network.name
}

output "vpc_subnet_name" {
  description = "Name of the VPC subnetwork"
  value       = google_compute_subnetwork.vpc_subnet.name
}

output "vpc_connector_id" {
  description = "ID of the VPC Access Connector"
  value       = google_vpc_access_connector.vpc_connector.id
}
