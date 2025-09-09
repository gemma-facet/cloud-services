output "preprocessing_service_url" {
  description = "URL of the preprocessing service"
  value       = google_cloud_run_v2_service.preprocessing_service.uri
}

output "training_service_url" {
  description = "URL of the training service"
  value       = google_cloud_run_v2_service.training_service.uri
}

output "inference_service_url" {
  description = "URL of the inference service"
  value       = google_cloud_run_v2_service.inference_service.uri
}

output "training_job_name" {
  description = "Name of the training job"
  value       = google_cloud_run_v2_job.training_job.name
}

output "preprocessing_service_name" {
  description = "Name of the preprocessing service"
  value       = google_cloud_run_v2_service.preprocessing_service.name
}

output "training_service_name" {
  description = "Name of the training service"
  value       = google_cloud_run_v2_service.training_service.name
}

output "inference_service_name" {
  description = "Name of the inference service"
  value       = google_cloud_run_v2_service.inference_service.name
}
