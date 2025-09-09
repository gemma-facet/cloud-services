output "firebase_project_id" {
  description = "Firebase project ID"
  value       = var.enable_firebase ? google_firebase_project.gemma_project[0].project : null
}

output "firebase_web_app_id" {
  description = "Firebase web app ID"
  value       = var.enable_firebase ? google_firebase_web_app.gemma_web_app[0].app_id : null
}

output "firebase_config" {
  description = "Firebase web app configuration"
  value       = var.enable_firebase ? data.google_firebase_web_app_config.gemma_web_app_config[0] : null
  sensitive   = true
}
