output "preprocessing_service_url" {
  description = "URL of the preprocessing service"
  value       = module.compute.preprocessing_service_url
}

output "training_service_url" {
  description = "URL of the training service"
  value       = module.compute.training_service_url
}

output "inference_service_url" {
  description = "URL of the inference service"
  value       = module.compute.inference_service_url
}

output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = module.api_gateway.api_gateway_url
}

output "data_bucket_name" {
  description = "Name of the data bucket"
  value       = module.storage.data_bucket_name
}

output "export_bucket_name" {
  description = "Name of the export bucket"
  value       = module.storage.export_bucket_name
}

output "files_bucket_name" {
  description = "Name of the files bucket"
  value       = module.storage.files_bucket_name
}

output "config_bucket_name" {
  description = "Name of the config bucket"
  value       = module.storage.config_bucket_name
}

output "artifact_registry_repo_url" {
  description = "URL of the Artifact Registry repository"
  value       = module.core.artifact_registry_repo_url
}

output "service_account_email" {
  description = "Email of the service account"
  value       = module.core.service_account_email
}

# Firebase Frontend Configuration
output "firebase_api_key" {
  description = "Firebase API Key (NEXT_PUBLIC_FIREBASE_API_KEY)"
  value       = module.firebase.firebase_api_key
}

output "firebase_auth_domain" {
  description = "Firebase Auth Domain (NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN)"
  value       = module.firebase.firebase_auth_domain
}

output "firebase_project_id" {
  description = "Firebase Project ID (NEXT_PUBLIC_FIREBASE_PROJECT_ID)"
  value       = module.firebase.firebase_project_id
}

output "firebase_storage_bucket" {
  description = "Firebase Storage Bucket (NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET)"
  value       = module.firebase.firebase_storage_bucket
}

output "firebase_messaging_sender_id" {
  description = "Firebase Messaging Sender ID (NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID)"
  value       = module.firebase.firebase_messaging_sender_id
}

output "firebase_app_id" {
  description = "Firebase App ID (NEXT_PUBLIC_FIREBASE_APP_ID)"
  value       = module.firebase.firebase_app_id
}
