output "firebase_project_id" {
  description = "Firebase project ID"
  value       = google_firebase_project.gemma_project[0].project
}

output "firebase_web_app_id" {
  description = "Firebase web app ID"
  value       = google_firebase_web_app.gemma_web_app[0].app_id
}

output "firebase_config" {
  description = "Firebase web app configuration"
  value       = data.google_firebase_web_app_config.gemma_web_app_config[0]
}

output "firebase_api_key" {
  description = "Firebase API Key (NEXT_PUBLIC_FIREBASE_API_KEY)"
  value       = data.google_firebase_web_app_config.gemma_web_app_config[0].api_key
}

output "firebase_auth_domain" {
  description = "Firebase Auth Domain (NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN)"
  value       = data.google_firebase_web_app_config.gemma_web_app_config[0].auth_domain
}

output "firebase_storage_bucket" {
  description = "Firebase Storage Bucket (NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET)"
  value       = data.google_firebase_web_app_config.gemma_web_app_config[0].storage_bucket
}

output "firebase_messaging_sender_id" {
  description = "Firebase Messaging Sender ID (NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID)"
  value       = data.google_firebase_web_app_config.gemma_web_app_config[0].messaging_sender_id
}

output "firebase_app_id" {
  description = "Firebase App ID (NEXT_PUBLIC_FIREBASE_APP_ID)"
  value       = google_firebase_web_app.gemma_web_app[0].app_id
}
