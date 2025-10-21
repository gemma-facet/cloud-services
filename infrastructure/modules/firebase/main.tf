variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
}

variable "enable_firebase" {
  description = "Enable Firebase services"
  type        = bool
  default     = true
}

variable "firebase_location" {
  description = "Location for Firebase resources"
  type        = string
  default     = "us-central"
}

# Firebase project
resource "google_firebase_project" "gemma_project" {
  count      = var.enable_firebase ? 1 : 0
  provider   = google-beta
  project    = var.project_id
}

# Firebase Authentication
resource "google_identity_platform_config" "auth_config" {
  count    = var.enable_firebase ? 1 : 0
  provider = google-beta
  project  = var.project_id
  
  sign_in {
    allow_duplicate_emails = false
    
    # anonymous {
    #   enabled = true
    # }
    
    email {
      enabled           = true
      password_required = true
    }
  }

  authorized_domains = [
    "localhost",
    "${google_firebase_project.gemma_project[0].project}.firebaseapp.com",
    "${google_firebase_project.gemma_project[0].project}.web.app",
    "${google_firebase_project.gemma_project[0].project}.vercel.app",
  ]
  
  depends_on = [google_firebase_project.gemma_project]
}

# NOTE: The following setup for google IDP is not used, we create it on the console because it's faster that way

# Default supported identity providers
# resource "google_identity_platform_default_supported_idp_config" "google_config" {
#   count         = var.enable_firebase ? 1 : 0
#   project       = var.project_id
#   idp_id        = "google.com"
#   client_id     = "your-google-oauth-client-id"  # Update this
#   client_secret = "your-google-oauth-client-secret"  # Update this
#   enabled       = true
  
#   depends_on = [google_identity_platform_config.auth_config]
# }

# Firebase Web App
resource "google_firebase_web_app" "gemma_web_app" {
  count        = var.enable_firebase ? 1 : 0
  provider     = google-beta
  project      = var.project_id
  display_name = "Gemma Fine-tuning Web App"
  
  depends_on = [google_firebase_project.gemma_project]
}

# Firebase Web App config
data "google_firebase_web_app_config" "gemma_web_app_config" {
  count      = var.enable_firebase ? 1 : 0
  provider   = google-beta
  web_app_id = google_firebase_web_app.gemma_web_app[0].app_id
}
