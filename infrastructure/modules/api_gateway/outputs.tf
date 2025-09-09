output "api_gateway_url" {
  description = "URL of the API Gateway"
  value       = google_api_gateway_gateway.gemma_gateway.default_hostname
}

output "api_gateway_id" {
  description = "ID of the API Gateway"
  value       = google_api_gateway_gateway.gemma_gateway.gateway_id
}

output "api_config_id" {
  description = "ID of the API Config"
  value       = google_api_gateway_api_config.gemma_api_config.api_config_id
}
