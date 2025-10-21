# Gemma Fine-tuning Infrastructure Management
# Multi-environment deployment using Terraform workspaces

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Configuration
PROJECT_ID := $(shell gcloud config get-value project 2>/dev/null)
ENVIRONMENT := $(or $(ENV),staging)

# Validate environment
ifeq ($(filter $(ENVIRONMENT),staging production),)
$(error Environment must be 'staging' or 'production'. Use: make target ENV=staging)
endif

# Workspace mapping: staging -> staging, production -> default
WORKSPACE := $(if $(filter $(ENVIRONMENT),production),default,$(ENVIRONMENT))
# Get artifact registry repository name from Terraform output (after deploy-core)
# This is the full registry path: region-docker.pkg.dev/project-id/repo-name
ARTIFACT_REGISTRY_URL := $(shell terraform output -raw artifact_registry_repo_url 2>/dev/null || echo "")

.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Deploy Facet AI's fine-tuning backend infrastructure$(NC)"
	@echo ""
	@echo "$(YELLOW)Current Environment: $(ENVIRONMENT) (workspace: $(WORKSPACE))$(NC)"
	@echo "$(YELLOW)Change environment with: ENV=production make <target>$(NC)"
	@echo ""
	@echo "$(YELLOW)Quick Start:$(NC)"
	@echo "  1. make init ENV=staging            # Initialize staging workspace"
	@echo "  2. make full-deploy ENV=staging     # Deploy to staging"
	@echo "  3. make full-deploy ENV=production  # Deploy to production"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Prerequisites check
.PHONY: check
check: ## Check prerequisites
	@echo "$(BLUE)Checking prerequisites for $(ENVIRONMENT) environment...$(NC)"
	@command -v terraform >/dev/null 2>&1 || { echo "$(RED)Error: terraform not installed$(NC)"; exit 1; }
	@command -v gcloud >/dev/null 2>&1 || { echo "$(RED)Error: gcloud CLI not installed$(NC)"; exit 1; }
	@test -n "$(PROJECT_ID)" || { echo "$(RED)Error: GCP project not set. Run: gcloud config set project YOUR_PROJECT$(NC)"; exit 1; }
	@gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q . || { echo "$(RED)Error: not authenticated. Run: gcloud auth login$(NC)"; exit 1; }
	@test -f environments/$(ENVIRONMENT).tfvars || { echo "$(RED)Error: environments/$(ENVIRONMENT).tfvars not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Prerequisites check passed for $(ENVIRONMENT)$(NC)"

# Build all containers using Cloud Build
.PHONY: build
build: check ## Build all container images with Cloud Build for current environment
	@echo "$(BLUE)Building all containers for $(ENVIRONMENT) environment...$(NC)"
	@if [ -z "$(ARTIFACT_REGISTRY_URL)" ]; then \
		echo "$(RED)Error: Artifact Registry URL not found in Terraform state$(NC)"; \
		echo "$(YELLOW)Please run 'make deploy-core ENV=$(ENVIRONMENT)' first to create the registry$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ Using Artifact Registry: $(ARTIFACT_REGISTRY_URL)$(NC)"
	@echo "$(YELLOW)Building preprocessing service...$(NC)"
	@cd ../preprocessing && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --substitutions=_ARTIFACT_REGISTRY_URL=$(ARTIFACT_REGISTRY_URL) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building training service...$(NC)"
	@cd ../training && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --substitutions=_ARTIFACT_REGISTRY_URL=$(ARTIFACT_REGISTRY_URL) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building inference service...$(NC)"
	@cd ../inference && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --substitutions=_ARTIFACT_REGISTRY_URL=$(ARTIFACT_REGISTRY_URL) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building training job...$(NC)"
	@cd ../jobs/training-job && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --substitutions=_ARTIFACT_REGISTRY_URL=$(ARTIFACT_REGISTRY_URL) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building export job...$(NC)"
	@cd ../jobs/export-job && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --substitutions=_ARTIFACT_REGISTRY_URL=$(ARTIFACT_REGISTRY_URL) --ignore-file=.gcloudignore
	@echo "$(GREEN)✓ All containers built successfully for $(ENVIRONMENT)$(NC)"

# Terraform operations
.PHONY: init
init: check ## Initialize Terraform and setup workspace
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	@terraform init $(if $(UPGRADE),-upgrade)
	@echo "$(YELLOW)Setting up workspace: $(WORKSPACE) for $(ENVIRONMENT)...$(NC)"
	@terraform workspace select $(WORKSPACE) 2>/dev/null || terraform workspace new $(WORKSPACE)
	@echo "$(GREEN)✓ Using Terraform workspace: $(WORKSPACE)$(NC)"

.PHONY: plan
plan: init ## Plan infrastructure changes for current environment
	@echo "$(BLUE)Planning infrastructure for $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@terraform plan -var-file=environments/$(ENVIRONMENT).tfvars

.PHONY: deploy
deploy: init ## Deploy infrastructure to current environment
	@echo "$(YELLOW)Deploying infrastructure to $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@terraform apply -var-file=environments/$(ENVIRONMENT).tfvars
	@echo "$(GREEN)✓ Infrastructure deployed successfully to $(ENVIRONMENT)$(NC)"
	@$(MAKE) output

.PHONY: deploy-core
deploy-core: init ## Deploy core infrastructure (APIs, registry, storage)
	@echo "$(YELLOW)Deploying core infrastructure to $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@echo "$(BLUE)Stage 1: APIs, registry, storage, firebase$(NC)"
	@terraform apply -var-file=environments/$(ENVIRONMENT).tfvars \
		-target=module.core \
		-target=module.storage \
		-target=module.firebase
	@echo "$(GREEN)✓ Core infrastructure deployed to $(ENVIRONMENT)$(NC)"

.PHONY: deploy-services
deploy-services: init ## Deploy Cloud Run services and API Gateway (requires images)
	@echo "$(YELLOW)Deploying Cloud Run services and API Gateway to $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@echo "$(BLUE)Stage 2: Cloud Run services and API Gateway$(NC)"
	@terraform apply -var-file=environments/$(ENVIRONMENT).tfvars \
		-target=module.compute \
		-target=module.api_gateway
	@echo "$(GREEN)✓ Services deployed successfully to $(ENVIRONMENT)$(NC)"
	@$(MAKE) output

.PHONY: plan-core
plan-core: init ## Plan core infrastructure only
	@echo "$(BLUE)Planning core infrastructure for $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@terraform plan -var-file=environments/$(ENVIRONMENT).tfvars \
		-target=module.core \
		-target=module.storage \
		-target=module.firebase

.PHONY: plan-services
plan-services: init ## Plan Cloud Run services only
	@echo "$(BLUE)Planning Cloud Run services for $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@terraform plan -var-file=environments/$(ENVIRONMENT).tfvars \
		-target=module.compute \
		-target=module.api_gateway

.PHONY: full-deploy
full-deploy: ## Full deployment: core -> build -> services
	@echo "$(BLUE)Starting full deployment to $(ENVIRONMENT) (workspace: $(WORKSPACE))...$(NC)"
	@$(MAKE) deploy-core
	@$(MAKE) build
	@$(MAKE) deploy-services
	@echo "$(GREEN)✓ Full deployment completed for $(ENVIRONMENT)!$(NC)"

.PHONY: output
output: ## Show infrastructure outputs
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Copy the following values to your frontend .env file:$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(YELLOW)API_GATEWAY_URL$(NC)=$(GREEN)$$(terraform output -raw api_gateway_url 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)INFERENCE_SERVICE_URL$(NC)=$(GREEN)$$(terraform output -raw inference_service_url 2>/dev/null || echo 'N/A')$(NC)"
	@echo ""
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_API_KEY$(NC)=$(GREEN)$$(terraform output -raw firebase_api_key 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN$(NC)=$(GREEN)$$(terraform output -raw firebase_auth_domain 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_PROJECT_ID$(NC)=$(GREEN)$$(terraform output -raw firebase_project_id 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET$(NC)=$(GREEN)$$(terraform output -raw firebase_storage_bucket 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID$(NC)=$(GREEN)$$(terraform output -raw firebase_messaging_sender_id 2>/dev/null || echo 'N/A')$(NC)"
	@echo "$(YELLOW)NEXT_PUBLIC_FIREBASE_APP_ID$(NC)=$(GREEN)$$(terraform output -raw firebase_app_id 2>/dev/null || echo 'N/A')$(NC)"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Backend Infrastructure (Internal)$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(YELLOW)Preprocessing Service:$(NC)   $$(terraform output -raw preprocessing_service_url 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Training Service:$(NC)        $$(terraform output -raw training_service_url 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Data Bucket:$(NC)             $$(terraform output -raw data_bucket_name 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Export Bucket:$(NC)           $$(terraform output -raw export_bucket_name 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Files Bucket:$(NC)            $$(terraform output -raw files_bucket_name 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Config Bucket:$(NC)           $$(terraform output -raw config_bucket_name 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Artifact Registry:$(NC)       $$(terraform output -raw artifact_registry_repo_url 2>/dev/null || echo 'N/A')"
	@echo "$(YELLOW)Service Account:$(NC)         $$(terraform output -raw service_account_email 2>/dev/null || echo 'N/A')"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"

.PHONY: destroy
destroy: check init ## Destroy all infrastructure for current environment
	@echo "$(RED)WARNING: This will destroy ALL $(ENVIRONMENT) infrastructure (workspace: $(WORKSPACE))!$(NC)"
	@read -p "Type 'DESTROY' to confirm: " confirm && [ "$$confirm" = "DESTROY" ]
	@terraform destroy -var-file=environments/$(ENVIRONMENT).tfvars -auto-approve

.PHONY: workspace-list
workspace-list: ## List all Terraform workspaces
	@echo "$(BLUE)Available Terraform workspaces:$(NC)"
	@terraform workspace list

.PHONY: workspace-show
workspace-show: ## Show current Terraform workspace
	@echo "$(BLUE)Current workspace:$(NC) $(shell terraform workspace show)"
	@echo "$(BLUE)Environment:$(NC) $(ENVIRONMENT)"

.PHONY: show-config
show-config: ## Show current configuration values
	@echo "$(BLUE)Configuration for $(ENVIRONMENT) environment:$(NC)"
	@echo "  $(YELLOW)Project ID:$(NC)          $(PROJECT_ID)"
	@echo "  $(YELLOW)Workspace:$(NC)           $(WORKSPACE)"
	@if [ -n "$(ARTIFACT_REGISTRY_URL)" ]; then \
		echo "  $(YELLOW)Artifact Registry:$(NC)   $(GREEN)$(ARTIFACT_REGISTRY_URL)$(NC) $(GREEN)✓$(NC)"; \
	else \
		echo "  $(YELLOW)Artifact Registry:$(NC)   $(RED)Not available (run deploy-core first)$(NC)"; \
	fi

.PHONY: clean
clean: ## Clean temporary files
	@rm -f *.tfplan *.tfstate.backup terraform.tfstate*
