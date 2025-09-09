# Gemma Fine-tuning Infrastructure Management
# Production-only deployment with proper build order

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Configuration
PROJECT_ID := $(shell gcloud config get-value project 2>/dev/null)

.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Gemma Fine-tuning Infrastructure - Production Deployment$(NC)"
	@echo ""
	@echo "$(YELLOW)Workflow Options:$(NC)"
	@echo "  1. make build           # Build all containers with Cloud Build"
	@echo "  2. make deploy          # Deploy all infrastructure"
	@echo "  3. make full            # Build + Deploy (complete workflow)"
	@echo ""
	@echo "$(YELLOW)Staged Deployment:$(NC)"
	@echo "  1. make deploy-core     # Deploy APIs, registry, storage first"
	@echo "  2. make build           # Build container images"
	@echo "  3. make deploy-services # Deploy Cloud Run services + API Gateway"
	@echo ""
	@echo "$(YELLOW)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Prerequisites check
.PHONY: check
check: ## Check prerequisites
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v terraform >/dev/null 2>&1 || { echo "$(RED)Error: terraform not installed$(NC)"; exit 1; }
	@command -v gcloud >/dev/null 2>&1 || { echo "$(RED)Error: gcloud CLI not installed$(NC)"; exit 1; }
	@test -n "$(PROJECT_ID)" || { echo "$(RED)Error: GCP project not set. Run: gcloud config set project YOUR_PROJECT$(NC)"; exit 1; }
	@gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q . || { echo "$(RED)Error: not authenticated. Run: gcloud auth login$(NC)"; exit 1; }
	@test -f terraform.tfvars || { echo "$(RED)Error: terraform.tfvars not found$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Prerequisites check passed$(NC)"

# Build all containers using Cloud Build
.PHONY: build
build: check ## Build all container images with Cloud Build
	@echo "$(BLUE)Building all containers with Cloud Build...$(NC)"
	@echo "$(YELLOW)Building preprocessing service...$(NC)"
	@cd ../preprocessing && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building training service...$(NC)"
	@cd ../training && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building inference service...$(NC)"
	@cd ../inference && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --ignore-file=.gcloudignore
	@echo "$(YELLOW)Building training job...$(NC)"
	@cd ../jobs/training-job && gcloud builds submit --config cloudbuild.yaml --project=$(PROJECT_ID) --ignore-file=.gcloudignore
	@echo "$(GREEN)✓ All containers built successfully$(NC)"

# Terraform operations
.PHONY: init
init: check ## Initialize Terraform
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	# Support optional upgrade: make init UPGRADE=true
	@terraform init $(if $(UPGRADE),-upgrade)

.PHONY: plan
plan: ## Plan infrastructure changes
	@echo "$(BLUE)Planning infrastructure...$(NC)"
	@terraform plan -var-file=terraform.tfvars

.PHONY: deploy
deploy: ## Deploy infrastructure
	@echo "$(YELLOW)Deploying infrastructure to PRODUCTION...$(NC)"
	@terraform apply -var-file=terraform.tfvars
	@echo "$(GREEN)✓ Infrastructure deployed successfully$(NC)"
	@$(MAKE) output

.PHONY: deploy-core
deploy-core: ## Deploy core infrastructure (APIs, registry, storage, firebase)
	@echo "$(YELLOW)Deploying core infrastructure...$(NC)"
	@echo "$(BLUE)Stage 1: APIs, registry, storage, and firebase$(NC)"
	@terraform apply -var-file=terraform.tfvars \
		-target=module.core \
		-target=module.storage \
		-target=module.firebase
	@echo "$(GREEN)✓ Core infrastructure deployed$(NC)"

.PHONY: deploy-services
deploy-services: ## Deploy Cloud Run services and API Gateway (requires images)
	@echo "$(YELLOW)Deploying Cloud Run services and API Gateway...$(NC)"
	@echo "$(BLUE)Stage 2: Cloud Run services and API Gateway$(NC)"
	@terraform apply -var-file=terraform.tfvars \
		-target=module.compute \
		-target=module.api_gateway
	@echo "$(GREEN)✓ Services deployed successfully$(NC)"
	@$(MAKE) output

.PHONY: plan-core
plan-core: ## Plan core infrastructure only
	@echo "$(BLUE)Planning core infrastructure...$(NC)"
	@terraform plan -var-file=terraform.tfvars \
		-target=module.core \
		-target=module.storage \
		-target=module.firebase

.PHONY: plan-services
plan-services: ## Plan Cloud Run services only
	@echo "$(BLUE)Planning Cloud Run services...$(NC)"
	@terraform plan -var-file=terraform.tfvars \
		-target=module.compute \
		-target=module.api_gateway

.PHONY: full-deploy
full-deploy: ## Full deployment: core -> build -> services
	@echo "$(BLUE)Starting full deployment...$(NC)"
	@$(MAKE) deploy-core
	@$(MAKE) build
	@$(MAKE) deploy-services
	@echo "$(GREEN)✓ Full deployment completed!$(NC)"

.PHONY: output
output: ## Show infrastructure outputs
	@terraform output -json | jq -r 'to_entries[] | "\(.key): \(.value.value)"' 2>/dev/null || terraform output

.PHONY: destroy
destroy: check init ## Destroy all infrastructure
	@echo "$(RED)WARNING: This will destroy ALL production infrastructure!$(NC)"
	@read -p "Type 'DESTROY' to confirm: " confirm && [ "$$confirm" = "DESTROY" ]
	@terraform destroy -var-file=terraform.tfvars -auto-approve

.PHONY: clean
clean: ## Clean temporary files
	@rm -f *.tfplan *.tfstate.backup terraform.tfstate*
