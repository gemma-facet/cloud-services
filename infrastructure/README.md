# IaC for Gemma Fine-tuning Services

You can use this Terraform setup to quickly deploy the Gemma fine-tuning services infrastructure on your own Google Cloud Platform project. This setup supports **multi-environment deployments** using Terraform workspaces.

1. This only works with Google Cloud Platform (GCP). Open issue / PR if you want to add support for AWS/Azure!

2. This requires installation of [Terraform](https://www.terraform.io/downloads.html) and [gcloud CLI](https://cloud.google.com/sdk/docs/install).

3. This does not create a GCP project for you. You should create one from the Console and make sure you have the correct IAM role (Owner or Editor) to create resources.

4. As the writing of this, certain APIs require billing to be enabled, so it's recommended to enable billing on your project first. We might consider adding 3 and 4 to the Terraform setup in the future.

> [!NOTE] We provide `Makefile` (see below) to simplify the entire deployment process. We do not recommend using `terraform` command manually because the deployment takes multiple stages (setup + build + deploy), modules, and workspaces.

## Environment Support

This infrastructure uses **Terraform workspaces** for environment isolation:

- **Staging**: Uses `staging` workspace - completely isolated state
- **Production**: Uses `default` workspace - completely isolated state

Both environments use the same resource names but are deployed to separate Terraform states, ensuring complete isolation. Use `ENV=staging` or `ENV=production` to specify the target environment.

## Quickstart

1. **Clone the repository**

```bash
git clone https://github.com/gemma-facet/cloud-services
cd cloud-services/infrastructure
```

2. **Authenticate with GCP**

First create a GCP project from the console. We don't automate this step because for beginners, you still need the console to setup billing, and for advanced users, creating projects is pretty straightforward.

```bash
gcloud auth login
# Set your project
gcloud config set project <your-project-id>
```

3. **Edit environment variables**

```bash
# Edit staging configuration
cp environments/staging.tfvars.example environments/staging.tfvars
vim environments/staging.tfvars

# Edit production configuration
cp environments/production.tfvars.example environments/production.tfvars
vim environments/production.tfvars
```

4. **Deploy infrastructure**

```bash
# Use ENV=production or ENV=staging
make init ENV=staging
make full-deploy ENV=staging
make output ENV=staging
```

`make full-deploy` performs the following steps:

1. Deploy core infrastructure (APIs, IAM, Artifact Registry, Storage, Firebase) with `make deploy-core`
2. Build and push docker images to Artifact Registry with `make build`
3. Deploy microservices and sets up networking (Cloud Run and API Gateway) with `make deploy-services`

4. **Use the results**

After running `make output`, the first section contains API endpoints and firebase configuration values that you can **directly copy and paste into your frontend `.env` file**. You do not need to access any GCP or firebase console unless you want to setup Google OAuth which is optional.

## How this works

### Workspace Management

> [!NOTE]
> We use the project id you provided to create names for resources such as storage buckets and service account to ensure that they are globally unique. Terraform workspaces ensure staging and production deployments are completely isolated.

**Key Benefits:**

- **Complete Isolation**: Separate Terraform state files prevent any cross-environment conflicts
- **Same Resource Names**: Both environments use identical resource names (e.g., `training-service`, `gemma-api`)
- **Zero Production Risk**: Staging changes cannot accidentally affect production
- **Simple Management**: Use `ENV=staging` or `ENV=production` to switch environments

### Cloud Build Integration

The make commands will automatically trigger Cloud Build as a build system using `cloudbuild.yaml` in each service directory to build and push docker images to the Artifact Registry for that workspace. ONLY builds are managed by cloud build, deployments are still managed by terraform. This is in contrast to manual deployment during dev, which can be done using `cloudbuild.dev.yaml` in each service directory.

For manual builds (this only works for staging environment):

```bash
gcloud builds submit --config cloudbuild.dev.yaml --ignore-file .gcloudignore .
```

### Staged Deployment

Terraform is not capable of handling build steps such as building docker images. Therefore, we split the deployment into multiple stages to use Cloud Build for building images and Terraform for deploying infrastructure. Having separate `make deploy-core` and `make deploy-services` commands allows for more flexibility during development, for example avoiding the need to rebuilding images when only changes are made to the infrastructure.

## Available Commands

```bash
make help          # Show all available commands with current environment

# Environment Selection
make <command> ENV=staging     # Deploy to staging workspace
make <command> ENV=production  # Deploy to production workspace (default)

# Prerequisites
make init          # Initialize Terraform and setup workspace
make check         # Check prerequisites

# Build & Deploy
make build         # Build all containers for current environment
make deploy        # Deploy infrastructure with Terraform
make deploy-core   # Deploy core infrastructure only
make deploy-services # Deploy microservices only
make full-deploy   # Complete workflow: core + build + services

# Management
make plan          # Plan infrastructure changes
make plan-core     # Plan core infrastructure changes
make plan-services # Plan microservices changes
make output        # Show infrastructure outputs
make destroy       # Destroy all infrastructure
make workspace-list # List all Terraform workspaces
make workspace-show # Show current workspace
```

The infrastructure is organized into modular components:

- **Core Module**: APIs, IAM, Artifact Registry
- **Storage Module**: GCS buckets, Firestore database
- **Firebase Module**: Firebase App, Auth and identity management
- **Compute Module**: Cloud Run services and jobs
- **API Gateway Module**: Unified API access point

## Importing current setup

This is meant for first time deployment. If you run into any issues with resources already existing, you should figure out your own way, perhaps by importing them first so that Terraform can manage them.

For example, you can try commands like

```bash
terraform import module.compute.google_cloud_run_v2_service.preprocessing_service \
  projects/<your-project-id>/locations/<your-region>/services/preprocessing-service
```

## Known Limitations

- We currently do not use terraform to setup firestore authentication because enabling Google provider requires more effort that way. You STILL need to go to the console to look up the OAuth ID, etc, and will need to use cloud secrets. Currently the config sets up everything except the Google IDP, which you need to manually enable by going to Firebase Console > Authentication > Sign-in method > Google > Enable.

- You may run into an issue saying your project is not a quota project and doesn't allow you to turn on the identity management API. To fix this, go to IAM & Admin > Settings and set a billing account for the project or do `gcloud auth application-default set-quota-project <your-project-id>`.

---

For more details, see the README in each service directory.
