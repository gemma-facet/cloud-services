# IaC for Gemma Fine-tuning Services

You can use this Terraform setup to quickly deploy the Gemma fine-tuning services infrastructure on your own Google Cloud Platform project.

1. This only works with Google Cloud Platform (GCP). Open issue / PR if you want to add support for AWS/Azure!

2. This requires installation of [Terraform](https://www.terraform.io/downloads.html) and [gcloud CLI](https://cloud.google.com/sdk/docs/install).

3. This does not create a GCP project for you. You should create one from the Console and make sure you have the correct IAM role (Owner or Editor) to create resources.

4. As the writing of this, certain APIs require billing to be enabled, so it's recommended to enable billing on your project first. We might consider adding 3 and 4 to the Terraform setup in the future.

> [!NOTE] We provide `Makefile` (see below) to simplify the entire deployment process. We do not recommend using `terraform` command manually because the deployment takes multiple stages (setup + build + deploy).

## Quickstart

1. **Clone the repository**

```bash
git clone https://github.com/gemma-facet/cloud-services
cd cloud-services/infrastructure
```

2. **Authenticate with GCP**

```bash
gcloud auth login
# Set your project
gcloud config set project <your-project-id>
```

3. **Edit variables**

Copy `terraform.tfvars.example` to `terraform.tfvars` and fill in your project details:

```bash
cp terraform.tfvars.example terraform.tfvars
```

> [!NOTE]
> We use the project id you provided to create names for resources such as storage buckets and service account to ensure that they are globally unique. If you want to customize the names, edit the `main.tf` files in each module.

4. **Deploy the infrastructure**

```bash
make init
make full-deploy
make output
```

`make full-deploy` performs the following steps:

1. Deploy core infrastructure (APIs, IAM, Artifact Registry, Storage, Firebase) with `make deploy-core`
2. Build and push docker images to Artifact Registry with `make build`
3. Deploy microservices and sets up networking (Cloud Run and API Gateway) with `make deploy-services`

## Available Commands

```bash
make help          # Show all available commands

# Prerequisites
make init
make check

# Build & Deploy
make build          # Build all containers with Cloud Build
make deploy         # Deploy infrastructure with Terraform
make deploy-core    # Deploy core infrastructure only
make deploy-services # Deploy microservices only
make full-deploy    # Complete workflow: build + deploy

# Management
make plan           # Plan infrastructure changes
make plan-core      # Plan core infrastructure changes
make plan-services  # Plan microservices changes
make output         # Show infrastructure outputs
make destroy        # Destroy all infrastructure
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
