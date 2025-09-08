# Infrastructure Setup for Gemma Fine-Tuning Services

This folder contains Terraform code to provision all required Google Cloud resources for the Gemma fine-tuning platform. This includes Cloud Run services, Cloud Run jobs, GCS buckets, Artifact Registry, and IAM roles.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads.html) >= 1.0
- [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- Permissions to create resources in your GCP project (Owner or Editor, or specific IAM roles)
- (Recommended) Enable billing and required APIs in your GCP project

## Quickstart

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd gemma-fine-tuning-services/infrastructure
```

2. **Authenticate with GCP**

```bash
gcloud auth login
# Set your project
export GOOGLE_CLOUD_PROJECT=<your-project-id>
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

3. **Edit variables**

Copy `terraform.tfvars.example` to `terraform.tfvars` and fill in your project details:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Values in `terraform.tfvars.example` are already prefilled with default values. Feel free to change them to your own. **You should at least fill in your project id.**

| Name                    | Description                           | Default            |
| ----------------------- | ------------------------------------- | ------------------ |
| project_id              | GCP Project ID                        | my-gcp-project     |
| region                  | GCP region for resources              | us-central1        |
| data_bucket_name        | GCS bucket for datasets               | gemma-dataset-dev  |
| export_bucket_name      | GCS bucket for model exports          | gemma-export-dev   |
| config_bucket_name      | GCS bucket for training configs       | gemma-train-config |
| training_image_tag      | Docker image tag for training service | latest             |
| preprocessing_image_tag | Docker image tag for preprocessing    | latest             |

4. **Initialize Terraform**

```bash
terraform init
```

5. **(Optional) Configure remote state**

To enable team collaboration, use a GCS bucket for remote state. Add this to `main.tf` before the `provider` block:

```hcl
terraform {
  backend "gcs" {
    bucket = "<your-tf-state-bucket>"
    prefix = "terraform/state"
  }
}
```

6. **Plan and apply**

```bash
terraform plan
terraform apply
```

## What gets created?

- **GCS Buckets**: For datasets, model exports, and training configs
- **Artifact Registry**: For Docker images
- **Cloud Run Services**: Preprocessing, Training, Inference
- **Cloud Run Job**: Training job (GPU)
- **IAM**: Service account and permissions

## IAM Roles

- `roles/run.admin`: Required for Cloud Run services
- `roles/storage.admin`: Required for managing GCS buckets
- `roles/logging.logWriter`: Required for logging
- `roles/datastore.user`: Required for Firestore access

## Outputs

- URLs for all deployed services
- Names of all buckets and resources

## Troubleshooting

- Make sure your user/service account has permission to create all resources
- If you see API errors, ensure the required APIs are enabled (Terraform will try to enable them)
- If you get quota or billing errors, check your GCP project quotas and billing status
- For GPU resources, ensure your project/region has available GPU quota
- If your project already has resources created by Terraform, you can import them using `terraform import`

## Cleaning Up

To delete all resources created by Terraform:

```bash
terraform destroy
```

---

For more details, see the README in each service directory.
