"""
This script is used to generate a high-level architecture diagram of the Facet Cloud using the `diagrams` library.
Please update the diagram manually whenever major changes are made to the infrastructure.
This is not automated by terraform but is just a visualization aid.

To run:
pip install diagrams
python infrastructure/generate-diagram.py

NOTE: Not all GCP components are available yet in the library e.g. Serverless VPC Connector
Terraform is always the authoritative source of truth for the actual infrastructure.
"""

from diagrams import Diagram, Cluster
from diagrams.gcp.api import APIGateway
from diagrams.gcp.compute import Run
from diagrams.gcp.database import Firestore
from diagrams.gcp.network import VPC, NAT
from diagrams.gcp.storage import GCS
from diagrams.firebase.develop import Authentication
from diagrams.onprem.client import Users
from diagrams.onprem.network import Internet
# from diagrams import Node


with Diagram("Facet AI - GCP Architecture v5", show=False, direction="LR"):
    # Users
    users = Users("Platform Users")

    auth_layer = Cluster("Authentication & API Layer")
    with auth_layer:
        firebase_auth = Authentication("Firebase\nAuthentication")
        api_gateway = APIGateway("API Gateway")

    compute_services = Cluster("Compute Services (Cloud Run)")
    with compute_services:
        core_services = Cluster("Core Services")
        with core_services:
            preprocessing = Run("Preprocessing\nService")
            training_svc = Run("Training\nService")
            inference = Run("Inference\nService (GPU)")

        background_jobs = Cluster("Background Jobs")
        with background_jobs:
            training_job = Run("Training Job\n(GPU)")
            export_job = Run("Export Job (GPU)")

    networking = Cluster("Networking")
    with networking:
        vpc = VPC("VPC")
        nat = NAT("Cloud NAT\n(Static IP)")

    external_services = Internet("External Services\n(e.g. Hugging Face)")

    storage_db = Cluster("Storage & Database")
    with storage_db:
        # cloud_storage = Cluster("Cloud Storage", "LR")
        # with cloud_storage:
        #     datasets_bucket = GCS("Datasets")
        #     models_bucket = GCS("Models")
        bucket = GCS("Datasets & Models")
        firestore = Firestore("Firestore\n(Jobs & Datasets)")

    # Connections
    users >> api_gateway
    api_gateway >> [preprocessing, training_svc, inference]
    training_svc >> [training_job, export_job]

    # Internal GCP traffic to storage
    [export_job] >> bucket
    [training_svc, inference] >> firestore

    # Outbound traffic via static IP
    (training_job >> vpc >> nat >> external_services)
