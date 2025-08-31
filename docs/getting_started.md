# Getting Started with Facet

Facet is a no-code managed platform-as-a-service where you can easily fine tune Gemma family of vision and small language models for your task and domain specific adaptations, without worrying about dataset formatting, cloud engineering, training algorithms -- you can get started from scratch in a few minutes with little programming & AI experience!

## Features and Limitations

### What we support

- Creating an account to manage your datasets, training, evaluation, etc.

- Augmentation and preparation of datasets from custom upload or huggingface hub for language modelling, preference tuning, reinforcement learning, and multimodal tasks

- Configuring fine tuning for Gemma 3 270M, 1B, 4B, 12B, 27B, Gemma 3n E2B and E4B using supervised fine tuning (SFT), direct policy optimization (DPO), and group-related policy optimization (GRPO).

- Run inference & evaluation suites on fine tuned models to assess their performance

- Export models in various formats (adapter, merged, GGUF) and quantizations and deploy them easily on Google Cloud Run or other providers with pre-built containers

### What we're not for

- Dataset collection: while we help you augment your dataset based on uploaded sources, we do not e.g. scrape the web to collect data

- LLM deployment: our service will not run the deployed model for you, you should setup your own runtime / cloud project to do that, we have tutorial for it...

## Getting Started

You can access our service in three most common use cases:

1. Directly through our web interface (runs on our managed Google Cloud Platform)

You may create an account and access [here]()

2. Deploy our project on your own cloud service provider (recommended: GCP)

Follow the [deloy your own service]() guide. Our docker container images are served on `docker-hub`. We provide `terraform` to easily setup the entire GCP project, but for other providers (e.g. AWS) you can use our pre-built containers.

You will need to provide the endpoint to the frontend on our web frontend or deploy your own web server (@ADARSH WHCIH IS BETTER!!)

3. Fork our project for your own customization

Feel free since we're totally open-source! See [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to us instead.

## Usage guide

If you have experience fine tuning LLM before, you may find the UI already very intuitive. If you're confused at any point, refer to the following docs for the stpes and features:

1. [Creating datasets](datasets.md)

2. [Creating fine tune jobs](training.md)

3. [Inference, evaluation, export](inference_export.md)

4. [Deploying the model](deployment.md)

## Acknowledgement

- We are extremely grateful for HuggingFace `datasets`, `transformers`, `trl`, and Unsloth `unsloth, unsloth-zoo` as our underlying backend for fine tuning, and `vLLM, llama.cpp, ollama` as our inference backend.

- We would also like to thank various mentors at Google Cloud, Google DeepMind, Google Summer of Code for their dedicated support

## Developer's Documentation

This guide is mainly for users of the platform. It is meant to be high-level, beginner-friendly. If you are interested in contributing or the technical implementation, this is not the right place to look. Please refer to the `README.md` in each subfolder for more details.
