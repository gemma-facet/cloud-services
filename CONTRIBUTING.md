# Contributing to Facet AI

Thank you for your interest in contributing to Facet AI! We're excited to have you help make LLM fine-tuning accessible to everyone.

## Getting Started

Facet AI is a no-code web platform for fine-tuning DeepMind's Gemma 3 family of multimodal models. Before contributing, please familiarize yourself with the project by reading our [README](README.md) and exploring the architecture.

## How to Contribute

### 🐛 Bug Reports

We really appreciate bug reports! They help us improve the platform for everyone. When reporting a bug, please include:

- A clear description of the issue
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment details (browser, OS, etc.)
- Screenshots or logs if applicable

### 💡 Feature Requests

We love hearing your ideas for new features! Please check our [Project Roadmap](https://github.com/gemma-facet/cloud-services/issues/52) to see what's already planned, then feel free to suggest:

- New dataset preprocessing capabilities
- Additional training methods or frameworks
- Export format support
- Evaluation metrics
- UI/UX improvements

### 🔧 Pull Requests

We are currently working on a way to accept external PRs while ensuring the security and integrity of the platform. Testing is usually the tricky part here since the project is tightly integrated with GCP services. Please stay tuned for updates!

#### Before submitting a PR

1. **Open an issue first** to discuss the change (unless it's a small fix)
2. **Fork the repository** and create a feature branch
3. **Follow the existing code style** in each service directory
4. **Test your changes thoroughly** in the affected components
5. **Update documentation** if needed

#### PR Guidelines

- Keep changes focused and atomic
- Write clear commit messages
- Include a detailed description of what the PR does
- Reference any related issues
- Test locally using the deployment instructions in `infrastructure/README.md`

## Development Setup

For local development:

1. Check the `README.md` in each service directory for specific setup instructions
2. Use the Terraform scripts in `infrastructure/` for full deployment testing
3. Refer to our [developer documentation](https://facetai.mintlify.app) for API details

## Project Structure

- `preprocessing/` - Dataset preprocessing service
- `training/` - Model training service
- `inference/` - Model inference and evaluation service
- `export/` - Model export and format conversion
- `infrastructure/` - Terraform deployment scripts
- `docs/` - Documentation and guides

## Questions?

- Check our [user documentation](https://facetai.mintlify.app)
- Open an issue for technical questions
- Read the README in relevant service directories

## Code of Conduct

Please be respectful and constructive in all interactions. We're building this platform to democratize access to AI fine-tuning, and we want our community to reflect those inclusive values.

Thank you for contributing to Facet AI! 🚀
