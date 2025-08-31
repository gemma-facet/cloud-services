# All about Datasets

Fine tuning a LLM is only as good as the dataset you provide. Facet provides a no-code interface to help you create, augment, and prepare datasets for various fine tuning tasks.

## Creating Datasets

You can create datasets in Facet by uploading your own data files or by using our integration with the Hugging Face Hub. We support various file formats, including CSV, JSON, and Parquet.

### Subset, Splits, Slices

When creating a dataset, you can specify how to split the data into training, validation, and test sets. You can choose from predefined splits (e.g., 80/10/10) or create custom splits based on your requirements.

1. Some datasets have various subsets if they are very big on huggingface hub. You can choose the subset you want to use when importing from huggingface.

2. We often split the dataset into training, validation, and test sets. You can choose from predefined splits (e.g., 80/10/10) or create custom splits based on your requirements.

3. You can also create slices of the dataset by listing how many samples you want to use. **We plan to expand this to support filtering based on conditions to create slices that are customized to your needs.**

## Augmenting Datasets

Facet offers several data augmentation techniques to help you improve your dataset quality. You can apply techniques such as:

- Text paraphrasing
- Synonym replacement
- Back-translation

In addition, we support data synthesis using generative models (Gemini) to create new training examples based on your existing data. This is a method that is proven highly efficient in production systems.

> [!IMPORTANT]
> We currently do not support augmentation for vision dataset but are looking to ship it soon!

## Preparing and Formatting Datasets

Datasets come in a variety of formats and structures, and it's essential to ensure they are properly prepared for fine tuning for the specific task you are looking to achieve.

We use conversational format by default and handles all the conversion for you internally. You can read more at `preprocessing/README.md`.

Datasets are often tightly coupled with the fine tuning task you want to achieve. We currently support three main tasks:

1. **Language Modelling (LM)**: For LM tasks, we expect dataset to be in turns of user and assistant format, and include system messages if available. This will be used for supervised fine tuning (SFT) where you can do instruction tuning or domain adaptation.

2. **Preference Alignment**: For preference alignment tasks, we expect dataset to include pairs of good and bad responses for the same user query (rejected and chosen). This will be used for direct policy optimization (DPO) where the model learns to prefer the good response over the bad one.

3. **Reinforcement Learning (RL)**: For RL (reasoning) tasks, we expect dataset to include user queries only. When used with group-related policy optimization (GRPO), the model generate respones and get feedback by scoring with graders (or reward functions) and learn to maximize the reward.

### Multimodal Datasets

We also support multimodal datasets that include both text and images. For such datasets, we expect the text data to be in conversational format as described above, and the image data to be provided as URLs or base64-encoded strings within the text.

Vision datasets is now supported across all tasks (SFT, DPO, GRPO).

> [!IMPORTANT]
> Currently, multimodal datasets are only supported when you import dataset from hugging face datasets, not custom upload. We are working to support custom upload soon!

## What's Next?

We will handle everything else for you, including tokenization, batching, and formatting the data for training. You can then proceed to create a fine tune job using the prepared dataset. See [Creating Fine Tune Jobs](training.md) for more details.
