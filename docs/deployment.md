# Now it's time to ship

## Running Fine-tuned Models Locally (GGUF)

This provides a guide to run fine tuned models locally. For on the cloud, we have provided direct deployment options already using various backend providers :)

You will need to download the converted GGUF file from the export section first. If you are using HF hub, you should directly provide the HF hub repo name to the following commands.

### llama.cpp

1. Install `llama.cpp` following the [llama.cpp Installation Guide](https://github.com/ggerganov/llama.cpp#installation)

2. Run the model:

   ```bash
   llama-cli -m path-to-gguf-file -sys "system-prompt-if-any"
   ```

### Ollama

1. Install Ollama: [Ollama Installation Guide](https://ollama.com/docs/installation)

2. Start Ollama server:

   ```bash
   ollama serve
   ```

3. Create `Modelfile`

   ```txt
   FROM path-to-gguf-file

   SYSTEM """
   system-prompt-if-any
   """

   PARAMETER...
   ```

4. Create a model using the GGUF file:

   ```bash
   ollama create <model_name> -f Modelfile
   ```

5. Run the model:

   ```bash
    ollama run <model_name>
   ```

6. If the model is on HF Hub:

   ```bash
   ollama run hf.co/{username}/{repository}:{quantization}
   ```

   [references](https://huggingface.co/docs/hub/en/ollama)

## Creating your own deployment service on Cloud (vLLM)

We will guide you to deploying model on Cloud Run!

We will plan to support other inference framework like SGLang soon. We might also support Ollama but that's more likely used on local deployment.
