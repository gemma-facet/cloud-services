import logging
import random
from typing import List, Dict, Any, Optional
from google import genai

logger = logging.getLogger(__name__)


class GeminiSynthesizer:
    """
    Gemini-based synthetic data generator for conversation datasets.
    Generates new conversation samples based on existing patterns.
    Uses the new Google GenAI library.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-001"):
        import os
        self.model_name = model_name
        self.client = None
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        try:
            if self.api_key:
                # Use provided API key or env var
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Gemini synthesizer with API key")
            else:
                raise ValueError("API key is required (provide via api_key param or GOOGLE_API_KEY env var)")

            logger.info(f"Using model: {model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client = None

    def _create_synthesis_prompt(
        self,
        sample_conversations: List[Dict],
        num_samples: int = 1,
        user_custom_prompt: str = "",
    ) -> str:
        """Create a prompt for synthesizing new conversations based on examples"""

        # Extract example conversations for the prompt
        examples = []
        for i, conv in enumerate(sample_conversations[:3]):  # Use up to 3 examples
            if "messages" in conv:
                example_text = f"Example {i + 1}:\n"
                for msg in conv["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    # Skip system messages in examples to avoid confusion
                    if role.lower() != "system":
                        example_text += f"{role.capitalize()}: {content}\n"
                examples.append(example_text)

        examples_text = "\n".join(examples)

        # Adjust format instruction based on whether system messages are used
        format_instruction = """Output each conversation in this exact format:
```
User: [question or request]
Assistant: [helpful response]
```"""

        # Build the base prompt
        base_prompt = f"""You are a dataset generator. Create {num_samples} new conversation(s) similar to the examples below, but with different topics and content. Follow the exact same format and conversation style.

{examples_text}"""

        # Add custom synthesis instructions if provided
        synthesis_instructions = ""
        if user_custom_prompt.strip():
            synthesis_instructions = f"""

CUSTOM SYNTHESIS INSTRUCTIONS:
{user_custom_prompt.strip()}

Please follow these custom instructions when generating the conversations."""

        # Add standard generation guidelines
        standard_guidelines = f"""

Generate {num_samples} new conversation(s) following the same pattern:
- Keep the same roles (user/assistant)
- Maintain similar conversation flow and style
- Use different topics and content
- Make each conversation realistic and coherent
- Ensure responses are helpful and informative"""

        # Combine all parts
        prompt = f"""{base_prompt}{synthesis_instructions}{standard_guidelines}

{format_instruction}
"""
        return prompt

    def _parse_generated_conversations(
        self, generated_text: str, system_message: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Parse generated text into conversation format
        The user needs to specify a system message if required
        We do not let the model generate repeated system messages to save tokens
        """
        conversations = []

        # Split by conversation blocks (looking for User: patterns)
        parts = generated_text.split("```")

        for part in parts:
            part = part.strip()
            if not part or "User:" not in part:
                continue

            messages = []

            # Add system message once at the beginning if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})

            lines = part.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("User:"):
                    content = line[5:].strip()
                    messages.append({"role": "user", "content": content})
                elif line.startswith("Assistant:"):
                    content = line[10:].strip()
                    messages.append({"role": "assistant", "content": content})

            # Validate we have at least user + assistant (plus optional system)
            min_required = 3 if system_message else 2
            if len(messages) >= min_required:
                conversations.append({"messages": messages})

        return conversations

    def synthesize_conversations(
        self,
        sample_conversations: List[Dict[str, Any]],
        num_samples: int = 5,
        system_message: str = "",
        max_batch_size: int = 10,
        user_custom_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Generate new conversations based on sample conversations

        Args:
            sample_conversations: List of example conversations to learn from
            num_samples: Number of new conversations to generate
            system_message: Optional system message to add to each conversation
            max_batch_size: Maximum number of samples to generate in one API call
            user_custom_prompt: Optional custom instructions for synthesis

        Returns:
            List of generated conversation samples
        """
        if not self.client:
            logger.warning("Gemini client not initialized, returning empty list")
            return []

        if not sample_conversations:
            logger.warning("No sample conversations provided")
            return []

        try:
            # Use intelligent batching - larger batches for efficiency but not too large to overwhelm the model
            # Research suggests 5-10 samples per batch works well for conversation generation
            batch_size = min(max_batch_size, num_samples)
            all_generated = []
            total_batches = (
                num_samples + batch_size - 1
            ) // batch_size  # Ceiling division

            logger.info(
                f"Generating {num_samples} conversations in {total_batches} batch(es) of up to {batch_size} each"
            )

            batch_count = 0
            while len(all_generated) < num_samples:
                batch_count += 1
                remaining = num_samples - len(all_generated)
                current_batch_size = min(batch_size, remaining)

                logger.info(
                    f"Batch {batch_count}/{total_batches}: Generating {current_batch_size} conversations"
                )

                # Create prompt for this batch
                prompt = self._create_synthesis_prompt(
                    random.sample(
                        sample_conversations, min(3, len(sample_conversations))
                    ),
                    current_batch_size,
                    user_custom_prompt=user_custom_prompt,
                )

                # Generate with Gemini using new API
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name, contents=prompt
                    )

                    if response.text:
                        # Parse generated conversations
                        generated_convs = self._parse_generated_conversations(
                            response.text, system_message=system_message
                        )
                        all_generated.extend(generated_convs)

                        logger.info(
                            f"Batch {batch_count} generated {len(generated_convs)} conversations"
                        )

                        # If we got significantly fewer conversations than requested, warn user
                        if (
                            len(generated_convs) < current_batch_size * 0.5
                        ):  # Less than 50% success rate
                            logger.warning(
                                f"Low generation success rate in batch {batch_count}: {len(generated_convs)}/{current_batch_size}"
                            )
                    else:
                        logger.warning(
                            f"Empty response from Gemini in batch {batch_count}"
                        )
                        # Continue to next batch instead of breaking - sometimes one batch fails but others succeed
                        continue

                except Exception as e:
                    logger.error(f"Error in batch {batch_count}: {e}")
                    # Continue to next batch
                    continue

            # Return only the requested number
            result = all_generated[:num_samples]
            success_rate = len(result) / num_samples * 100
            logger.info(
                f"Successfully generated {len(result)}/{num_samples} conversations ({success_rate:.1f}% success rate)"
            )
            return result

        except Exception as e:
            logger.error(f"Error generating conversations with Gemini: {e}")
            return []

    def is_available(self) -> bool:
        """Check if the synthesizer is properly configured and available"""
        return self.client is not None
