# This file will contain the IdiomaticConfig class.
import os
from dataclasses import dataclass, field

@dataclass
class IdiomQuizGenerationConfig:
    """Configuration for the idiom quiz generation model."""
    model_name: str = "gemini-2.0-flash" # Changed from "gemini-2.0-flash" to match usage in idiomatic.py (was gemini-1.5-flash before, but generate_idiom_question uses gemini-2.0-flash)
    max_output_tokens: int = 200
    temperature: float = 1.5
    top_p: float = 0.95
    response_mime_type: str = "application/json"
    # response_schema will be set dynamically in the app code

@dataclass
class OrchestrationLLMConfig:
    """Configuration for the orchestration LLM."""
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7

@dataclass
class IdiomaticConfig:
    """Configuration for the Idiomatic application."""
    user_data_path: str = 'user_data.json'

    # User-specific settings, can be prompted or set directly
    user_name: str | None = None
    user_level: str | None = None # e.g., "beginner", "intermediate", "advanced"
    idiom_category: str | None = None # e.g., "business", "general"

    # Model configurations
    quiz_generation_config: IdiomQuizGenerationConfig = field(default_factory=IdiomQuizGenerationConfig)
    orchestration_llm_config: OrchestrationLLMConfig = field(default_factory=OrchestrationLLMConfig)

    # API Key - this should be set up in the environment, but can be passed if needed.
    # For security, it's better to load from env directly in the app.
    google_api_key: str | None = None

    def __post_init__(self):
        if self.google_api_key is None:
            self.google_api_key = os.getenv('GOOGLE_API_KEY')
        # Basic validation or defaulting for user inputs can be added here if desired
        # For example, ensuring user_level is one of the expected values.
        # For now, we'll keep it simple.
