from dataclasses import dataclass, field
import os

@dataclass
class IdiomaticConfig:
    """Configuration for the Idiomatic application."""
    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY', ''))

    # LLM for orchestration (chat, routing, tool decisions)
    orchestrator_llm_model: str = "gemini-1.5-flash"
    orchestrator_llm_temperature: float = 0.7

    # LLM/Client for Q&A generation
    qna_generator_model: str = "gemini-2.0-flash" # Matching original setting for Q&A generation
    qna_generator_temperature: float = 1.5 # As per original GENERATE_QnA_CONFIG
    qna_generator_top_p: float = 0.95      # As per original GENERATE_QnA_CONFIG
    qna_max_output_tokens: int = 200       # As per original GENERATE_QnA_CONFIG

    user_data_path: str = 'user_data.json'

    # Optional: User details if you want to preload them
    user_name: str = ""
    user_level: str = "intermediate" # Default level
    user_category: str = "general"   # Default category

    def __post_init__(self):
        if not self.google_api_key:
            # This will be caught by the main script if still not set
            print("Warning: GOOGLE_API_KEY is not set in environment or by argument.")

# Example usage (for testing or if run directly):
if __name__ == "__main__":
    config = IdiomaticConfig()
    if not config.google_api_key:
        print("GOOGLE_API_KEY is not set. Please set it in your environment or pass it as an argument.")
    else:
        print(f"Config loaded. API Key starts with: {config.google_api_key[:5]}")
    print(f"Orchestrator Model: {config.orchestrator_llm_model}")
    print(f"User Data Path: {config.user_data_path}")
