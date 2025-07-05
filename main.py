from idiomatic import IdiomaticConfig, run_app
import os

# This is an example of how to run the Idiomatic application.

# Option 1: Minimal configuration (will prompt for user details if not found in user_data.json)
# config = IdiomaticConfig()

# Option 2: Pre-configure some user details
# These will be used if no existing user_data.json provides them, or if it's a new user.
# If user_data.json exists for a user, those details might take precedence on load.
config = IdiomaticConfig(
    user_name="Alex",             # Optional: Pre-fill name
    user_level="intermediate",    # Optional: "beginner", "intermediate", "advanced"
    idiom_category="general"      # Optional: e.g., "business", "food", or "general"
)

# The GOOGLE_API_KEY is expected to be in your environment variables.
# IdiomaticConfig will try to load it automatically.
# If you need to pass it explicitly (e.g., from a different source), you can do:
# config.google_api_key = "YOUR_ACTUAL_API_KEY"
# However, environment variables are generally preferred for API keys.


if __name__ == "__main__":
    # Check if the API key was successfully loaded by IdiomaticConfig or is set
    if not config.google_api_key:
        print("Error: GOOGLE_API_KEY is not set.")
        print("Please ensure the GOOGLE_API_KEY environment variable is set before running the application.")
        print("Alternatively, you can set it directly in this script (not recommended for production):")
        print("# config = IdiomaticConfig(google_api_key=\"YOUR_API_KEY_HERE\")")
        # Example of how to prompt if really needed, though env var is best:
        # api_key_input = input("Please enter your GOOGLE_API_KEY: ").strip()
        # if api_key_input:
        #     config.google_api_key = api_key_input
        #     run_app(config)
        # else:
        #     print("API Key is required.")
    else:
        print("Attempting to run the Idiomatic app...")
        run_app(config)
