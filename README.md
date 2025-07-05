![idiomatic](idiomatic.jpeg)

---

Many non-native English speakers — even those with strong proficiency — sometimes struggle to understand or confidently use idiomatic expressions. Idioms, whether common or obscure, can become barriers to communication when misunderstood, yet avoiding them can strip language of its natural rhythm and charm.

**Idiomatic** takes an active learning approach to help learners grasp and apply idioms in everyday conversation. Built with an agentic workflow and powered by spaced repetition, the tool delivers an engaging, personalized learning experience that grows with the user. While intended for non-native speakers, the tool can be used by anyone who wants to improve their language prowess, as it attempts to adapt to the user's level of expertise.

A few points of technicality here:
 1. I first came upon this issue when a fellow classmate, who was well-spoken, lamented how preparations for his [GMAT](https://www.mba.com/exams/gmat-exam) were severely impeded by idioms/expressions and he found them not intuitive.
 2. Previously, building such a tool might require constructing a database of known idioms, extensive classification etc. However, availability of powerful models like Gemini now allows one to deliver the tool based exclusively on smart prompting and well defined user workflows.

The tool relies on following Gen AI capabilities:
 1. Structured output/JSON mode/controlled generation
 2. Few-shot prompting
 3. Agents

## Models

 We instantiate two LLMs for use in the workflow:
 - an orchestration `llm` to function as a chat model
 - a client model to generate output (using `generate_content`)

Our reasoning is simple. We want to separate the functioning of the two models so that one is more diverse in its output (using different temperature settings and prompts etc.) for question and answer generation, whereas the orchestration LLM is set to a lower temperture to have more predictable output.

## Agent Workflow

We define an LangGraph state to maintain information throught the workflow. 

## Question & Answer Generation

Since, this is *the meat and potatoes* (ha!) of the app, we use a separate call to an LLM model to generate questions here. In this manner, I can adjust model parameters and as well have a more appropreiately engineered prompt. Currently, I'm using Gemini Flash right now, but we could also change this to a more sophisticated model while restircting output tokens thereby managing cost but elevating quality. 

## Tools 

I added `tools` for code interruptions like quitting, explanations or current score. Admittedly, adding this caused the app to become a bit brittle because my understanding of how tools would work was different from how they actually work in `LangChain`. Regardless, with some help from Gemini, I was able to hook up the tools such that reporting score and quitting work well. However, explanations are still a bit of hit or a miss.

I used the technique taught in the course of:
- listing the tools
- adding a ToolNode
- binding the tools to the LLM

## Orchestration LLM & Routing

I followed the Agents code lab to have a Chatbot node that could orchestrate different actions. I needed to refine the system prompt to strongly emphasize tool usage, otherwise tools were not being called. I also needed some help from Gemini, as I couldn't get the routing correct by myself and the code labs don't go deep into orchestration. 

## Project Structure

The codebase has been refactored into a Python package structure for better organization and maintainability:

- `idiomatic/`: The main package directory.
  - `__init__.py`: Makes the directory a Python package and exports `IdiomaticConfig` and `run_app`.
  - `app.py`: Contains the `run_app` function, which orchestrates the application setup (LLM clients, tool binding) and execution. It also defines global instances for LLM clients and application configuration that other modules can import at runtime.
  - `config.py`: Defines the `IdiomaticConfig` dataclass for managing all application settings.
  - `agent.py`: Defines the `IdiomaticState` TypedDict for the LangGraph agent and includes core agent nodes like `chatbot_node` (for user interaction and LLM routing) and `get_user_input`. It also contains the main system prompt for the agent.
  - `qna_generation.py`: Handles the logic for generating idiom questions and evaluating user answers (e.g., `generate_idiom_question`, `evaluate_quiz_answer`, `IdiomQuizItem` dataclass, and Q&A prompt).
  - `tools.py`: Defines the custom tools available to the agent (e.g., `show_score`, `explain_last_question`) and creates the `ToolNode` instance.
  - `graph.py`: Contains the `route_logic` function and the `create_graph` function, which builds and compiles the LangGraph `StateGraph`.
  - `utils.py`: Provides utility functions, currently including data persistence (`load_user_data`, `save_user_data`) and display fallbacks for non-IPython environments.
- `main.py`: An example script demonstrating how to import `IdiomaticConfig` and `run_app` to start the application.
- `requirements.txt`: Lists project dependencies.
- `user_data.json`: Default file for storing user progress (created when the app runs).

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set your Google API Key:**
    The application requires a Google API Key with access to the Gemini models. Set it as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
    (On Windows, use `set GOOGLE_API_KEY="YOUR_API_KEY_HERE"` or set it through system properties).
5.  **Run the application:**
    ```bash
    python main.py
    ```
    The application will start, and you can interact with it in your terminal. User data will be saved in `user_data.json` by default.

## The Agent Graph

The graph below shows the workflow. The core logic for the graph nodes is now organized into specific modules:
- Agent interaction nodes (`chatbot_node`, `get_user_input`) are in `idiomatic/agent.py`.
- Question generation and evaluation nodes (`generate_idiom_question`, `evaluate_quiz_answer`) are in `idiomatic/qna_generation.py`.
- The `ToolNode` is defined in `idiomatic/tools.py`.
- The graph itself, including `route_logic` and connections, is constructed in `idiomatic/graph.py`.

I had to add the `get_input` node as an intermediate as I wanted the user to break the question and answer flow to interact with the tutor at any instance. Presently, the chatbot engages in some niceties, generates questions for the user to answer and then scores them. At any instance the user might provide a natural language prompt, at which point the Chatbot will either use its tools to reply or politely that it doesn't know, at which point the user can continue playing the quiz.

## Final Remarks

This was a very fun exercise for me from which I learned a lot of new things. I personally used the tool myself and found a few idioms that I had been using wrongly in the past. I left in the debug statements, as I think it is cool to see how routing is happening. Despite some bias, I do think this is a cool app, however, the present implementation has some shortcomings that I must disclose:
* the implementation is a brittle and will not scale well
* while I tried to tweak model parameters for diverse output, I still see the same idioms being repeated several times
* when asked to explain the last idiom, or why user got the answer wrong, the chatbot will often default to the idiom "break a leg" and not explain the idiom that is being studied
* data persistence doesn't work well, so while the app quits graefully, it is not able to save user data
* another consequence of not saving user data is that spaced repetition couldn't be implemented properly, it's something I am keen to revisit later on

Having said, the chat interface does allow the user to correct the bot and when prodded the chatbot politely informs the user what might be going wrong with the workflow. Redemption, yet ?
