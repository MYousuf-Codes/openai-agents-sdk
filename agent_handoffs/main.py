import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, enable_verbose_stdout_logging

from agents.run import RunConfig

from dotenv import load_dotenv
load_dotenv()

# Enable verbose logging for debugging purposes
enable_verbose_stdout_logging()

set_tracing_disabled(False)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
)

python_agent: Agent = Agent(
    name="Python Agent",
    instructions="You are a Python programming expert. You can write optimized and scalable code and debug Python code.",
    model=model,
    handoff_description="Your expertise is needed for Python-related tasks."
)

typescript_agent: Agent = Agent(
    name="TypeScript Agent",
    instructions="You are a TypeScript programming expert. You can write optimized and scalable code and debug TypeScript code.",
    model=model,
    handoff_description="Your expertise is needed for TypeScript-related tasks."
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a triage agent. Your job is to determine whether a given programming task should be handled by the Python programming expert or the TypeScript programming expert. If the task is related to Python, hand it off to the PythonAgent. If it's related to TypeScript, hand it off to the TypeScriptAgent. If the task is ambiguous or involves both languages, choose the agent that is most appropriate based on the context."
    ),
    model=model,
    handoff_description="Determine which programming expert should handle the task.",
    handoffs=[python_agent, typescript_agent]
)

result = Runner.run_sync(
    triage_agent, 
    "What is python?",
    run_config=config
)
print("\nCALLING AGENT\n")
print(result.final_output)