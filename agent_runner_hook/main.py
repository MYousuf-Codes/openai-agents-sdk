import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunHooks
from agents.run import RunConfig

from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")


external_client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

agent: Agent = Agent(
    name="",
    instructions="",
    model=model
)

result = Runner.run_sync(
    agent, 
    "how to make a paper airplane?", 
    run_config=config
)

print("\nCALLING AGENT...\n")
print(result.final_output)