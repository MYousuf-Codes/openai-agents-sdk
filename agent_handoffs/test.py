import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

from dotenv import load_dotenv
load_dotenv()
from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set")


#Reference: https://ai.google.dev/gemini-api/docs/openai
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
    tracing_disabled=False

)

agent: Agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=model
)

result = Runner.run_sync(
    agent, 
    "Hello, how are you.", 
    run_config=config
)

print("\nCALLING AGENT\n")
print(result.final_output)
