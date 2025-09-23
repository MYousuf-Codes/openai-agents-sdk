import os
from agents import Agent, ModelSettings, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

from dotenv import load_dotenv
load_dotenv()

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
    tracing_disabled=True
)

# Low temperature (0.1) = Very focused, consistent answers
agent_focused = Agent(
    name="Math Tutor",
    instructions="You are a focused, realistic agent.",
    model_settings=ModelSettings(temperature=0.1)
)

# High temperature (0.9) = More creative, varied responses
agent_creative = Agent(
    name="Story Writer",
    instructions="You are a creative storyteller.",
    model_settings=ModelSettings(temperature=0.9)
)

main_agent = Agent(
    name="Main Agent",
    instructions="You are an agent that decides whether to use the Math Tutor or Story Writer based on the user's request.",
    handoffs=[agent_focused, agent_creative],
    model=model,
    model_settings=ModelSettings(temperature=0.5)
)

result = Runner.run_sync(
    main_agent,
    "Write a short creative story about a robot who learns to paint",
    run_config=config
)

print("\nCALLING AGENT\n")
print(result.final_output)