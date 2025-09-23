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

# Define a custom RunHook
class LoggingHook(RunHooks):
    async def on_run_start(self, run_id, ctx, **kwargs):
        print(f"[HOOK] Run {run_id} started with input: {ctx.input}")

    async def on_step_end(self, run_id, step, **kwargs):
        print(f"[HOOK] Step {step.step_type} ended with output: {step.output}")

    async def on_run_end(self, run_id, result, **kwargs):
        print(f"[HOOK] Run {run_id} ended with final output: {result.final_output}")

# Attach hooks to RunConfig
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

print("\nCALLING AGENT...\n")

result = Runner.run(
    agent,
    "how are you?",
    run_config=config,
    hooks=[LoggingHook()]  # Attached the custom hook
)

print("\nFINAL RESULT...\n")
print(result.final_output)
