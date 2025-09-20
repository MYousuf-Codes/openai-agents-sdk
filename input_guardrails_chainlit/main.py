from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import input_guardrail, RunContextWrapper, TResponseInputItem, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import chainlit as cl

load_dotenv()

set_tracing_disabled(disabled=True)

gemini = os.getenv("GEMINI_API_KEY")
if not gemini:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

client = AsyncOpenAI(
    api_key=gemini,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

class OutputPython(BaseModel):
    is_python_related : bool
    reasoning : str

input_guardrail_agent = Agent(
    name="Input Guardrail Checker Agent",
    instructions="Determine if the user input is related to Python programming. If it is, respond with is_python_related set to true and provide reasoning. If not, set is_python_related to false and provide reasoning.",
    model=model,
    output_type=OutputPython
)

@input_guardrail
async def input_guardrail_function(
    ctx: RunContextWrapper,
    agent: Agent,
    input : str | list[TResponseInputItem]
)-> GuardrailFunctionOutput:
    result = await Runner.run(
        input_guardrail_agent,
        input
    )
    return GuardrailFunctionOutput(
        output_info = result.final_output,
        tripwire_triggered = not result.final_output.is_python_related
    )

main_agent = Agent(
    name="Main Agent",
    instructions= "You are an experienced Python programming assistant. Provide detailed and accurate information about Python programming topics.",
    model=model,
    input_guardrails=[input_guardrail_function]
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("Hello! How can I assist you with Python programming today?").send()

@cl.on_message
async def on_message(message: str):
    try:
        result = await Runner.run(
            main_agent,
            input=message.content
        )
        await cl.Message(content=result.final_output).send()
    except InputGuardrailTripwireTriggered :
        await cl.Message(content="I'm Python expert, so please ask only python related questions.").send()