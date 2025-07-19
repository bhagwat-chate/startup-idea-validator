import os
import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, SocietyOfMindAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from dotenv import load_dotenv
from typing import List, Optional, Callable, Any

load_dotenv()
api_key = os.getenv('OPEN_ROUTER_API_KEY')

if not api_key:
    raise ValueError("OPEN_ROUTER_API_KEY is missing.")

llm_client = OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o",
    api_key=api_key,
    model_info={
        "family": "openai",
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "structured_output": True
    }
)


async def main() -> None:
    model_client = llm_client

    agent1 = AssistantAgent("assistant1", model_client=model_client, system_message="You are a writer, write well.")
    agent2 = AssistantAgent(
        "assistant2",
        model_client=model_client,
        system_message="You are an editor, provide critical feedback. Respond with 'APPROVE' if the text addresses all feedbacks.",
    )
    inner_termination = TextMentionTermination("APPROVE")
    inner_team = RoundRobinGroupChat([agent1, agent2], termination_condition=inner_termination)

    society_of_mind_agent = SocietyOfMindAgent("society_of_mind", team=inner_team, model_client=model_client)

    agent3 = AssistantAgent(
        "assistant3", model_client=model_client, system_message="Translate the text to Spanish."
    )
    team = RoundRobinGroupChat([society_of_mind_agent, agent3], max_turns=2)

    stream = team.run_stream(task="Write a short story with a surprising ending.")
    await Console(stream)


asyncio.run(main())
