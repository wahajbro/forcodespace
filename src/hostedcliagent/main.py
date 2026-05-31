import os
from agent_framework import Message
from agent_framework.azure import AzureAIAgentClient
from agent_framework.orchestrations import SequentialBuilder
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from typing import cast
import asyncio

load_dotenv()

async def run_agents(feedback: str):
    summarizer_instructions = "Summarize the feedback in one short sentence."
    classifier_instructions = "Classify as Positive, Negative, or Feature request."
    action_instructions = "Suggest next action in one short sentence."

    credential = DefaultAzureCredential()

    async with AzureAIAgentClient(credential=credential) as chat_client:
        summarizer = chat_client.as_agent(instructions=summarizer_instructions, name="summarizer")
        classifier = chat_client.as_agent(instructions=classifier_instructions, name="classifier")
        action = chat_client.as_agent(instructions=action_instructions, name="action")

        workflow = SequentialBuilder(participants=[summarizer, classifier, action]).build()

        outputs = []
        async for event in workflow.run(f"Customer feedback: {feedback}", stream=True):
            if event.type == "output":
                outputs.append(cast(list[Message], event.data))

        if outputs:
            return "\n".join(msg.text for msg in outputs[-1])
        return "No output generated"


def main():
    server = ResponsesHostServer(run_agents)
    server.run()


if __name__ == "__main__":
    main()