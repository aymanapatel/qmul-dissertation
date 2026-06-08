import asyncio
import os
from pathlib import Path

from browser_use import Agent
from browser_use.llm.openrouter.chat import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

async def main():
    llm = ChatOpenRouter(
        model="moonshotai/kimi-k2-thinking",
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    task = "Find the number 1 post on Show HN"
    agent = Agent(task=task, llm=llm)
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
