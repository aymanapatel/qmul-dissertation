import asyncio
import os
from pathlib import Path

from browser_use import Agent
from browser_use.browser.profile import BrowserProfile
from browser_use.llm.openrouter.chat import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

async def main():
    llm = ChatOpenRouter(
        model="deepseek/deepseek-v4-flash",
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    task = "Find the number 1 post on Show HN"
    profile = BrowserProfile(keep_alive=True)
    agent = Agent(task=task, llm=llm, browser_profile=profile)
    await agent.run()

    page = await agent.browser_session.get_current_page()
    if page:
        html = await page.evaluate("() => document.documentElement.outerHTML")
        print(html)
    else:
        print("No page available")

if __name__ == "__main__":
    asyncio.run(main())
