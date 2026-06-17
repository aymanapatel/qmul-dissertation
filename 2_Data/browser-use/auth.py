import functools
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PASSWORDS_FILE = BASE_DIR / ".env.passwords"

load_dotenv(BASE_DIR / ".env")
load_dotenv(PASSWORDS_FILE)

EMAIL = os.getenv("EMAIL", "")


def get_saved_password(domain: str) -> str | None:
    load_dotenv(PASSWORDS_FILE, override=True)
    return os.getenv(domain) or None


def login(func):
    @functools.wraps(func)
    async def wrapper(domain: str, *args, **kwargs):
        email = EMAIL
        password = get_saved_password(domain)
        if not password:
            raise ValueError(f"No password found for {domain} in .env.passwords")
        return await func(domain, email, password, *args, **kwargs)

    return wrapper
