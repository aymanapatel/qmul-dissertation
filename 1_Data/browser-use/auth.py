import functools
import os
import secrets
import string
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PASSWORDS_FILE = BASE_DIR / ".env.passwords"

load_dotenv(BASE_DIR / ".env")
load_dotenv(PASSWORDS_FILE)

EMAIL = os.getenv("EMAIL", "")
POSTFIX = os.getenv("POSTFIX", "")
YEAR = datetime.now().year


def get_saved_password(domain: str) -> str | None:
    load_dotenv(PASSWORDS_FILE, override=True)
    return os.getenv(domain) or None


def make_password(domain: str, postfix: str = POSTFIX, year: int = YEAR) -> str:
    return f"{domain}_{postfix}_{year}"


def generate_random_password(length: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def save_password(domain: str, password: str) -> None:
    lines: list[str] = []
    if PASSWORDS_FILE.exists():
        lines = PASSWORDS_FILE.read_text(encoding="utf-8").splitlines()

    updated = False
    prefix = f"{domain}="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = f"{domain}={password}"
            updated = True
            break

    if not updated:
        lines.append(f"{domain}={password}")

    PASSWORDS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def login(func):
    @functools.wraps(func)
    async def wrapper(domain: str, *args, **kwargs):
        email = EMAIL
        password = get_saved_password(domain)
        if not password:
            raise ValueError(f"No password found for {domain} in .env.passwords")
        return await func(domain, email, password, *args, **kwargs)

    return wrapper


def signup(func):
    @functools.wraps(func)
    async def wrapper(domain: str, *args, **kwargs):
        email = EMAIL
        password = kwargs.pop("_password", None) or make_password(domain)
        return await func(domain, email, password, *args, **kwargs)

    return wrapper
