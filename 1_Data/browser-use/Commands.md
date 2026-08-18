# Browser-Use

---

## Table of Contents

1. [One-Time Setup](#one-time-setup)
2. [Running Tests](#running-tests)
3. [Environment Variables](#environment-variables)
4. [Output Structure](#output-structure)
5. [Troubleshooting](#troubleshooting)
6. [Architecture Overview](#architecture-overview)

---

## One-Time Setup

### 1. Install Node.js Dependencies

```bash
cd 2_Data/browser-use
npm install
```

**What this does:**
- Installs `playwright`
- Installs TypeScript compiler and `ts-node` for running the runner script

**Required packages:**
- `playwright` — Browser automation library
- `typescript` + `ts-node` — TypeScript execution environment


### 3. Verify Python Environment

Ensure your Python virtual environment is active and has the required packages:

```bash
source .venv/bin/activate
python -c "from browser_use import Agent; print('browser-use OK')"
```

---

## Running Tests

### Basic axe-core Accessibility Audit

Run only the automated axe-core WCAG checks (no screen reader):

```bash
python main.py --site example.com
```

### Audit Multiple Sites Directly

Pass multiple domains after `--site`. Use `--workers` to process more than one
domain concurrently:

```bash
python main.py --site example.com example.org example.net --workers 3
```

The flag can also be repeated. Duplicate domains are processed only once:

```bash
python main.py --site example.com --site example.org --workers 2
```
### Sign Up Instead of Log In

Create a new account and test the signup flow:

```bash
python main.py --site example.com --signup
```

---

## Environment Variables

Create a `.env` file in `2_Data/browser-use/` with the following:

```bash
# LLM API Configuration
AI_API_KEY=your_api_key_here
OPENCODE_GO_URL=https://api.opencode.ai/v1

# Email for login/signup
EMAIL=your_test_email@example.com
POSTFIX=testpassword
```

### Password Storage

Generated and saved passwords are stored in `.env.passwords` (auto-created):

```bash
# Example contents
github.com=github_testpassword_2025
facebook.com=facebook_testpassword_2025
```

---

## Output Structure

After a successful run, outputs are organized as:

```
outputs/
├── axe-core/                          # axe-core accessibility reports
│   └── example.com/
│       ├── page-0_home.json
│       ├── page-1_dashboard.json
│       ├── page-2_settings.json
│       └── summary.json
```

### Report File Formats

**axe-core report** (`page-{i}_{slug}.json`):
```json
{
  "url": "https://example.com/dashboard",
  "violations": [...],
  "passes": [...],
  "incomplete": [...]
}
```


---


## Architecture Overview

### How It Works

```
Python (main.py)
  │
  ├─► Browser Use Agent logs in and navigates
  │
  ├─► For each page:
  │    ├─► axe-core audit (JavaScript injection)
  │
  └─► Summarize and save per-domain reports
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **CDP Bridge** | Python (Browser Use) and Node.js (Playwright) share the same browser instance via Chrome DevTools Protocol. This preserves login state, cookies, and page context without re-authentication. |
| **LLM-Generated Plans** | Different pages need different tests (a login form vs. a dashboard). The LLM inspects DOM metadata and decides what screen reader actions matter most for that specific page. |
| **Test Plan Caching** | DOM metadata is hashed (SHA-256). If the same page structure is encountered again, the cached plan is reused — saving LLM API calls and cost. |
| **Headed Browser** | Screen readers (VoiceOver) cannot operate against headless browsers. Chrome must be visible. |
| **macOS Only** | VoiceOver is the macOS built-in screen reader. For Windows NVDA support, the runner would need to detect `process.platform` and switch drivers. |

### Files and Responsibilities

| File | Language | Responsibility |
|------|----------|----------------|
| `main.py` | Python | CLI entrypoint, orchestrates Browser Use agent, axe-core tests |
| `axe.py` | Python | axe-core injection, execution, report saving |
| `auth.py` | Python | Login/signup helpers, password management |
| `package.json` | JSON | Node.js dependencies |
| `tsconfig.json` | JSON | TypeScript compiler configuration |

---

## Quick Reference Card

```bash
# Setup (run once)
cd 2_Data/browser-use && npm install
# axe-core only
python main.py --site example.com

# Multiple sites directly (three concurrent workers)
python main.py --site example.com example.org example.net --workers 3


