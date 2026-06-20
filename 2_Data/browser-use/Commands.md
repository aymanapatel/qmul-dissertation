# Browser-Use + Guidepup Accessibility Testing Commands

This document lists all commands required to set up, run, and troubleshoot the **axe-core** and **Guidepup (VoiceOver)** accessibility testing pipeline.

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
- Installs `playwright`, `@guidepup/guidepup`, `@guidepup/playwright`
- Installs TypeScript compiler and `ts-node` for running the runner script

**Required packages:**
- `playwright` — Browser automation library
- `@guidepup/guidepup` — VoiceOver screen reader driver for macOS
- `@guidepup/playwright` — Playwright integration for Guidepup
- `typescript` + `ts-node` — TypeScript execution environment

### 2. Configure VoiceOver Automation Permissions (macOS Only)

```bash
npx @guidepup/setup
```

**What this does:**
- Grants Terminal / Node.js permission to control VoiceOver via AppleScript
- Enables screen reader automation on macOS
- **This is mandatory** — without it, Guidepup cannot start VoiceOver

**You may need to:**
- Go to **System Settings → Privacy & Security → Accessibility**
- Add your terminal emulator (e.g., Terminal.app, iTerm.app, VS Code)
- Enable the permission

### 3. Verify Python Environment

Ensure your Python virtual environment is active and has the required packages:

```bash
source .venv/bin/activate
python -c "from browser_use import Agent; print('browser-use OK')"
python -c "import guidepup, guidepup_llm; print('Guidepup modules OK')"
```

---

## Running Tests

### Basic axe-core Accessibility Audit

Run only the automated axe-core WCAG checks (no screen reader):

```bash
python main.py --site example.com
```

### axe-core + Guidepup VoiceOver Tests

Run axe-core **and** VoiceOver screen reader navigation tests:

```bash
python main.py --site example.com --guidepup
```

**What happens:**
1. Browser Use logs into the site
2. Extracts navigation links (up to 15 pages)
3. Runs axe-core on each page
4. Extracts a11y DOM metadata and sends it to the LLM
5. LLM generates a custom VoiceOver test plan for each page
6. VoiceOver navigates the page and records spoken phrases
7. Saves JSON reports for both axe-core and Guidepup results

### Muted VoiceOver Tests

Run Guidepup tests with system audio muted (no VoiceOver speech audible):

```bash
python main.py --site example.com --guidepup --guidepup-mute
```

**What this does:**
- Mutes macOS system volume before starting VoiceOver
- Unmutes after tests complete (or on crash)
- Useful for CI environments or when you don't want audio output

### Sign Up Instead of Log In

Create a new account and test the signup flow:

```bash
python main.py --site example.com --signup
```

### Sign Up + VoiceOver

Test the signup flow with screen reader coverage:

```bash
python main.py --site example.com --signup --guidepup --guidepup-mute
```

### Skip Navigation Link Extraction

Only test the current page after login (don't crawl navigation links):

```bash
python main.py --site example.com --no-nav --guidepup
```

### Run from CSV List

Test multiple domains listed in `temp.csv`:

```bash
python main.py --guidepup --guidepup-mute
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
└── guidepup/                          # VoiceOver screen reader reports
    ├── plans/                         # Cached LLM test plans
    │   ├── a3f7b2c1.json
    │   └── 9e1d4a8f.json
    └── example.com/
        ├── page-0_home_guidepup.json
        ├── page-1_dashboard_guidepup.json
        ├── page-2_settings_guidepup.json
        └── summary.json
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

**Guidepup report** (`page-{i}_{slug}_guidepup.json`):
```json
{
  "url": "https://example.com/dashboard",
  "stepsExecuted": 8,
  "actions": [
    {
      "step": 1,
      "command": "navigateToWebContent",
      "spoken": "Dashboard, web content",
      "itemText": "Dashboard web content"
    },
    {
      "step": 2,
      "command": "findNextHeading",
      "spoken": "Welcome, heading level 1",
      "itemText": "Welcome heading level 1"
    }
  ],
  "summary": {
    "headingsFound": 4,
    "landmarksFound": 2,
    "controlsFound": 3,
    "linksFound": 5,
    "tabStops": 6
  }
}
```

---

## Troubleshooting

### VoiceOver Not Starting

**Error:** `ERR_VOICE_OVER_NOT_RUNNING`

**Fix:**
```bash
# Re-run the setup command
npx @guidepup/setup

# Check System Settings → Privacy & Security → Accessibility
# Ensure your terminal app is listed and enabled
```

### TypeScript Compilation Errors

**Check compilation:**
```bash
npx tsc --noEmit -p tsconfig.json
```

If there are errors in `node_modules`, they are usually harmless due to `skipLibCheck`. The runner should still execute.

### Playwright Cannot Connect to CDP

**Error:** `connectOverCDP failed`

**Fix:**
- Ensure Browser Use is running with `headless=False` when `--guidepup` is used
- Check that the CDP URL is valid and the browser session is active
- The browser session may have expired — re-run `main.py`

### Mute/Unmute Not Working

**Test manually:**
```bash
# Mute
osascript -e "set volume with output muted"

# Unmute
osascript -e "set volume without output muted"
```

If these fail, you may need to grant Script Editor or Terminal permission in **System Settings → Privacy & Security → Automation**.

### Guidepup Runner Returns Invalid JSON

**Debug the runner directly:**
```bash
npx ts-node guidepup-runner.ts \
  --cdp-url "ws://127.0.0.1:9222" \
  --url "https://example.com" \
  --plan '{"steps":[{"action":"navigateToWebContent","objective":"Enter page content"}]}'
```

### No Audio from VoiceOver

- Check System Volume is not at 0
- If using `--guidepup-mute`, audio is intentionally suppressed
- VoiceOver must be enabled in **System Settings → Accessibility → VoiceOver**

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
  │    │
  │    └─► Guidepup audit (if --guidepup):
  │         ├─► Extract DOM a11y metadata
  │         ├─► Send to LLM → generate test plan
  │         ├─► [Cache plan by DOM hash]
  │         ├─► Launch guidepup-runner.ts (Node.js)
  │         │    ├─► Playwright connects to same CDP browser
  │         │    ├─► VoiceOver.start()
  │         │    ├─► Execute test plan
  │         │    └─► Return JSON spoken phrases
  │         └─► Save report
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
| `main.py` | Python | CLI entrypoint, orchestrates Browser Use agent, axe-core, and Guidepup tests |
| `axe.py` | Python | axe-core injection, execution, report saving |
| `guidepup_llm.py` | Python | DOM metadata extraction, LLM test plan generation, plan caching |
| `guidepup.py` | Python | Node.js bridge, invokes runner, saves reports, mutes/unmutes audio |
| `guidepup-runner.ts` | TypeScript | Playwright CDP connection, VoiceOver control, test plan execution |
| `auth.py` | Python | Login/signup helpers, password management |
| `package.json` | JSON | Node.js dependencies |
| `tsconfig.json` | JSON | TypeScript compiler configuration |

---

## Quick Reference Card

```bash
# Setup (run once)
cd 2_Data/browser-use && npm install
npx @guidepup/setup

# axe-core only
python main.py --site example.com

# axe-core + VoiceOver (audible)
python main.py --site example.com --guidepup

# axe-core + VoiceOver (muted)
python main.py --site example.com --guidepup --guidepup-mute

# Signup flow + VoiceOver
python main.py --site example.com --signup --guidepup --guidepup-mute

# Single page only
python main.py --site example.com --no-nav --guidepup

# Multiple sites from CSV
python main.py --guidepup --guidepup-mute
```
