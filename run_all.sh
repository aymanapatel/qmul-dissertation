(
  cd 3_Learning &&
  .venv/bin/python -m accessibility_system.api --host 127.0.0.1 --port 8000
) &
api_pid=$!

trap 'kill "$api_pid" 2>/dev/null' EXIT INT TERM
cd 4_UI/learning-v2-demo && npm run dev