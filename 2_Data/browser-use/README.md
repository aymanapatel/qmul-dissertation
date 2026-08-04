uv run


uv run main.py --no-nav-landing-page --site www.attio.com

# Multiple sites; use --workers to process them concurrently
uv run main.py --no-nav-landing-page --site www.attio.com www.example.com --workers 2
