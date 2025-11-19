# learning-the-null

Run the model selection that preceded the actual experiments:
````bash
# Basic usage
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv

# Sort by different metric
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv --metric prauc

# CSV output
uv run python -m code.utils.model_summary experiments/model_selection/breast.csv --format csv

# Multiple files
uv run python -m code.utils.model_summary experiments/model_selection/*.csv
````