smoke:
    uv run python pipeline.py smoke

generate:
    uv run python pipeline.py generate

qa:
    uv run python pipeline.py qa

export:
    uv run python pipeline.py export

verify-local:
    uv run python pipeline.py verify-local

verify-hosted:
    uv run python pipeline.py verify-hosted

test:
    uv run pytest -q
