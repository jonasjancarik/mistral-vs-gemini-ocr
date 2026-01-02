# Mistral vs Gemini OCR Compare

Side-by-side OCR comparison tool to help decide which model is the better fit for your documents.

## What this is good for

- Compare OCR output quality on the same file.
- Tweak Gemini prompts and config to see differences quickly.
- Estimate rough costs per doc and for 1000 similar docs.

## Setup

```bash
uv venv .venv
uv sync
```

## Run

```bash
cp .env.example .env
# then fill in API keys in .env
uv run streamlit run app.py
```

You can also paste keys in the sidebar. Adjust model names and pricing toggles there if needed.

## Notes

- Supports PDF and common image formats.
- Batch pricing toggle applies a 50% discount to both providers in the cost estimates.
