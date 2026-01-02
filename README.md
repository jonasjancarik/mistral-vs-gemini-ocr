# Mistral OCR vs Gemini 3 Flash Preview

Streamlit app to run Mistral OCR and Gemini OCR side-by-side on the same file.

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

You can also place keys in a `.env` file or paste them in the sidebar. Adjust the model names there if needed.

## Notes

- Supports PDF and common image formats.
- Mistral output is rendered as Markdown; Gemini output is plain text.
- PDFs sent to Mistral are uploaded as temporary files for OCR.
- Gemini uses the Files API for uploads and requests text-only output.
