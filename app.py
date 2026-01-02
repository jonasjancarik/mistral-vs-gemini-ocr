import base64
import io
import hashlib
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

import streamlit as st
from google import genai
from google.genai import types
from mistralai import Mistral

load_dotenv()

SUPPORTED_TYPES = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "tif",
    "tiff",
    "bmp",
    "webp",
]

GEMINI_INPUT_COST_PER_1M = 0.50
GEMINI_OUTPUT_COST_PER_1M = 3.00
MISTRAL_COST_PER_1000_PAGES = 1.00


def guess_mime_type(file_name: str, file_bytes: bytes) -> str:
    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type:
        return mime_type
    if file_bytes.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def format_markdown_preview(text: str, preserve_line_breaks: bool) -> str:
    if not preserve_line_breaks:
        return text
    return re.sub(r"(?<!\n)\n(?!\n)", "  \n", text)


def format_usd(amount: float, precision: int = 4) -> str:
    return f"${amount:,.{precision}f}"


def extract_gemini_text_and_thoughts(
    response: types.GenerateContentResponse,
) -> tuple[str, str]:
    text_parts = []
    thought_parts = []
    for candidate in response.candidates or []:
        content = candidate.content
        if not content or not content.parts:
            continue
        for part in content.parts:
            if not part.text:
                continue
            if part.thought:
                thought_parts.append(part.text)
            else:
                text_parts.append(part.text)

    text = "\n".join(text_parts).strip()
    if not text:
        text = (response.text or "").strip()
    thoughts = "\n".join(thought_parts).strip()
    return text, thoughts


def gemini_cost_summary(
    usage: object | None,
    include_thinking_in_output: bool,
) -> dict[str, float | int] | None:
    if not usage:
        return None
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    response_tokens = getattr(usage, "response_token_count", None)
    if response_tokens is None:
        response_tokens = getattr(usage, "candidates_token_count", 0) or 0
    thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", None)
    if total_tokens is None:
        total_tokens = prompt_tokens + response_tokens + thoughts_tokens
    output_tokens = response_tokens + (thoughts_tokens if include_thinking_in_output else 0)
    input_cost = (prompt_tokens / 1_000_000) * GEMINI_INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * GEMINI_OUTPUT_COST_PER_1M
    total_cost = input_cost + output_cost
    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "thoughts_tokens": thoughts_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "input_cost_1000": input_cost * 1000,
        "output_cost_1000": output_cost * 1000,
        "total_cost_1000": total_cost * 1000,
    }


def run_mistral_ocr(
    file_bytes: bytes,
    mime_type: str,
    file_name: str,
    api_key: str,
    model: str,
) -> tuple[str, int]:
    client = Mistral(api_key=api_key)
    if mime_type.startswith("image/"):
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{encoded}"
        document = {"type": "image_url", "image_url": data_url}
        response = client.ocr.process(model=model, document=document)
    else:
        upload = client.files.upload(
            file={
                "file_name": file_name,
                "content": file_bytes,
                "content_type": mime_type,
            },
            purpose="ocr",
        )
        try:
            response = client.ocr.process(
                model=model,
                document={"type": "file", "file_id": upload.id},
            )
        finally:
            try:
                client.files.delete(upload.id)
            except Exception:
                pass

    pages = [page.markdown for page in response.pages]
    return "\n\n".join(pages).strip(), len(response.pages)


def run_gemini_ocr(
    file_bytes: bytes,
    mime_type: str,
    file_name: str,
    api_key: str,
    model: str,
    media_resolution: types.MediaResolution,
    thinking_level: types.ThinkingLevel | None,
    system_instruction: str,
    prompt: str,
    temperature: float,
    response_mime_type: str | None,
    include_thoughts: bool,
) -> tuple[str, types.UsageMetadata | None, str]:
    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(
        file=io.BytesIO(file_bytes),
        config=types.UploadFileConfig(mime_type=mime_type, display_name=file_name),
    )
    thinking_config = None
    if thinking_level or include_thoughts:
        thinking_config = types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type=response_mime_type,
        media_resolution=media_resolution,
        system_instruction=system_instruction,
        thinking_config=thinking_config,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
            config=config,
        )
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    text, thoughts = extract_gemini_text_and_thoughts(response)
    return text, response.usage_metadata, thoughts


st.set_page_config(page_title="Mistral OCR vs Gemini Flash", layout="wide")

st.title("Mistral OCR vs Gemini 3 Flash Preview")
st.write(
    "Upload a PDF or image and run both OCR models side-by-side. "
    "Outputs are shown below with basic timing and word counts."
)

with st.sidebar:
    st.header("Settings")
    mistral_api_key = st.text_input(
        "Mistral API key",
        type="password",
        value=os.getenv("MISTRAL_API_KEY", ""),
    )
    gemini_api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
    )
    mistral_model = st.text_input("Mistral model", value="mistral-ocr-latest")
    gemini_model = st.text_input("Gemini model", value="gemini-3-flash-preview")
    with st.expander("Gemini prompt & config", expanded=True):
        default_system_instruction = (
            "You are an OCR engine. Return only extracted text from the provided "
            "document. Preserve line breaks and basic structure. "
            "Do not add commentary, explanations, or summaries."
        )
        gemini_system_instruction = st.text_area(
            "System instruction",
            value=default_system_instruction,
            height=140,
        )
        gemini_prompt = st.text_area(
            "Prompt",
            value="Extract all text from the document.",
            height=100,
        )
        gemini_temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
        )
        gemini_response_mime_type = st.selectbox(
            "Response MIME type",
            options=["text/plain", "text/markdown", "application/json", "None"],
            index=0,
        )
        gemini_include_thoughts = st.checkbox("Include model thoughts", value=False)
        count_thoughts_in_cost = st.checkbox(
            "Count thinking tokens in cost",
            value=True,
        )
    gemini_resolution = st.selectbox(
        "Gemini media resolution",
        options=["HIGH", "MEDIUM", "LOW"],
        index=0,
    )
    gemini_thinking = st.selectbox(
        "Gemini thinking level",
        options=["MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"],
        index=0,
    )
    preserve_line_breaks = st.checkbox("Preserve line breaks in preview", value=True)
    show_raw = st.checkbox("Show raw outputs", value=False)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=SUPPORTED_TYPES,
    help="Accepted: PDF, PNG, JPG, TIFF, BMP, WEBP",
)

run_button = st.button("Run OCR", type="primary", disabled=uploaded_file is None)

if run_button:
    if not uploaded_file:
        st.error("Please upload a file.")
    elif not mistral_api_key or not gemini_api_key:
        st.error("Please provide both API keys (or set env vars).")
    else:
        file_bytes = uploaded_file.getvalue()
        mime_type = guess_mime_type(uploaded_file.name, file_bytes)
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        mistral_cache_key = f"{file_hash}:{mistral_model}"
        if "mistral_cache" not in st.session_state:
            st.session_state["mistral_cache"] = {}
        mistral_cache = st.session_state["mistral_cache"]
        mistral_cached = mistral_cache_key in mistral_cache

        st.session_state["results"] = {
            "file_name": uploaded_file.name,
            "mime_type": mime_type,
            "mistral": None,
            "gemini": None,
            "mistral_time": None,
            "gemini_time": None,
            "mistral_cached": mistral_cached,
            "mistral_pages": None,
            "gemini_usage": None,
            "gemini_thoughts": "",
        }

        def timed_call(fn, *args):
            start = time.perf_counter()
            result = fn(*args)
            return result, time.perf_counter() - start

        with st.spinner("Running OCR..."):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                if not mistral_cached:
                    futures["mistral"] = executor.submit(
                        timed_call,
                        run_mistral_ocr,
                        file_bytes,
                        mime_type,
                        uploaded_file.name,
                        mistral_api_key,
                        mistral_model,
                    )
                futures["gemini"] = executor.submit(
                    timed_call,
                    run_gemini_ocr,
                    file_bytes,
                    mime_type,
                    uploaded_file.name,
                    gemini_api_key,
                    gemini_model,
                    types.MediaResolution[f"MEDIA_RESOLUTION_{gemini_resolution}"],
                    None if gemini_thinking == "NONE" else types.ThinkingLevel[gemini_thinking],
                    gemini_system_instruction.strip(),
                    gemini_prompt.strip(),
                    gemini_temperature,
                    None if gemini_response_mime_type == "None" else gemini_response_mime_type,
                    gemini_include_thoughts,
                )
                if mistral_cached:
                    cached = mistral_cache[mistral_cache_key]
                    mistral_text = cached["text"]
                    mistral_time = cached["time"]
                    mistral_pages = cached["pages"]
                else:
                    (mistral_text, mistral_pages), mistral_time = futures["mistral"].result()
                    mistral_cache[mistral_cache_key] = {
                        "text": mistral_text,
                        "time": mistral_time,
                        "pages": mistral_pages,
                    }

                (gemini_text, gemini_usage, gemini_thoughts), gemini_time = futures["gemini"].result()

        st.session_state["results"]["mistral"] = mistral_text
        st.session_state["results"]["gemini"] = gemini_text
        st.session_state["results"]["mistral_time"] = mistral_time
        st.session_state["results"]["gemini_time"] = gemini_time
        st.session_state["results"]["mistral_pages"] = mistral_pages
        st.session_state["results"]["gemini_usage"] = gemini_usage
        st.session_state["results"]["gemini_thoughts"] = gemini_thoughts

results = st.session_state.get("results")

if results:
    st.subheader("Results")
    left, right = st.columns(2)

    with left:
        st.markdown("### Mistral OCR")
        mistral_caption = (
            f"{results['mistral_time']:.2f}s | "
            f"{word_count(results['mistral'])} words | "
            f"{len(results['mistral'])} chars"
        )
        if results.get("mistral_cached"):
            mistral_caption = f"{mistral_caption} | cached"
        st.caption(mistral_caption)
        mistral_pages = results.get("mistral_pages") or 0
        mistral_cost = (mistral_pages / 1000) * MISTRAL_COST_PER_1000_PAGES
        mistral_cost_1000 = mistral_pages * MISTRAL_COST_PER_1000_PAGES
        st.markdown(
            f"**Cost estimate**  \n"
            f"Pages: `{mistral_pages}`  \n"
            f"Per doc: {format_usd(mistral_cost)}  \n"
            f"1000 docs: {format_usd(mistral_cost_1000, precision=2)}"
        )
        if results["mistral"]:
            st.markdown(format_markdown_preview(results["mistral"], preserve_line_breaks))
        else:
            st.info("No text returned.")

        st.download_button(
            "Download Mistral output",
            data=results["mistral"],
            file_name="mistral_ocr.txt",
        )

        if show_raw:
            st.code(results["mistral"], language="markdown")

    with right:
        st.markdown("### Gemini 3 Flash Preview")
        st.caption(
            f"{results['gemini_time']:.2f}s | "
            f"{word_count(results['gemini'])} words | "
            f"{len(results['gemini'])} chars"
        )
        gemini_costs = gemini_cost_summary(
            results.get("gemini_usage"),
            include_thinking_in_output=count_thoughts_in_cost,
        )
        if gemini_costs:
            st.markdown(
                f"**Cost estimate**  \n"
                f"Input tokens: `{gemini_costs['prompt_tokens']}`  \n"
                f"Output tokens: `{gemini_costs['response_tokens']}`  \n"
                f"Thinking tokens: `{gemini_costs['thoughts_tokens']}`  \n"
                f"Billable output tokens: `{gemini_costs['output_tokens']}`  \n"
                f"Total tokens: `{gemini_costs['total_tokens']}`  \n"
                f"Per doc: {format_usd(gemini_costs['total_cost'])} "
                f"(in {format_usd(gemini_costs['input_cost'])} + "
                f"out {format_usd(gemini_costs['output_cost'])})  \n"
                f"1000 docs: {format_usd(gemini_costs['total_cost_1000'], precision=2)}"
            )
        else:
            st.caption("Token usage not available for this Gemini response.")
        if results["gemini"]:
            st.markdown(format_markdown_preview(results["gemini"], preserve_line_breaks))
        else:
            st.info("No text returned.")

        st.download_button(
            "Download Gemini output",
            data=results["gemini"],
            file_name="gemini_ocr.txt",
        )

        if show_raw:
            st.code(results["gemini"], language="markdown")
        if results.get("gemini_thoughts"):
            with st.expander("Gemini thoughts"):
                st.code(results["gemini_thoughts"], language="text")
