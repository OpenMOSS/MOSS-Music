from __future__ import annotations

import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import requests

from src.hf_inference import MossMusicHFInference, read_env_model_id, resolve_device

TITLE = "MOSS-Music Demo"

DEFAULT_QUESTION = "请从风格与速度、调性与和声、乐器编配、结构安排以及整体情绪几个方面描述这段音乐。"
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_BACKEND = "sglang"
DEFAULT_SGLANG_BASE_URL = "http://127.0.0.1:30100"
DEFAULT_SGLANG_API_KEY = ""
DEFAULT_REQUEST_TIMEOUT = 600


@lru_cache(maxsize=2)
def get_inference(model_name_or_path: str, device: str) -> MossMusicHFInference:
    return MossMusicHFInference(
        model_name_or_path=model_name_or_path,
        device=device,
        torch_dtype="auto",
        enable_time_marker=True,
    )


def format_status(
    backend: str,
    model_name_or_path: str,
    target: str,
    elapsed_seconds: float,
) -> str:
    return (
        f"Backend: `{backend}`  \n"
        f"Model: `{model_name_or_path}`  \n"
        f"Target: `{target}`  \n"
        f"Elapsed: `{elapsed_seconds:.2f}s`"
    )


def read_backend() -> str:
    return os.environ.get("MOSS_MUSIC_BACKEND", DEFAULT_BACKEND).strip().lower()


def normalize_sglang_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def read_sglang_base_url() -> str:
    base_url = os.environ.get("MOSS_MUSIC_SGLANG_BASE_URL", DEFAULT_SGLANG_BASE_URL)
    return normalize_sglang_base_url(base_url)


def read_sglang_api_key() -> str:
    return os.environ.get("MOSS_MUSIC_SGLANG_API_KEY", DEFAULT_SGLANG_API_KEY)


def read_request_timeout() -> int:
    return int(os.environ.get("MOSS_MUSIC_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))


def build_sglang_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = read_sglang_api_key().strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip()
    if isinstance(payload.get("generated_text"), str):
        return payload["generated_text"].strip()

    choices = payload.get("choices") or []
    if choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"].strip()

            message = first_choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_chunks.append(item.get("text", ""))
                if text_chunks:
                    return "".join(text_chunks).strip()

    raise ValueError("The SGLang response did not contain a text message.")


def generate_with_sglang(
    question: str,
    media_path: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str, str]:
    base_url = read_sglang_base_url()
    payload = {
        "text": question,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    }
    if media_path:
        payload["audio_data"] = media_path

    response = requests.post(
        f"{base_url}/generate",
        headers=build_sglang_headers(),
        json=payload,
        timeout=read_request_timeout(),
    )
    response.raise_for_status()
    model_name_or_path = (
        os.environ.get("MOSS_MUSIC_SGLANG_MODEL", "").strip()
        or read_env_model_id()
        or "default"
    )
    return extract_response_text(response.json()), model_name_or_path, f"{base_url}/generate"


def convert_media_to_mp3(media_path: str, output_path: str) -> None:
    command = [
        "ffmpeg", "-y", "-i", media_path,
        "-vn", "-acodec", "libmp3lame", output_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise gr.Error(
            "Failed to extract audio from the uploaded media.\n"
            f"{exc.stderr}"
        ) from exc


def resolve_media_path(audio_path: str | None, video_path: str | None) -> str | None:
    if video_path:
        return video_path
    return audio_path


def run_inference(
    audio_path: str | None,
    video_path: str | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    prompt = (question or "").strip() or DEFAULT_QUESTION
    media_path = resolve_media_path(audio_path, video_path)
    backend = read_backend()

    try:
        started_at = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="moss-music-") as temp_dir:
            prepared_audio_path = media_path
            if media_path and Path(media_path).suffix.lower() in VIDEO_EXTENSIONS:
                prepared_audio_path = os.path.join(temp_dir, "input.mp3")
                convert_media_to_mp3(media_path, prepared_audio_path)

            if backend == "sglang":
                answer, model_name_or_path, target = generate_with_sglang(
                    question=prompt,
                    media_path=prepared_audio_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            else:
                model_name_or_path = read_env_model_id()
                device = resolve_device()
                try:
                    inference = get_inference(model_name_or_path, device)
                except Exception as exc:  # pragma: no cover
                    raise gr.Error(
                        "Failed to load the model. Please check the weights path or "
                        f"Hugging Face download status.\n{exc}"
                    ) from exc

                answer = inference.generate(
                    question=prompt,
                    audio_path=prepared_audio_path,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                target = device
        elapsed_seconds = time.perf_counter() - started_at
    except Exception as exc:  # pragma: no cover
        raise gr.Error(
            f"{backend} inference failed. Please make sure the uploaded file is readable "
            f"and the format is supported.\n{exc}"
        ) from exc

    return answer, format_status(backend, model_name_or_path, target, elapsed_seconds)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        "A music-specialised variant of MOSS-Audio for caption, tagging, "
        "lyrics ASR, structural analysis, and musical reasoning."
    )

    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            with gr.Accordion("Optional Video Input (.mp4)", open=False):
                gr.Markdown(
                    "Upload an mp4 only when needed. If a video is provided, "
                    "its audio track will be extracted and used for inference."
                )
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4"],
                    type="filepath",
                )
            question_input = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_QUESTION,
                placeholder=(
                    "For example: Describe the mood and instrumentation; "
                    "transcribe the lyrics; what is the key and tempo?"
                ),
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens_input = gr.Slider(
                    minimum=64, maximum=2048,
                    value=DEFAULT_MAX_NEW_TOKENS, step=32,
                    label="Max New Tokens",
                )
                temperature_input = gr.Slider(
                    minimum=0.0, maximum=1.5,
                    value=DEFAULT_TEMPERATURE, step=0.1,
                    label="Temperature",
                )
                top_p_input = gr.Slider(
                    minimum=0.1, maximum=1.0,
                    value=DEFAULT_TOP_P, step=0.05,
                    label="Top-p",
                )
                top_k_input = gr.Slider(
                    minimum=1, maximum=100,
                    value=DEFAULT_TOP_K, step=1,
                    label="Top-k",
                )

            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                gr.ClearButton(
                    [
                        audio_input, video_input, question_input,
                        max_new_tokens_input, temperature_input,
                        top_p_input, top_k_input,
                    ],
                    value="Clear",
                )

        with gr.Column(scale=5):
            output_text = gr.Textbox(label="Output", lines=16)
            status_text = gr.Markdown("Waiting for input.")

    gr.Examples(
        examples=[
            ["Please give a detailed musical description of this clip."],
            ["Transcribe the lyrics with timestamps of this song."],
            ["What is the key, tempo and mood of this track?"],
            ["Transcribe the Chords progression with timestamps of this song, use json format to output the result."],
            ["Segment the song into verse / chorus / bridge sections."],
        ],
        inputs=[question_input],
        label="Prompt Examples",
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[
            audio_input, video_input, question_input,
            max_new_tokens_input, temperature_input,
            top_p_input, top_k_input,
        ],
        outputs=[output_text, status_text],
    )


if __name__ == "__main__":
    server_name = os.environ.get("MOSS_MUSIC_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("MOSS_MUSIC_SERVER_PORT", "7860"))
    demo.queue(max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
    )
