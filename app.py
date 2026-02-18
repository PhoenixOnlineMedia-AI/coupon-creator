# pip install streamlit fal-client requests

from __future__ import annotations

import hmac
import os
import tempfile
from typing import Any

import fal_client
import requests
import streamlit as st

FLUX_MODEL_ID = "fal-ai/flux-2-pro/edit"
HUNYUAN_MODEL_ID = "fal-ai/hunyuan-image/v3/instruct/edit"
KLING_MODEL_ID = "fal-ai/kling-image/o3/image-to-image"
GPT_IMAGE_MODEL_ID = "fal-ai/gpt-image-1.5/edit"
DARREN_TACKETT_PROMPT = """Create a high-end real estate marketing visual for Darren Tackett.
Use @image1 as the logo in the top-left corner.
Use @image2 as the logo in the top-right corner.
Use @image3 as the main property hero photo.
Keep typography clean, premium, and modern.
Add a headline that reads: "Darren Tackett".
Add a subheadline that reads: "Luxury Real Estate".
Use balanced spacing, strong contrast, and polished social-media-ready composition."""
FLUX_LIKE_ASPECT_OPTIONS: dict[str, str] = {
    "Portrait (9:16)": "portrait_16_9",
    "Portrait (3:4)": "portrait_4_3",
    "Landscape (16:9)": "landscape_16_9",
    "Landscape (4:3)": "landscape_4_3",
    "Square HD": "square_hd",
    "Square": "square",
    "Auto": "auto",
}
KLING_ASPECT_OPTIONS: dict[str, str] = {
    "9:16 (Portrait)": "9:16",
    "3:4 (Portrait)": "3:4",
    "16:9 (Landscape)": "16:9",
    "4:3 (Landscape)": "4:3",
    "1:1 (Square)": "1:1",
    "2:3 (Portrait)": "2:3",
    "3:2 (Landscape)": "3:2",
    "21:9 (Cinematic)": "21:9",
    "Auto": "auto",
}
GPT_ASPECT_OPTIONS: dict[str, str] = {
    "1024x1536 (Portrait)": "1024x1536",
    "1536x1024 (Landscape)": "1536x1024",
    "1024x1024 (Square)": "1024x1024",
    "Auto": "auto",
}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "Flux 2 Pro Edit": {
        "endpoint": FLUX_MODEL_ID,
        "aspect_options": FLUX_LIKE_ASPECT_OPTIONS,
        "default_aspect_label": "Portrait (9:16)",
        "supports_guidance_scale": False,
    },
    "Hunyuan Image v3 Instruct Edit": {
        "endpoint": HUNYUAN_MODEL_ID,
        "aspect_options": FLUX_LIKE_ASPECT_OPTIONS,
        "default_aspect_label": "Portrait (9:16)",
        "supports_guidance_scale": True,
    },
    "Kling Image O3 Image-to-Image": {
        "endpoint": KLING_MODEL_ID,
        "aspect_options": KLING_ASPECT_OPTIONS,
        "default_aspect_label": "9:16 (Portrait)",
        "supports_guidance_scale": False,
    },
    "GPT Image 1.5 Edit": {
        "endpoint": GPT_IMAGE_MODEL_ID,
        "aspect_options": GPT_ASPECT_OPTIONS,
        "default_aspect_label": "1024x1536 (Portrait)",
        "supports_guidance_scale": False,
    },
}


def upload_reference_image(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Upload a Streamlit file to Fal storage and return a public URL."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    temp_path = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        return fal_client.upload_file(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_image_url(result: Any) -> str:
    """Best-effort extraction of generated image URL from Fal response."""
    if isinstance(result, dict):
        images = result.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, dict):
                for key in ("url", "image_url"):
                    value = first.get(key)
                    if isinstance(value, str) and value:
                        return value
            if isinstance(first, str) and first:
                return first

        for key in ("image", "output_image", "result"):
            nested = result.get(key)
            if isinstance(nested, dict):
                for nested_key in ("url", "image_url"):
                    value = nested.get(nested_key)
                    if isinstance(value, str) and value:
                        return value
            if isinstance(nested, str) and nested:
                return nested

        direct_url = result.get("url")
        if isinstance(direct_url, str) and direct_url:
            return direct_url

    raise ValueError("Could not find an image URL in the Fal response.")


def require_authentication() -> bool:
    """Gate app access with APP_PASSWORD from Streamlit secrets."""
    expected_password = str(st.secrets.get("APP_PASSWORD", "")).strip()
    if not expected_password:
        st.error("Missing `APP_PASSWORD` in Streamlit secrets.")
        return False

    if st.session_state.get("is_authenticated", False):
        return True

    password_input = st.text_input("Password", type="password", key="app_password_input")
    if st.button("Login", type="primary", use_container_width=True):
        if hmac.compare_digest(password_input, expected_password):
            st.session_state["is_authenticated"] = True
            st.session_state.pop("app_password_input", None)
            st.rerun()
        else:
            st.error("Incorrect password.")

    return False


def build_generation_attempts(
    model_endpoint: str,
    prompt: str,
    image_urls: list[str],
    guidance_scale: float,
    aspect_value: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Build endpoint-specific request payloads with safe fallbacks."""
    if model_endpoint == FLUX_MODEL_ID:
        return [
            (
                FLUX_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "image_size": aspect_value,
                },
            ),
            (
                FLUX_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                },
            ),
        ]

    if model_endpoint == HUNYUAN_MODEL_ID:
        return [
            (
                HUNYUAN_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "image_size": aspect_value,
                    "guidance_scale": guidance_scale,
                },
            ),
            (
                HUNYUAN_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "image_size": aspect_value,
                },
            ),
            (
                HUNYUAN_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                },
            ),
        ]

    if model_endpoint == KLING_MODEL_ID:
        return [
            (
                KLING_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "aspect_ratio": aspect_value,
                    "output_format": "png",
                },
            ),
            (
                KLING_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "output_format": "png",
                },
            ),
        ]

    if model_endpoint == GPT_IMAGE_MODEL_ID:
        return [
            (
                GPT_IMAGE_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "image_size": aspect_value,
                    "quality": "high",
                    "output_format": "png",
                },
            ),
            (
                GPT_IMAGE_MODEL_ID,
                {
                    "prompt": prompt,
                    "image_urls": image_urls,
                    "quality": "high",
                    "output_format": "png",
                },
            ),
        ]

    raise ValueError(f"Unsupported model endpoint: {model_endpoint}")


def normalize_prompt_for_model(model_endpoint: str, prompt: str) -> str:
    """Adapt prompt reference tags to model-specific conventions."""
    if model_endpoint != KLING_MODEL_ID:
        return prompt

    return (
        prompt.replace("@image1", "@Image1")
        .replace("@image2", "@Image2")
        .replace("@image3", "@Image3")
    )


def generate_image(
    prompt: str,
    image1_file: Any,
    image2_file: Any,
    image3_file: Any,
    model_endpoint: str,
    guidance_scale: float,
    aspect_value: str,
) -> str:
    """Upload references and generate an image using selected Fal model."""
    os.environ["FAL_KEY"] = str(st.secrets["FAL_KEY"]).strip()
    normalized_prompt = normalize_prompt_for_model(model_endpoint, prompt)
    image1_url = upload_reference_image(image1_file)
    image2_url = upload_reference_image(image2_file)
    image3_url = upload_reference_image(image3_file)

    # Order maps to @image1, @image2, and @image3 in prompt text.
    image_urls = [image1_url, image2_url, image3_url]

    attempts = build_generation_attempts(
        model_endpoint=model_endpoint,
        prompt=normalized_prompt,
        image_urls=image_urls,
        guidance_scale=guidance_scale,
        aspect_value=aspect_value,
    )

    errors: list[str] = []
    for model_id, arguments in attempts:
        try:
            result = fal_client.subscribe(model_id, arguments=arguments, with_logs=False)
            return extract_image_url(result)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_id}: {exc}")

    joined_errors = " | ".join(errors)
    raise RuntimeError(f"All Fal generation attempts failed. {joined_errors}")


def main() -> None:
    st.set_page_config(page_title="Fal Multi-Model Image Generator", layout="wide")

    if not require_authentication():
        return

    st.title("Fal Multi-Model Image Generator")
    st.caption(
        "Generate a custom image from three references + prompt using Flux, Hunyuan, "
        "Kling, or GPT Image."
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            "<span style='font-size: 0.85rem; color: #6b7280;'>Powered by</span> "
            "<span style='font-size: 0.95rem; font-weight: 700;'>Fal AI Models</span>",
            unsafe_allow_html=True,
        )
        model_label = st.selectbox(
            "Model",
            options=list(MODEL_CONFIGS.keys()),
            index=0,
        )
        model_config = MODEL_CONFIGS[model_label]
        model_endpoint = str(model_config["endpoint"])
        st.caption(f"Endpoint: `{model_endpoint}`")

        with st.expander("Advanced Settings", expanded=False):
            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=10.0,
                value=3.5,
                step=0.1,
                disabled=not bool(model_config["supports_guidance_scale"]),
                help=(
                    "Used by Hunyuan edit. Higher values follow your prompt more "
                    "strictly."
                ),
            )
            aspect_options = dict(model_config["aspect_options"])
            aspect_labels = list(aspect_options.keys())
            default_aspect_label = str(model_config["default_aspect_label"])
            aspect_ratio_label = st.selectbox(
                "Aspect Ratio",
                options=aspect_labels,
                index=(
                    aspect_labels.index(default_aspect_label)
                    if default_aspect_label in aspect_labels
                    else 0
                ),
                key=f"aspect_ratio_{model_endpoint.replace('/', '_')}",
                help="Aspect ratio is applied using the selected model's valid enum.",
            )
            aspect_ratio = aspect_options[aspect_ratio_label]

    st.subheader("Image References")
    image1_file = st.file_uploader("Logo 1 (Top Left)", type=["jpg", "jpeg", "png"])
    image2_file = st.file_uploader("Logo 2 (Top Right)", type=["jpg", "jpeg", "png"])
    image3_file = st.file_uploader("Main Property Photo", type=["jpg", "jpeg", "png"])

    st.subheader("Prompt")
    st.caption(
        "Prompt references: `@image1` = top-left logo, `@image2` = top-right logo, "
        "`@image3` = main property photo. Some models also support `@Image1/@Image2`."
    )
    if "prompt_text" not in st.session_state:
        st.session_state["prompt_text"] = ""

    if st.button("Prompt Template"):
        st.session_state["prompt_text"] = DARREN_TACKETT_PROMPT

    prompt = st.text_area(
        "Describe the image to generate",
        height=240,
        key="prompt_text",
        placeholder=(
            "Example: Use @image1 for branding in top-left, @image2 for branding in "
            "top-right, and @image3 as the property hero image."
        ),
    )

    if st.button("Generate", type="primary", use_container_width=True):
        if "FAL_KEY" not in st.secrets or not str(st.secrets["FAL_KEY"]).strip():
            st.error(
                "Missing `FAL_KEY` in Streamlit secrets. Add it in your app settings "
                "before generating."
            )
            return

        if any(file is None for file in [image1_file, image2_file, image3_file]):
            st.error("Please upload all three images before generating.")
            return

        if not prompt.strip():
            st.error("Please enter a prompt before generating.")
            return

        try:
            with st.spinner("Uploading references and generating image..."):
                image_url = generate_image(
                    prompt=prompt.strip(),
                    image1_file=image1_file,
                    image2_file=image2_file,
                    image3_file=image3_file,
                    model_endpoint=model_endpoint,
                    guidance_scale=guidance_scale,
                    aspect_value=aspect_ratio,
                )

            st.success("Image generated successfully.")
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            image_bytes = image_response.content

            _, center_col, _ = st.columns([1, 2, 1])
            with center_col:
                st.image(image_bytes, caption="Generated Image", use_container_width=True)
                st.download_button(
                    label="Download Image",
                    data=image_bytes,
                    file_name=f"{model_endpoint.split('/')[-1]}-generated.png",
                    mime="image/png",
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")


if __name__ == "__main__":
    main()
