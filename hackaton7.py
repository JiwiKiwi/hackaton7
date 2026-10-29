import io
import threading
from pathlib import Path
import requests

import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from rfdetr import RFDETRMedium


# Set up the app.
st.set_page_config(
    page_title="Eyes on Earth",
    layout="centered",
)

st.title("Eyes on Earth")
st.write("Detect cigarette butts in photos of green spaces.")
st.caption(
    "The model runs on your computer. "
    "Photos are not sent to Roboflow."
)

WEIGHTS = Path(__file__).resolve().parent / "weights.pt"


# Load the local model once and reuse it.
@st.cache_resource
def load_model(weights_path, modified_time):
    model = RFDETRMedium(
        pretrain_weights=str(weights_path),
        num_classes=1,
        device="cpu",
    )

    return model, threading.Lock()


# Check whether the model file exists.
if not WEIGHTS.is_file():
    try:
        with st.spinner("Downloading model weights..."):
            headers = {
                "Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}",
                "Accept": "application/vnd.github+json",
            }

            release_url = (
                "https://api.github.com/repos/"
                "JiwiKiwi/AI-For-Good-JalieVermeegen/"
                "releases/tags/hackaton"
            )

            response = requests.get(
                release_url,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            asset = next(
                (
                    item
                    for item in response.json()["assets"]
                    if item["name"] == "weights.pt"
                ),
                None,
            )

            if asset is None:
                raise RuntimeError(
                    "weights.pt was not found in the GitHub release."
                )

            download_headers = {
                **headers,
                "Accept": "application/octet-stream",
            }

            temporary_file = WEIGHTS.with_suffix(".pt.part")

            try:
                with requests.get(
                    asset["url"],
                    headers=download_headers,
                    stream=True,
                    timeout=(30, 300),
                ) as download:
                    download.raise_for_status()

                    with temporary_file.open("wb") as output:
                        for chunk in download.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                output.write(chunk)

                if temporary_file.stat().st_size != asset["size"]:
                    raise RuntimeError("The download was incomplete.")

                temporary_file.replace(WEIGHTS)

            finally:
                temporary_file.unlink(missing_ok=True)

    except Exception:
        st.error(
            "Could not download the model. Check the GitHub token, "
            "repository access, release tag and weights filename."
        )
        st.stop()

confidence = st.slider(
    "Minimum confidence (%)",
    min_value=1,
    max_value=99,
    value=47,
)

uploaded_file = st.file_uploader(
    "Upload a photo",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.stop()

# Open the photo and correct its orientation.
try:
    image = ImageOps.exif_transpose(
        Image.open(uploaded_file)
    ).convert("RGB")

except Exception:
    st.error("This file could not be opened as an image.")
    st.stop()

st.image(
    image,
    caption="Original photo",
    use_container_width=True,
)

if st.button("Detect cigarette butts", type="primary"):
    try:
        with st.spinner("Loading the model and analysing your photo..."):
            model, lock = load_model(
                str(WEIGHTS),
                WEIGHTS.stat().st_mtime_ns,
            )

            # Run one prediction at a time.
            with lock:
                detections = model.predict(
                    image,
                    threshold=confidence / 100,
                )

        result_image = image.copy()
        draw = ImageDraw.Draw(result_image)

        line_width = max(
            2,
            round(min(image.size) / 250),
        )

        # Draw a box and confidence score for each detection.
        for box, score in zip(
            detections.xyxy,
            detections.confidence,
        ):
            left, top, right, bottom = map(float, box)

            draw.rectangle(
                [left, top, right, bottom],
                outline="#ff3366",
                width=line_width,
            )

            draw.text(
                (max(0, left), max(0, top - 14)),
                f"Cigarette butt {score:.0%}",
                fill="#ff3366",
            )

        # Show the count and annotated photo.
        st.metric(
            "Possible cigarette butts detected",
            len(detections),
        )

        st.image(
            result_image,
            caption="Detection result",
            use_container_width=True,
        )

        st.caption(
            "Check the results yourself. "
            "No detections does not mean the area is clean."
        )

        # Keep the result available for downloading.
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")

        st.session_state["last_result"] = (
            uploaded_file.getvalue(),
            confidence,
            buffer.getvalue(),
        )

    except Exception as error:
        st.error(
            "The model could not process the photo. "
            "See the error below."
        )
        st.exception(error)

saved = st.session_state.get("last_result")

# Only offer the download for the current photo and threshold.
if (
    saved
    and saved[0] == uploaded_file.getvalue()
    and saved[1] == confidence
):
    st.download_button(
        label="Download result",
        data=saved[2],
        file_name="detections.png",
        mime="image/png",
    )
