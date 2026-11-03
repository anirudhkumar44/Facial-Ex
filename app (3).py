import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# Facial Expression Recognition - Streamlit Application
# CNN model prediction only (Training removed)
# ------------------------------------------------------------

st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="centered",
)

MODEL_PATH = Path("facial_expression.keras")
CLASSES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)


# ------------------------------------------------------------
# Load Saved Model
# ------------------------------------------------------------

@st.cache_resource
def load_saved_model():
    if not MODEL_PATH.exists():
        return None, None

    if not CLASSES_PATH.exists():
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    with CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names


# ------------------------------------------------------------
# Application Title
# ------------------------------------------------------------

st.title("😊 Facial Expression Recognition")

st.write(
    "Upload a face image to identify its visible facial expression "
    "using a trained Convolutional Neural Network (CNN)."
)


# ------------------------------------------------------------
# Load Model
# ------------------------------------------------------------

st.subheader("1. Load Facial Expression Model")

try:
    model, class_names = load_saved_model()

    if model is not None:
        st.success("Facial Expression Model Loaded Successfully!")
        st.caption("Available classes: " + ", ".join(class_names))
    else:
        st.error(
            "Model files not found. Please add "
            "facial_expression.keras and classes.json "
            "to the project folder."
        )

except Exception as exc:
    model = None
    class_names = None
    st.error(f"Could not load model: {exc}")


# ------------------------------------------------------------
# Upload Image
# ------------------------------------------------------------

st.divider()

st.subheader("2. Upload Face Image")

uploaded_file = st.file_uploader(
    "Choose a face image",
    type=["jpg", "jpeg", "png", "webp"],
)


# ------------------------------------------------------------
# Predict Facial Expression
# ------------------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Face Image",
            use_container_width=True
        )

        if st.button(
            "Predict Expression",
            type="primary",
            use_container_width=True
        ):

            if model is None:
                st.warning(
                    "Please add the trained model files "
                    "before making predictions."
                )

            else:

                # Resize image
                resized = image.resize(IMAGE_SIZE)

                # Convert image into NumPy array
                image_array = np.asarray(
                    resized,
                    dtype=np.float32
                )

                # Add batch dimension
                image_batch = np.expand_dims(
                    image_array,
                    axis=0
                )

                # Model prediction
                probabilities = model.predict(
                    image_batch,
                    verbose=0
                )[0]

                # Get predicted class
                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_label = class_names[predicted_index]

                confidence = (
                    float(probabilities[predicted_index]) * 100
                )

                # Display result
                st.divider()

                st.subheader("3. Prediction Result")

                st.success(
                    f"Predicted Expression: "
                    f"**{predicted_label.title()}**"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                # Display all class probabilities
                st.write("### Expression Probabilities")

                sorted_indices = np.argsort(probabilities)[::-1]

                for index in sorted_indices:

                    label = class_names[int(index)]

                    score = float(probabilities[index])

                    st.write(
                        f"**{label.title()}** — "
                        f"{score * 100:.2f}%"
                    )

                    st.progress(score)

    except Exception as exc:

        st.error(
            f"Could not process this image: {exc}"
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Facial Expression Recognition using CNN. "
    "This is an educational image-classification project. "
    "Predictions may be inaccurate and do not establish "
    "a person's actual feelings or mental state. "
    "Use images only with appropriate permission."
)
```
