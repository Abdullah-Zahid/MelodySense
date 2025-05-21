# streamlit_app.py

import streamlit as st
import numpy as np
import librosa
import pickle
import tensorflow as tf
import tempfile
import os

import vggish_input
import vggish_slim
import vggish_params

# ─── Constants ─────────────────────────────────────────────────────
MODEL_PATH = 'final_model.keras'
LABEL_ENCODER_PATH = 'label_encoder.pkl'
VGGISH_CHECKPOINT = 'vggish_model.ckpt'
SR = 22050
DURATION = 30  # seconds

# ─── Load model and label encoder ──────────────────────────────────
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, le

# ─── Feature extraction ────────────────────────────────────────────
def extract_mfcc_and_vggish(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav",) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    y, _ = librosa.load(tmp_path, sr=SR, duration=DURATION)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # MFCC + deltas
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    d1 = librosa.feature.delta(mf)
    d2 = librosa.feature.delta(mf, order=2)
    mfcc_feature = np.mean(np.vstack([mf, d1, d2]), axis=1)

    # VGGish
    examples = vggish_input.wavfile_to_examples(tmp_path)
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT)
            inp = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            out = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding] = sess.run([out], feed_dict={inp: examples})
    vggish_feature = embedding.mean(axis=0)

    os.remove(tmp_path)  # clean up

    return mfcc_feature.reshape(1, 120, 1), vggish_feature.reshape(1, 128, 1)

# ─── Streamlit App ─────────────────────────────────────────────────
st.set_page_config(page_title="MelodySense", page_icon="🎵")
st.title("🎵 MelodySense - Music Genre Classifier")
st.markdown("Upload a `.wav` music file and get the predicted genre in real-time!")

uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav","mp3"])

if uploaded_file:
    with st.spinner("Analyzing and predicting genre..."):
        try:
            model, le = load_model_and_encoder()
            mfcc_input, vggish_input = extract_mfcc_and_vggish(uploaded_file)
            prediction = model.predict([mfcc_input, vggish_input])
            genre = le.inverse_transform([np.argmax(prediction)])[0]
            st.success(f"🎧 **Predicted Genre:** `{genre}`")

            # Optional: Show confidence scores
            st.subheader("Prediction Confidence:")
            confidence = prediction[0]
            for i, label in enumerate(le.classes_):
                st.write(f"{label}: {confidence[i]:.2%}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
