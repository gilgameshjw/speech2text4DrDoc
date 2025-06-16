
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import yaml
import openai
import streamlit as st
import numpy as np
import base64
from TTS.api import TTS
from openai import OpenAI
import soundfile as sf
from io import BytesIO
from streamlit.components.v1 import html

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

letter_variables = config["letter_variables"]
openai_api_key = config["language_model"]["openai"]["api_key"]

# Set OpenAI API key
client = OpenAI(api_key=openai_api_key)

# Initialize session state
if 'letter_vars' not in st.session_state:
    st.session_state.letter_vars = letter_variables.copy()
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'tts_enabled' not in st.session_state:
    st.session_state.tts_enabled = False
if 'first_run' not in st.session_state:
    st.session_state.first_run = True  # To trigger initial prompt

# Load Coqui TTS
@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/fr/css10/vits", progress_bar=True, gpu=False)

tts = load_tts()

# Function to speak text (in-memory)
def speak_js(text):
    if not tts:
        st.warning("Le moteur TTS n'est pas chargé.")
        return

    try:
        wav_tensor = tts.tts(
            text=text,
            speaker=tts.speakers[0] if tts.speakers else None,
            language=tts.languages[0] if tts.languages else None
        )

        if isinstance(wav_tensor, list):
            wav_tensor = wav_tensor[0]

        if hasattr(wav_tensor, "cpu"):
            audio_np = wav_tensor.cpu().numpy().flatten()
        elif isinstance(wav_tensor, np.ndarray):
            audio_np = wav_tensor.flatten()
        else:
            audio_np = np.array(wav_tensor).flatten()

        with BytesIO() as wav_io:
            sf.write(wav_io, audio_np, samplerate=tts.synthesizer.output_sample_rate, format='WAV')
            wav_data_bytes = wav_io.getvalue()

        audio_b64 = base64.b64encode(wav_data_bytes).decode()

        html_audio = f"""
        <audio id="ttsAudio" controls autoplay>
          <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
          Your browser does not support the audio element.
        </audio>
        <script>
          const audio = document.getElementById('ttsAudio');
          audio.play().catch(function(error) {{
            console.log("Autoplay blocked:", error);
          }});
        </script>
        """
        html(html_audio, height=0)

    except Exception as e:
        st.error(f"Erreur dans la synthèse vocale : {str(e)}")

# Conversational agent function
def get_agent_response(conversation_history, filled_vars):
    missing_fields = [k for k, v in filled_vars.items() if not v]
    if not missing_fields:
        return "✅ Tous les champs ont été remplis ! Merci pour vos réponses."

    prompt = f"""
Vous êtes un assistant médical chargé de collecter des données pour remplir ce formulaire.
Voici les champs déjà remplis : {filled_vars}

Votre objectif est d’identifier les champs manquants et de poser des questions ou interpréter les réponses de l’utilisateur pour les remplir.

Historique de la conversation :
{conversation_history}

Répondez en français, uniquement avec une phrase à dire à l'utilisateur ou une question pour le guider.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                *[{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(conversation_history)]
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur avec l'API OpenAI : {str(e)}"

# Dummy parser - extend this later
def parse_user_input(user_text, letter_dict):
    if "alcool" in user_text.lower():
        letter_dict["consommation d'alcool"] = user_text
    if "légumes" in user_text.lower() or "fruits" in user_text.lower():
        letter_dict["consommation de fruits et légumes"] = user_text
    if "année" in user_text.lower() or "ans" in user_text.lower():
        letter_dict["âge biologique du patient"] = user_text
    if "minutes" in user_text.lower() or "sport" in user_text.lower():
        letter_dict["minutes d'activité physique par semaine"] = user_text
    return letter_dict

# UI
st.title("🧠 Assistant Épigénétique")
st.markdown("### Remplissez le formulaire via une conversation vocale")

# Initial message from agent
if st.session_state.first_run:
    initial_prompt = (
        "Bonjour ! Je vais vous aider à remplir un formulaire médical. "
        "Veuillez me parler des éléments suivants : ID du patient, consommation d'alcool, "
        "consommation de fruits et légumes, âge biologique, minutes d'activité physique par semaine."
    )
    st.session_state.conversation.append(initial_prompt)
    if st.session_state.tts_enabled:
        speak_js(initial_prompt)
    st.session_state.first_run = False

# Chat display
for i in range(0, len(st.session_state.conversation), 2):
    user_msg = st.session_state.conversation[i]
    bot_msg = st.session_state.conversation[i+1] if i+1 < len(st.session_state.conversation) else ""
    st.markdown(f"🧑‍⚕️ **Patient**: {user_msg}")
    if bot_msg:
        st.markdown(f"🤖 **Assistant**: {bot_msg}")

# Sidebar: Filled fields
st.sidebar.header("✅ Champs remplis")
for key, value in st.session_state.letter_vars.items():
    if value:
        st.sidebar.markdown(f"- **{key}**: {value}")

# Option: Enable TTS
st.sidebar.checkbox("🔊 Activer la synthèse vocale", key='tts_enabled')

# Microphone input
st.subheader("🎙️ Enregistrez votre voix")
audio_value = st.audio_input("Appuyez sur le bouton d'enregistrement et répondez")


if audio_value and 'processed_audio' not in st.session_state:
    with st.spinner("Transcription en cours..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_value
        )
        user_input = transcript.text
        st.write("🗣️ Vous avez dit :", user_input)

        # Mark audio as processed to prevent re-processing
        st.session_state.processed_audio = True

        # Update conversation and variables
        st.session_state.conversation.append(user_input)
        st.session_state.letter_vars = parse_user_input(user_input, st.session_state.letter_vars)

        # Get agent reply
        bot_reply = get_agent_response(st.session_state.conversation, st.session_state.letter_vars)
        st.session_state.conversation.append(bot_reply)

        if st.session_state.tts_enabled:
            speak_js(bot_reply)

# Reset flag at the end of the script (optional, for future inputs)
if audio_value is None:
    st.session_state.pop('processed_audio', None)



if st.sidebar.button("🔄 Réinitialiser la conversation"):
    st.session_state.conversation = []
    st.session_state.letter_vars = letter_variables.copy()
    st.session_state.first_run = True
    st.session_state.processed_audio = False
    st.experimental_rerun()
