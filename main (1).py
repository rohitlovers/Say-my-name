from ast import With
from typing import Final
import streamlit as st
import pyttsx3
from pyttsx3.engine import Engine
from gtts import gTTS
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Tokenizer,
)  # Transformers, a State-of-the-art Natural Language
from scipy.io import wavfile
import eng_to_ipa as p
import torch
import librosa
import numpy as np
import os


MODEL_NAME: Final[str] = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


def osx_convert_text_to_speech(text: str, file_name: str = "output.wav") -> None:
    """
    Purpose: Convert text to speech on macOS.

    Args:
        text (str): _description_
        file_name (str, optional): _description_. Defaults to "output.wav".
    """

    print("Initializing Text-to-speech engine...")
    tts = gTTS(text=text, lang="en", slow=False)

    print(f"Converting text to speech: {text}")
    tts.save(savefile=file_name)

    print("Done!")


# end def


def win_convert_text_to_speech(text: str, file_name: str = "output.wav"):
    """
    Purpose: Convert text to speech on Windows.

    Args:
        text (str): _description_
        file_name (str, optional): _description_. Defaults to "output.wav".
    """

    print("Initializing Text-to-speech engine...")
    engine: Engine = pyttsx3.init(driverName="nsss")

    print(f"Converting text to speech: {text}")
    engine.save_to_file(text, filename=file_name)

    rate = engine.getProperty("rate")
    engine.setProperty("rate", rate - 100)
    print(f"Rate: {rate}")

    engine.runAndWait()
    print("Done!")


# end def


def convert_text_to_speech(text: str) -> None:
    """
    Purpose: Convert text to speech.

    Parameters:
        text: str - text to convert to speech.

    Returns: None
    """

    file_name = "output.mp3"

    match os.name:
        # comment:
        case "nt":
            win_convert_text_to_speech(text=text, file_name=file_name)
        case "posix":
            osx_convert_text_to_speech(text=text, file_name=file_name)

            from pydub import AudioSegment

            sound = AudioSegment.from_mp3(file_name)
            sound.export("output.wav", format="wav")

        case _:
            raise Exception("OS not supported")
    # end match

    st.text("Play")
    st.audio(data=file_name)


# end def


def convert_speech_to_text(file_name: str) -> None:
    """
    Purpose: arg
    """
    print("Initializing Speech-to-text engine...")

    # Sample rate is the number of samples per second that are taken of a waveform to create a discrete digital signal.
    # The higher the sample rate, the more snapshots you capture of the audio signal.
    # The audio sample rate is measured in kilohertz (kHz) and it determines the range of frequencies captured in digital audio
    print(f"Reading audio file...{file_name}")
    data = wavfile.read(file_name)
    framerate = data[0]
    sounddata = data[1]
    time = np.arange(0, len(sounddata)) / framerate
    print("Sampling rate:", framerate, "Hz")

    # We have to convert the sample rate to 16000 Hz as Facebook’s model accepts the sampling rate at this range.
    # using Librosa
    input_audio, _ = librosa.load(file_name, sr=16000)

    # The input audio is fed to the tokenizer which is then processed by the model.
    # The final result will be stored in the transcription variable
    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # transcribes the audio to text
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    st.write(f"Transcription: {transcription}")

    # convert the text to phonetics using IPA phonemes.
    phonetic = p.convert(transcription)
    st.text(f"Phonetic: {phonetic}")


# end def


def init_page_config() -> None:
    """
    Purpose: Set the title and description of the app.

    Returns: None
    """
    title: Final[str] = "Speech-to-Text / Text-to-Speech"
    icon: Final[str] = "📃/🎙️"

    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="centered",
    )

    st.title(f"Say My Name App")
    st.header(f"Convert speech to text and text to speech")


# end def


def main() -> None:
    """
    Purpose: Main function for the program.
    """

    # set title and description of the app.
    init_page_config()

    tab1, tab2 = st.tabs(["Speech-to-Text 📃", "Text-to-Speech 🎙️"])

    with tab1:
        tab1.title(f"Speech to Text 📃")

        audio_file = tab1.file_uploader(
            "Upload an WAV audio file", type=["wav"], accept_multiple_files=False
        )
        if audio_file:
            try:
                tab1.audio(audio_file)

                with open("output.wav", "wb") as f:
                    f.write(audio_file.getbuffer())

                # end with
                convert_speech_to_text(file_name="output.wav")
            except Exception as e:
                tab1.error(body=e)

    with tab2:
        tab2.title(f"Text-to-Speech 🎙️")
        tab2.subheader(f"Convert any text to speech")

        name: str = st.text_input(
            label="Text",
            key="name",
            placeholder="Please enter text to convert to speech",
        )

        if name:
            try:
                convert_text_to_speech(text=name)

            except Exception as e:
                tab2.error(body=e)


# end def


if __name__ == "__main__":
    main()
# end main
