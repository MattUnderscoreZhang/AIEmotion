from dataclasses import asdict
from dotenv import load_dotenv
import os
from rich import print
from typing import cast

from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion
from ai_emotion.conversion import convert_emotion


vectoral_emotion = VectoralEmotion.random().normalize()


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))
plutchik_emotion = convert_emotion(
    vectoral_emotion,
    PlutchikEmotion,
    openai_api_key=openai_api_key,
)
vectoral_emotion_2 = convert_emotion(
    plutchik_emotion,
    VectoralEmotion,
    openai_api_key=openai_api_key,
)
d_vectoral_emotion = (
    vectoral_emotion_2
    .normalize()
    .subtract(vectoral_emotion)  # .add() works the same way
)


print(
f"""
Vectoral: {asdict(vectoral_emotion)},
Plutchik: {asdict(plutchik_emotion)},
d_Vectoral: {asdict(d_vectoral_emotion)},
d_length: {d_vectoral_emotion.length()},
"""
)
