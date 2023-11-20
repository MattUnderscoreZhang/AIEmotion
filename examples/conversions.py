from dataclasses import asdict
from dotenv import load_dotenv
import os
from rich import print
from typing import cast

from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion
from ai_emotion.conversion import convert_emotion


vectoral_emotion = VectoralEmotion(
    valence=0.1,
    arousal=0.2,
    control=0.8,
)


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))


plutchik_emotion = convert_emotion(
    vectoral_emotion,
    PlutchikEmotion,
    openai_api_key=openai_api_key,
    context="Getting prepared for a presentation.",
)
physiological_emotion = convert_emotion(
    plutchik_emotion,
    PhysiologicalEmotion,
    openai_api_key=openai_api_key,
    context="Got attacked by a raccoon.",
)
vectoral_emotion_2 = convert_emotion(
    physiological_emotion,
    VectoralEmotion,
    openai_api_key=openai_api_key,
)
print(
f"""
Vectoral Emotion: {asdict(vectoral_emotion)}
Plutchik Emotion: {asdict(plutchik_emotion)}
Physiological Emotion: {asdict(physiological_emotion)}
Vectoral Emotion 2: {asdict(vectoral_emotion_2)}
"""
)
