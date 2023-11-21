from dataclasses import asdict
from dotenv import load_dotenv
import os
from rich import print
from typing import cast

from ai_emotion.generation import generate_emotion
from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))


vectoral_emotion = generate_emotion(
    emotion_type=VectoralEmotion,
    prompt="Getting rejected for a job.",
    openai_api_key=openai_api_key,
)
plutchik_emotion = generate_emotion(
    emotion_type=PlutchikEmotion,
    prompt="Seeing a cute puppy.",
    openai_api_key=openai_api_key,
)
physiological_emotion = generate_emotion(
    emotion_type=PhysiologicalEmotion,
    prompt="Eating the world's hottest pepper.",
    openai_api_key=openai_api_key,
)


print(
f"""
Getting rejected for a job: {asdict(vectoral_emotion)}
Seeing a cute puppy: {asdict(plutchik_emotion)}
Eating the world's hottest pepper: {asdict(physiological_emotion)}
"""
)
