from dataclasses import asdict
from dotenv import load_dotenv
import os
from rich import print
from typing import cast

from ai_emotion.generation import generate_emotion
from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion
from ai_emotion.transition import transition_emotion


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))


vectoral_emotion = generate_emotion(
    emotion_type=VectoralEmotion,
    prompt="Getting rejected for a job.",
    openai_api_key=openai_api_key,
)
new_vectoral_emotion = transition_emotion(
    emotion=vectoral_emotion,
    context="Getting rejected for a job.",
    prompt="Getting a better job.",
    openai_api_key=openai_api_key,
)


plutchik_emotion = generate_emotion(
    emotion_type=PlutchikEmotion,
    prompt="Seeing a cute puppy.",
    openai_api_key=openai_api_key,
)
new_plutchik_emotion = transition_emotion(
    emotion=plutchik_emotion,
    context="Seeing a cute puppy.",
    prompt="The puppy bites you.",
    openai_api_key=openai_api_key,
)


physiological_emotion = generate_emotion(
    emotion_type=PhysiologicalEmotion,
    prompt="Eating the world's hottest pepper.",
    openai_api_key=openai_api_key,
)
new_physiological_emotion = transition_emotion(
    emotion=physiological_emotion,
    context="Eating the world's hottest pepper.",
    prompt="You drink milk.",
    openai_api_key=openai_api_key,
)


print(
f"""
Getting rejected for a job, then getting a better job: {asdict(new_vectoral_emotion)}
Seeing a cute puppy, then the puppy bites you: {asdict(new_plutchik_emotion)}
Eating the world's hottest pepper, then you drink milk: {asdict(new_physiological_emotion)}
"""
)
