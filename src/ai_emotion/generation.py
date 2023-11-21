import json
from typing import Type, TypeVar

from ai_emotion.gpt import get_gpt_interface
from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion


A = TypeVar("A", PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion)


def generate_emotion(
    emotion_type: Type[A],
    prompt: str,
    openai_api_key: str,
) -> A:
    interface = get_gpt_interface(openai_api_key)
    response = interface.say( 
f"""
{emotion_type.description()}

Create a {A.__name__} that corresponds to the following description:
{prompt}
"""
    )
    return emotion_type(**json.loads(response))
