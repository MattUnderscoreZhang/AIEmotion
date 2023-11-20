from dataclasses import asdict
import json
from typing import Type, TypeVar

from ai_emotion.simple_emotion import PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion
from ai_emotion.gpt import get_gpt_interface


A = TypeVar("A", PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion)
B = TypeVar("B", PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion)


def convert_emotion(
    emotion_a: A,
    type_b: Type[B],
    openai_api_key: str,
    context: str | None = None,
) -> B:
    interface = get_gpt_interface(openai_api_key)
    prompt = ( 
        f"""
        {emotion_a.description()}

        {type_b.description()}

        Convert the following {emotion_a.__class__.__name__} to a {type_b.__name__}:
        {asdict(emotion_a)}
        """
    )
    if context:
        prompt += f"The context for this emotion is: {context}"
    response = interface.say(prompt)
    return type_b(**json.loads(response))
