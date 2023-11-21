from dataclasses import asdict
import json
from typing import TypeVar

from ai_emotion.simple_emotion import PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion
from ai_emotion.gpt import get_gpt_interface


A = TypeVar("A", PhysiologicalEmotion, PlutchikEmotion, VectoralEmotion)


def transition_emotion(
    emotion: A,
    context: str,
    prompt: str,
    openai_api_key: str,
) -> A:
    interface = get_gpt_interface(openai_api_key)
    response = interface.say(
f"""
{emotion.description()}

As a cause of this:
{context}

A person feels this emotion:
{asdict(emotion)}

Determine this person's updated emotion after the following occurs:
{prompt}

Reply as a new {type(emotion).__name__}.
Make sure JSON format uses double quotes for keys.
"""
    )
    return emotion.__class__(**json.loads(response))
