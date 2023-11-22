from dataclasses import asdict

from ai_emotion.gpt import get_gpt_interface
from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion


def describe_emotion(
    emotion: PhysiologicalEmotion | PlutchikEmotion | VectoralEmotion,
    openai_api_key: str,
    context: str | None = None,
) -> str:
    interface = get_gpt_interface(openai_api_key)
    prompt = (
f"""
{emotion.__class__.description()}

Describe the following emotion in words.
First describe it in one or two words, then in detail (e.g. "Anxious. Nervous about an upcoming presentation, and worried that I may perform poorly.").
Don't describe the individual components of the emotion.
Emotion:
{asdict(emotion)}
"""
    )
    if context:
        prompt += (
f"""
The context is:
{context}
"""
        )
    response = interface.say(prompt)
    return response
