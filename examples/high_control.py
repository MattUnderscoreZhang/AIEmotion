import argparse
from dotenv import load_dotenv
import os
from rich import print
from typing import cast

from ai_emotion.description import describe_emotion
from ai_emotion.generation import generate_emotion
from ai_emotion.simple_emotion import VectoralEmotion


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--emotion', help='description of an emotion')
    parser.add_argument('-c', '--context', help='context of emotion', default=None)
    args = parser.parse_args()

    prompt = args.emotion
    if args.context:
        prompt += f"\n\nContext: {args.context}"
    vectoral_emotion = generate_emotion(
        emotion_type=VectoralEmotion,
        prompt=prompt,
        openai_api_key=openai_api_key,
    )

    vectoral_emotion.control = 1
    high_control_emotion_description = describe_emotion(
        vectoral_emotion,
        context=args.context,
        openai_api_key=openai_api_key,
    )

    print(
f"""
In a high-control state, the emotion is: {high_control_emotion_description}.
"""
    )
