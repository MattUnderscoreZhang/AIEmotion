from dataclasses import asdict
from rich import print

from ai_emotion.simple_emotion import VectoralEmotion


vectoral_emotion_1 = VectoralEmotion.random().normalize()
vectoral_emotion_2 = VectoralEmotion.random()
d_vectoral_emotion = (
    vectoral_emotion_2
    .normalize()
    .subtract(vectoral_emotion_1)  # .add() works the same way
)


print(
f"""
Vectoral 1: {asdict(vectoral_emotion_1)},
Vectoral 2: {asdict(vectoral_emotion_2)},
d_Vectoral: {asdict(d_vectoral_emotion)},
d_length: {d_vectoral_emotion.length()},
"""
)
