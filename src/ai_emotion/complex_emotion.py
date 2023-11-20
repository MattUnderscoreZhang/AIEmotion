from dataclasses import dataclass

from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion, SimpleEmotion


@dataclass
class ComplexEmotion:
    emotions: list[tuple[float, SimpleEmotion]]
