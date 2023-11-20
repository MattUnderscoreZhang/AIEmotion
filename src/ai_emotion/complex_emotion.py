from dataclasses import dataclass

from ai_emotion.simple_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion


SimpleEmotion = PlutchikEmotion | PhysiologicalEmotion | VectoralEmotion


# TODO
@dataclass
class ComplexEmotion:
    emotions: list[tuple[float, SimpleEmotion]]
