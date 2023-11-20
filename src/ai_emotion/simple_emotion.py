from dataclasses import dataclass
from typing import cast, Self
from random import random

from ai_emotion.emotion_vector import EmotionVector


@dataclass
class PhysiologicalEmotion(EmotionVector):
    heart_rate: float
    breathing_rate: float
    hair_raised: float
    blood_pressure: float
    body_temperature: float
    muscle_tension: float
    pupil_dilation: float
    gi_blood_flow: float
    amygdala_blood_flow: float
    prefrontal_cortex_blood_flow: float
    muscle_blood_flow: float
    genitalia_blood_flow: float
    smile_muscle_activity: float
    brow_furrow_activity: float
    lip_tightening: float
    throat_tightness: float
    mouth_dryness: float
    voice_pitch_raising: float
    speech_rate: float
    cortisol_level: float
    adrenaline_level: float
    oxytocin_level: float

    @classmethod
    def description(cls) -> str:
        return (
            """
            PhysiologicalEmotion is a vector with the following values, where each value is a float from 0 to 1 inclusive, representing the intensity of the physiological response:
            - heart_rate
            - breathing_rate
            - hair_raised
            - blood_pressure
            - body_temperature
            - muscle_tension
            - pupil_dilation
            - gi_blood_flow
            - amygdala_blood_flow
            - prefrontal_cortex_blood_flow
            - muscle_blood_flow
            - genitalia_blood_flow
            - smile_muscle_activity
            - brow_furrow_activity
            - lip_tightening
            - throat_tightness
            - mouth_dryness
            - voice_pitch_raising
            - speech_rate
            - cortisol_level
            - adrenaline_level
            - oxytocin_level
            """
        )

    @classmethod
    def random(cls) -> Self:
        return PhysiologicalEmotion(
            heart_rate=random(),
            breathing_rate=random(),
            hair_raised=random(),
            blood_pressure=random(),
            body_temperature=random(),
            muscle_tension=random(),
            pupil_dilation=random(),
            gi_blood_flow=random(),
            amygdala_blood_flow=random(),
            prefrontal_cortex_blood_flow=random(),
            muscle_blood_flow=random(),
            genitalia_blood_flow=random(),
            smile_muscle_activity=random(),
            brow_furrow_activity=random(),
            lip_tightening=random(),
            throat_tightness=random(),
            mouth_dryness=random(),
            voice_pitch_raising=random(),
            speech_rate=random(),
            cortisol_level=random(),
            adrenaline_level=random(),
            oxytocin_level=random(),
        )


@dataclass
class PlutchikEmotion(EmotionVector):
    joy: float
    sadness: float
    trust: float
    disgust: float
    fear: float
    anger: float
    surprise: float
    anticipation: float

    @classmethod
    def description(cls) -> str:
        return (
            """
            PlutchikEmotion is a vector with the following values, where each value is a float from 0 to 1 inclusive, representing the intensity of the emotion:
            - joy
            - sadness
            - trust
            - disgust
            - fear
            - anger
            - surprise
            - anticipation
            """
        )

    @classmethod
    def random(cls) -> Self:
        return PlutchikEmotion(
            joy=random(),
            sadness=random(),
            trust=random(),
            disgust=random(),
            fear=random(),
            anger=random(),
            surprise=random(),
            anticipation=random(),
        )


@dataclass
class VectoralEmotion(EmotionVector):
    valence: float
    arousal: float
    control: float

    @classmethod
    def description(cls) -> str:
        return (
            """
            VectoralEmotion is a vector with the following values, where each value is a float from -1 to 1 inclusive, representing the intensity of the emotion:
            - valence (negative is bad, positive is good)
            - arousal (negative is calm, positive is excited)
            - control (negative is submissive, positive is dominant)
            """
        )

    @classmethod
    def random(cls) -> Self:
        return VectoralEmotion(
            valence=random()*2-1,
            arousal=random()*2-1,
            control=random()*2-1,
        )
