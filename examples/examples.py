from dotenv import load_dotenv
import os
from typing import cast

from ai_emotion import (
    VectoralEmotion,
    PlutchikEmotion,
    PhysiologicalEmotion,
    ComplexEmotion,
)


vectoral_emotion = VectoralEmotion(
    valence=0.1,
    arousal=0.2,
    control=0.8,
)
plutchik_emotion = PlutchikEmotion(
    joy=0.1,
    sadness=0.2,
    trust=0.1,
    disgust=0.3,
    fear=0.8,
    anger=0.6,
    surprise=0.7,
    anticipation=0.2,
)
physiological_emotion = PhysiologicalEmotion(
    heart_rate=0.1,
    breathing_rate=0.2,
    hair_raised=0.1,
    blood_pressure=0.3,
    body_temperature=0.2,
    muscle_tension=0.1,
    pupil_dilation=0.3,
    gi_blood_flow=0.6,
    amygdala_blood_flow=0.1,
    prefrontal_cortex_blood_flow=0.8,
    muscle_blood_flow=0.2,
    genitalia_blood_flow=0.0,
    smile_muscle_activity=0.0,
    brow_furrow_activity=0.0,
    lip_tightening=0.0,
    throat_tightness=0.0,
    mouth_dryness=0.1,
    voice_pitch_raising=0.1,
    speech_rate=0.3,
    cortisol_level=0.6,
    adrenaline_level=0.2,
    oxytocin_level=0.5,
)


load_dotenv()
openai_api_key = cast(str, os.getenv("OPENAI_API_KEY"))


vectoral_to_plutchik = vectoral_emotion.to_plutchik(
    openai_api_key=openai_api_key,
    context="Getting prepared for a presentation.",
)
print(vectoral_to_plutchik)
vectoral_to_physiological = vectoral_emotion.to_physiological(
    openai_api_key=openai_api_key,
    context="Getting prepared for a presentation.",
)
print(vectoral_to_physiological)
physiological_to_vectoral = physiological_emotion.to_vectoral(
    openai_api_key=openai_api_key,
    context="Thinking about my project.",
)
print(physiological_to_vectoral)
physiological_to_plutchik = physiological_emotion.to_plutchik(
    openai_api_key=openai_api_key,
    context="Thinking about my project.",
)
print(physiological_to_plutchik)
plutchik_to_vectoral = plutchik_emotion.to_vectoral(
    openai_api_key=openai_api_key,
    context="Got attacked by a raccoon.",
)
print(plutchik_to_vectoral)
plutchik_to_physiological = plutchik_emotion.to_physiological(
    openai_api_key=openai_api_key,
    context="Got attacked by a raccoon.",
)
print(plutchik_to_physiological)


# complex_emotion = ComplexEmotion([
    # (0.7, vectoral_emotion),
    # (0.3, converted_vectoral_emotion),
# ])
# later_complex_emotion = complex_emotion.transition(
    # openai_api_key=cast(str, os.getenv("OPENAI_API_KEY")),
    # context="Started my presentation.",
# )
