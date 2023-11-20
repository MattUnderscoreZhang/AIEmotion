from dataclasses import dataclass, asdict
import json
from typing import Type, cast

from gpt_interface import GptInterface


def get_gpt_interface(openai_api_key: str) -> GptInterface:
    interface = GptInterface(
        openai_api_key=openai_api_key,
        model='gpt-4',
        json_mode=True,
    )
    interface.set_system_message(
        """
        Create a possible conversion between emotion representations that seems reasonable.
        Give your best guesses, and respond with the emotion in JSON format.
        Don't add any extra information in your response.
        """
    )
    return interface


@dataclass
class PhysiologicalEmotion:
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

    def to_plutchik(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'PlutchikEmotion':
        return cast(
            PlutchikEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=PhysiologicalEmotion,
                type_b=PlutchikEmotion,
                emotion_a=self,
                context=context,
            ),
        )

    def to_vectoral(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'VectoralEmotion':
        return cast(
            VectoralEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=PhysiologicalEmotion,
                type_b=VectoralEmotion,
                emotion_a=self,
                context=context,
            ),
        )


@dataclass
class PlutchikEmotion:
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

    def to_physiological(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'PhysiologicalEmotion':
        return cast(
            PhysiologicalEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=PlutchikEmotion,
                type_b=PhysiologicalEmotion,
                emotion_a=self,
                context=context,
            ),
        )

    def to_vectoral(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'VectoralEmotion':
        return cast(
            VectoralEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=PlutchikEmotion,
                type_b=VectoralEmotion,
                emotion_a=self,
                context=context,
            ),
        )


@dataclass
class VectoralEmotion:
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

    def to_physiological(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'PhysiologicalEmotion':
        return cast(
            PhysiologicalEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=VectoralEmotion,
                type_b=PhysiologicalEmotion,
                emotion_a=self,
                context=context,
            ),
        )

    def to_plutchik(
        self,
        openai_api_key: str,
        context: str | None = None,
    ) -> 'PlutchikEmotion':
        return cast(
            PlutchikEmotion,
            convert_emotion(
                openai_api_key=openai_api_key,
                type_a=VectoralEmotion,
                type_b=PlutchikEmotion,
                emotion_a=self,
                context=context,
            ),
        )


SimpleEmotion = PhysiologicalEmotion | PlutchikEmotion | VectoralEmotion


def convert_emotion(
    openai_api_key: str,
    type_a: Type[SimpleEmotion],
    type_b: Type[SimpleEmotion],
    emotion_a: SimpleEmotion,
    context: str | None,
) -> SimpleEmotion:
    interface = get_gpt_interface(openai_api_key)
    prompt = ( 
        f"""
        {type_a.description()}

        {type_b.description()}

        Convert the following {type_a.__name__} to a {type_b.__name__}:
        {asdict(emotion_a)}
        """
    )
    if context:
        prompt += f"The context for this emotion is: {context}"
    response = interface.say(prompt)
    return type_b(**json.loads(response))
