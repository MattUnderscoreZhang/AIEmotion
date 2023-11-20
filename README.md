Representations of emotion for use in AI systems.

# Theoretical Overview

## Simple Emotional Representations

There are several ways of representing emotion available in this package:

### Simple Vectoral Representation

An emotion is given as a vector in a 3D space. Each dimension can have a value from -1 to 1 inclusive, and the dimensions are:
- valence (unpleasant-pleasant): This axis represents how positive or negative an emotion is. For example, happiness has high positive valence, while sadness has high negative valence.
- arousal (deactivated-activated): This axis indicates the level of activation or energy associated with the emotion. For instance, excitement is a high-arousal emotion, while calmness is a low-arousal emotion.
- control (submissive-dominant): This axis describes the degree of control or influence a person feels they have in a given emotional state. Anger and pride are high positive, while helplessness and fear are high negative.

### Plutchik's Wheel of Emotions

Plutchik's wheel is shown below:

![Puutchi's Wheel of Emotions](https://en.m.wikipedia.org/wiki/File:Plutchik-wheel.svg)

Here Plutchik identifies eight primary emotions. These primary emotions can be combined to form secondary emotions, which can be combined to form tertiary emotions.

In our package, we represent an emotional state as an 8D vector with one dimension for each primary emotion. Furthermore, the length of the vector can range from 0 to 1 inclusive, representing the intensity of the emotion. The eight primary emotions are:
- joy
- sadness
- trust
- disgust
- fear
- anger
- surprise
- anticipation

### Physiological Representation

With inspiration from theories like James-Lange, Cannon-Bard, and Schachter-Singer, we can describe an emotional state using a physiological vector. Each dimension can range from 0 to 1 inclusive, representing the intensity of the physiological response. Here our dimensions are:
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

### Conversion Between Representations

Conversion between representations requires using an LLM. This is because the emotional interpretation of a physiological reaction is context-dependent.

## Complex Emotional Representations

Complex emotions may not be easily representable by a single vector. For example, a person may feel exhausted at work and want to watch some TV, but they may also know that watching TV would make them feel anxious and guilty. Simultaneously they may also feel a sense of pride from working on their project.

In this package we represent complex emotions in the following ways:
- Conditional Emotions: The person in this example knows that if they watch TV, they will feel a certain way. This is a prediction about how their emotional state will change in response to a certain action. Thus we provide a transition function on an emotional representation object, allowing the emotion to change in relation to a context and an action.
- Combined Emotions: The physiological representation is a unitary representation, and is a single vector regardless of how complicated a person's emotional state is. The other representations are the sum of weighted vectors.

# Examples

```
from dotenv import load_dotenv
import os
from typing import cast

from ai_emotion import VectoralEmotion, PlutchikEmotion, PhysiologicalEmotion, ComplexEmotion


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
converted_vectoral_emotion = physiological_emotion.to_vectoral(
    openai_api_key=cast(str, os.getenv("OPENAI_API_KEY")),
    context="Thinking about my project.",
)
converted_plutchik_emotion = vectoral_emotion.to_plutchik(
    openai_api_key=cast(str, os.getenv("OPENAI_API_KEY")),
    context="Getting prepared for a presentation.",
)


complex_emotion = ComplexEmotion([
    (0.7, vectoral_emotion),
    (0.3, converted_vectoral_emotion),
])
later_complex_emotion = complex_emotion.transition(
    openai_api_key=cast(str, os.getenv("OPENAI_API_KEY")),
    context="Started my presentation.",
)
```
