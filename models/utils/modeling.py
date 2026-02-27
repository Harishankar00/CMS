# built-in dependencies
from typing import Any

# project dependencies

from ..Emotion_detection import OpenCv
from .. import Emotion


def build_model(task: str, model_name: str) -> Any:

    # singleton design pattern
    global cached_models

    models = {
        "facial_attribute": {
            "Emotion": Emotion.EmotionClient,
        },
        "face_detector": {
            "opencv": OpenCv.OpenCvClient,
        },
    }

    if models.get(task) is None:
        raise ValueError(f"unimplemented task - {task}")

    if not "cached_models" in globals():
        cached_models = {current_task: {} for current_task in models.keys()}

    if cached_models[task].get(model_name) is None:
        model = models[task].get(model_name)
        if model:
            cached_models[task][model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {task}/{model_name}")

    return cached_models[task][model_name]
