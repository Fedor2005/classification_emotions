import os
from pathlib import Path
import numpy as np

# from predict import *
from AudioProcessing import convert_audio

from fastapi import FastAPI, UploadFile, File
# from pydantic import BaseModel
# from pydub import AudioSegment
# from sklearn.preprocessing import LabelEncoder
from starlette.responses import FileResponse, JSONResponse
# from tensorflow.keras.models import load_model

import model

app = FastAPI()


# Путь к предварительно обученной модели
MODEL_PATH = "model.keras"

# Загружаем модель
# model = load_model(MODEL_PATH)

# Предполагаем, что LabelEncoder был сохранен
# le = LabelEncoder()
# le.classes_ = np.load('classes.npy', allow_pickle=True)
#

class Task:
    file_name: str
    emotion: str
    accuracy: float
    def __init__(self, file_name, emotion, accucary):
        self.file_name = file_name
        self.emotion = emotion
        self.accuracy = accucary

# @app.get("/tasks")
# async def get_tasks():
#     task = Task(name="record this video")
#     return {"data": task}




@app.post("/predict/")
async def predict_emotion(file: UploadFile = File(...)):
    # Конвертируем аудио файл в формат .wav
    file_path = convert_audio(file)
    if not file_path:
        return JSONResponse(status_code=400, content={"message": "Error in converting the audio file."})



    # Прогнозируем эмоцию
    emotion, predictions = model.predict(file_path)
    confidence = np.max(predictions)

    os.remove(file_path)
    return {
        "emotion": emotion,
        "confidence": float(confidence)
    }
