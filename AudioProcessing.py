from pydub import AudioSegment
from fastapi import UploadFile
import os


def remove_chars_until_dot(string):
    dot_index = string.rfind('.')
    if dot_index != -1:
        return string[:dot_index + 1]
    else:
        return string


def convert_audio(file: UploadFile):
    try:
        audio = AudioSegment.from_file(file.file)
        file_path = "temp.wav"
        audio.export(file_path, format="wav")
        return file_path
    except Exception as e:
        print(f"Error encountered while converting file: {e}")
        return None