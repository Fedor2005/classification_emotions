import torch
import onnx
import toml
import librosa
import lazycon
import time
import os, sys

import coremltools as ct
import numpy as np
import soundfile as sf
# import sounddevice as sd

sys.path.append('../experiments')
import core

def create_features(
    data: np.array,
    hop_length_coef: float = 0.01,
    win_length_coef: float = 0.02,
    sample_rate: int = 16000,
    n_mels: int = 64,
) -> np.array:
    """
    As an input all models use standard speech features:
    64 Mel-filterbank calculated from 20ms windows with a 10ms overlap.
    """

    hop_length = int(sample_rate * hop_length_coef)
    win_length = int(sample_rate * win_length_coef)
    if len(data) != 0:
        spec = librosa.feature.melspectrogram(
                    y=data,
                    sr=sample_rate,
                    hop_length=hop_length,
                    n_fft=win_length,
                    n_mels=n_mels,
                )
    else:
        raise AttributeError
    mel_spec = librosa.power_to_db(spec, ref=np.max)

    return mel_spec
def create_features_for_audio(
    wav_name: str,
    hop_length_coef: float = 0.01,
    win_length_coef: float = 0.02,
    sample_rate: int = 16000,
    n_mels: int = 64,
) -> np.array:
    """
    As an input all models use standard speech features:
    64 Mel-filterbank calculated from 20ms windows with a 10ms overlap.
    """

    hop_length = int(sample_rate * hop_length_coef)
    win_length = int(sample_rate * win_length_coef)
    data, rate = librosa.load(wav_name, sr=sample_rate)
    if len(data) != 0:
        spec = librosa.feature.melspectrogram(
                    y=data,
                    sr=rate,
                    hop_length=hop_length,
                    n_fft=win_length,
                    n_mels=n_mels,
                )
    else:
        raise AttributeError
    mel_spec = librosa.power_to_db(spec, ref=np.max)
    return np.float64(mel_spec)


def index2name(
        index: int
) -> str:
    class_dict = {0: "angry", 1: "sad", 2: "neutral", 3: "positive"}

    if index > len(class_dict) or index < 0:
        raise AttributeError

    return class_dict[index]

dir_path = './model/'
model_name = 'podcasts_finetune_old_w_lr_1e-3_try1'
device = 'cpu'

config_path = os.path.join(dir_path, "train.config")
assert os.path.exists(config_path), f"No train.config in {dir_path}"

model_path = os.path.join(dir_path, model_name)
# check the model
if not os.path.exists(model_path):
    print(f"There is no saved model {model_path}. Nothing to inference")
#     return None

# load the model
cfg = lazycon.load(config_path)
model = cfg.model

model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model = model.double()
model.eval()


# sd.query_devices()
#
# fs=16000
# duration = 5 # seconds
# myrecording = sd.rec(duration * fs, samplerate=fs, channels=1, dtype='float64')
# print ("Recording Audio")
# sd.wait()
# print ("Audio recording complete , Play Audio")
# sd.play(myrecording, fs)
#
#
#
# feat = create_features(np.transpose(myrecording)[0])




def predict(file_path_wav) -> str:
    feat = create_features_for_audio(file_path_wav)

    inputs = torch.from_numpy(feat).to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probs = model(inputs)

    predictions = probs.cpu().numpy()
    pred_class = np.argmax(predictions, axis=1)

    return (index2name(pred_class[0]), predictions)


if __name__=="__main__":
    file = '/home/fedor/Desktop/projects/for_learning/data/train/angry/000a0d81f8d53ccf995d99632e1ffe5d.wav'
    print(predict(file))
    file = '/home/fedor/Desktop/projects/for_learning/data/train/angry/0a1bc50e8a26b3955fdef06430671090.wav'
    print(predict(file))
