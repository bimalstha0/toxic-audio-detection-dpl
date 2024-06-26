
import librosa
from pydub import AudioSegment
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
import soundfile as sf


model = load_model('toxic_audio_model.hdf5')

def convert_to_wav(audio):
    if audio.type == 'audio/wav' or audio.type == 'audio/x-wav':
        return audio
    else:
        audio_data = BytesIO(audio.read())
        audio_segment = AudioSegment.from_file(audio_data, format=str(audio.type)[8:]) if str(audio.type).startswith('audio/x-') else AudioSegment.from_file(audio_data, format=str(audio.type)[6:]) 
        wav_data = BytesIO()
        audio_segment.export(wav_data, format="wav")
        wav_data.seek(0)
        return wav_data

def make_prediction(audio, is_bytes = False):
    if not is_bytes:
        audio = convert_to_wav(audio)
    mfccs = get_mfccs_from_audio(audio, is_bytes = is_bytes)
    mfccs = np.array(mfccs)
    if np.all(np.isnan(mfccs)):
        return np.array([])
    mfccs = mfccs.reshape(mfccs.shape[0], mfccs.shape[1], mfccs.shape[2], 1)
    prediction = (model.predict(mfccs) > 0.5).astype(int)
    return prediction

def vol_normalization(audio, max_rms):
    rms = np.sqrt(np.mean(np.square(audio)))
    scaling_factor = max_rms / rms
    normalized_audio = audio * scaling_factor
    return normalized_audio


def get_mfccs_from_audio(audio,is_bytes = False):
    max_rms = 0.35941255
    y = None
    sr = 22050
    if is_bytes:
        y, s = sf.read(BytesIO(audio))
        y = librosa.resample(y=y.T,orig_sr=s,target_sr = sr)
        y = librosa.to_mono(y)
    else:
        y, sr = librosa.load(audio)
    y = vol_normalization(y, max_rms)
    mfccs = []
    start_sample = 0
    end_sample = len(y)
    segment_samples = int(5 * sr)
    
    while start_sample < end_sample:
        if start_sample + segment_samples > end_sample:
            break
        y_slice = y[start_sample:start_sample + segment_samples]
        ms = librosa.feature.melspectrogram(y=y_slice, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        padded_mfccs = np.pad(log_ms, ((0, 0), (0, max(0, 216 - log_ms.shape[1]))), mode='constant')
        mfccs.append(padded_mfccs)
        start_sample += segment_samples
    
    return mfccs
