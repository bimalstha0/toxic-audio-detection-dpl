import streamlit as st
import pipeline 
# import sounddevice as sd
import datetime

st.set_page_config(
    page_title='Toxicity Detection',
    page_icon='🔊'
)



st.title('Toxic Audio Detection')

audio_uploaded = st.file_uploader('Upload an audio file', type=['wav', 'mp3', 'm4a'])
predict_button = st.button('Predict')
if audio_uploaded:
    st.audio(audio_uploaded)
    if predict_button:
        predictions = pipeline.make_prediction(audio_uploaded)
        if not any(predictions):
            st.write('<span style="color:green">No Toxicity Detected</span>', unsafe_allow_html=True)
        else:
            st.write('<span style="color:red">!!! TOXICITY DETECTED !!!</span>', unsafe_allow_html=True)
    
        
        start = -1
        for i, prediction in enumerate(predictions):
            if prediction == 1:
                if start == -1:
                    start = i * 5
            elif start != -1:
                st.write(f'!!! TOXIC !!! {str(datetime.timedelta(seconds = start))}-{str(datetime.timedelta(seconds = i*5))}')
                st.audio(audio_uploaded, start_time=start, end_time=i*5)
                start = -1
        
        if start != -1:
            st.audio(audio_uploaded, start_time=start, end_time=len(predictions)*5)


st.write('OR')
st.write('Live Audio')



# def record_audio_chunk(chunk_duration=5, fs=22050):
#     chunk_samples = int(chunk_duration * fs)
#     recording = sd.rec(chunk_samples, samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     return recording




fs = 22050  # Sample rate
chunk_duration = 5 
recording = False

# st.title("Audio Recorder")

# if st.button("Start Recording"):
#     recording = True
#     while recording:
#         rec = record_audio_chunk()
#         st.audio(rec)
# if st.button("Stop Recording"):
#     recording = False


# wav_audio_data = st_audiorec()

# if wav_audio_data is not None:
#     st.audio(wav_audio_data, format='audio/wav')
#     if st.button('Examine Recording'):
#         predictions = pipeline.make_prediction(wav_audio_data, bytes = True)
#         st.write(predictions)
# from audio_recorder_streamlit import audio_recorder

# audio_bytes = audio_recorder()
# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")
#     if st.button('Examine Recording'):
#         predictions = pipeline.make_prediction(audio_bytes, bytes = True)
#         st.write(predictions)
        
st.markdown("---")
st.write("Made with ❤️ by GroupB1")

st.markdown(
    """
    <style>
        body {
            color: #333;
            background-color: #f8f9fa;
        }
        .st-ba {
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)
