import streamlit as st
import pipeline 
from st_audiorec import st_audiorec
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
        st.write(predictions)
        if predictions == -1:
            st.write('<span style="color:red">!!! AUDIO NEEDS TO BE LONGER THAN 5 SEC !!!</span>', unsafe_allow_html=True)  
        elif not any(predictions):
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
                st.write(f'!!! TOXIC !!! {str(datetime.timedelta(seconds = start))}-{str(datetime.timedelta(seconds = i*5))}')
                st.audio(audio_uploaded, start_time=start, end_time=len(predictions)*5)


st.write('OR')
st.write('Live Audio')


fs = 22050  # Sample rate
chunk_duration = 5 
recording = False

st.title("Audio Recorder")

wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    if st.button('Examine Recording'):
        # st.write(wav_audio_data)
        predictions = pipeline.make_prediction(wav_audio_data,is_bytes=True)
        st.write([predictions])
        if predictions == -1:
            st.write('<span style="color:red">!!! AUDIO NEEDS TO BE LONGER THAN 5 SEC !!!</span>', unsafe_allow_html=True)  
        elif not any(predictions):
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
                    st.audio(wav_audio_data, start_time=start, end_time=i*5)
                    start = -1
            
            if start != -1:
                st.write(f'!!! TOXIC !!! {str(datetime.timedelta(seconds = start))}-{str(datetime.timedelta(seconds = (i+1)*5))}')
                st.audio(wav_audio_data, start_time=start, end_time=len(predictions)*5)


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
