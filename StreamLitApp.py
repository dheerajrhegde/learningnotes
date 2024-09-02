import streamlit as st
import CentralGraph as cg
from langchain.document_loaders import YoutubeLoader

st.set_page_config(
    page_title="Middle School Learning",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Middle School Learning")
st.write("This app is designed to help middle school students learn by providing them \
with a summary of the video they are watching, as well as questions and answers to test their understanding.")

with st.form(key="video", border=False):
    url = st.text_input("Enter youtube video url", "https://www.youtube.com/watch?v=ewokFOSxabs")
    submitted = st.form_submit_button("Submit")

if submitted:
    loader = YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False,
        language=["en"],
        translation="en",
    )
    #transcription = loader.load().page_content
    transcription = [doc.page_content for doc in loader.load()]
    transcription = "\n\n".join(transcription)

    print("transcription = ", transcription)
    response = cg.app.invoke({"current_segment": transcription})
    st.markdown(response["writer_output"])
    st.markdown(response["questions_answers"])
    st.markdown(response["test_questions_answers"])