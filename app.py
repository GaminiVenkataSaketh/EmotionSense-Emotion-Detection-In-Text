import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from docx import Document
from track_utils import (
    create_page_visited_table, add_page_visited_details, view_all_page_visited_details,
    add_prediction_details, view_all_prediction_details, create_emotionclf_table, IST
)

# Load model
pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Helper functions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def read_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# App
def main():
    st.set_page_config(page_title="EmotionSense | NLP App", layout="wide")
    st.title("🧠 EmotionSense: Understand Emotions in Text")

    menu = ["🏠 Dashboard", "📊 Analytics", "ℹ️ About"]
    choice = st.sidebar.selectbox("Navigate", menu)

    create_page_visited_table()
    create_emotionclf_table()

    if choice == "🏠 Dashboard":
        add_page_visited_details("Home", datetime.now(IST))
        st.header("🔍 Real-Time Emotion Detection")

        with st.form(key='emotion_clf_form'):
            st.subheader("📝 Input Text or Upload File")
            raw_text = st.text_area("Enter your text here:")
            uploaded_file = st.file_uploader("Or upload a .txt or .docx file", type=['txt', 'docx'])
            submit_text = st.form_submit_button(label='Analyze Emotion')

        if submit_text:
            if uploaded_file is not None:
                if uploaded_file.type == "text/plain":
                    raw_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    raw_text = read_docx(uploaded_file)

            if raw_text.strip() == "":
                st.warning("Please enter text or upload a valid file to analyze.")
            else:
                col1, col2 = st.columns(2)

                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                add_prediction_details(raw_text, prediction, np.max(probability), datetime.now(IST))

                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]
                proba_df_clean.sort_values(by="Probability", ascending=False, inplace=True)

                with col1:
                    st.subheader("📝 Processed Text")
                    st.info(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)

                    st.subheader("🎯 Top Prediction")
                    emoji_icon = emotions_emoji_dict.get(prediction, "")
                    st.success(f"{prediction.capitalize()} {emoji_icon}")
                    st.write(f"**Confidence:** {np.max(probability):.2f}")

                    st.subheader("📋 All Detected Emotions")
                    for _, row in proba_df_clean.iterrows():
                        emoji = emotions_emoji_dict.get(row['Emotion'], '')
                        st.write(f"- {row['Emotion'].capitalize()} {emoji} → {row['Probability']:.2f}")

                with col2:
                    st.subheader("📊 Probability Distribution")
                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='Emotion',
                        y='Probability',
                        color='Emotion'
                    )
                    st.altair_chart(fig, use_container_width=True)

    elif choice == "📊 Analytics":
        add_page_visited_details("Monitor", datetime.now(IST))
        st.header("📈 Application Metrics & Usage")

        with st.expander("🗂️ Page Visit Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('📊 Emotion Prediction Records'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)

    else:
        add_page_visited_details("About", datetime.now(IST))
        st.header("ℹ️ About EmotionSense")

        st.subheader("🌟 What is EmotionSense?")
        st.write("""
        **EmotionSense** is a powerful NLP-powered tool designed to detect and interpret human emotions from text.
        Whether analyzing tweets, reviews, feedback, or messages—EmotionSense uncovers the feelings behind the words.
        """)

        st.subheader("🚀 Why Use It?")
        st.markdown("""
        - Gain **instant emotional insights** from any written content.
        - Understand your **audience’s mood**, intentions, and concerns.
        - Use data-driven **emotion metrics** for better decision-making.
        """)

        st.subheader("⚙️ How Does It Work?")
        st.write("""
        This app uses machine learning and natural language processing to:
        - Preprocess and analyze input text
        - Extract emotional features
        - Classify text into various emotions like joy, anger, sadness, and more
        """)

        st.subheader("🔍 Real-World Applications")
        st.markdown("""
        - 🗣️ Social Media Sentiment Analysis  
        - 📞 Customer Support Feedback  
        - 🎯 Targeted Marketing Campaigns  
        - 📰 Media & Content Analysis  
        - 📚 Mental Health & Research  
        """)

        st.subheader("🤝 Get Involved")
        st.write("We are continuously improving EmotionSense. Your feedback and ideas are welcome!")

if __name__ == '__main__':
    main()
