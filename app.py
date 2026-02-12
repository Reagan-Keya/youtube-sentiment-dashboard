 # ====== INSTALL LIBRARIES (COLAB) ======
 # ====== IMPORTS ======
import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
from transformers import pipeline

# ====== DOWNLOAD VADER ======
nltk.download('vader_lexicon')

# ====== API KEY ======
API_KEY = st.secrets["YOUTUBE_API_KEY"]
youtube = build('youtube', 'v3', developerKey=API_KEY)

# ====== SENTIMENT MODELS ======
sia = SentimentIntensityAnalyzer()
hf_sentiment = pipeline("sentiment-analysis")

# ====== HELPER FUNCTIONS ======
def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"shorts/([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)"
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return url

def get_video_title(video_id):
    try:
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        return response["items"][0]["snippet"]["title"]
    except:
        return "Unknown Title"

def get_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    while request:
        response = request.execute()
        for item in response.get("items", []):
            comments.append(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            )
        request = youtube.commentThreads().list_next(request, response)
    return comments

# ====== HYBRID SENTIMENT ======
def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']

    if score >= 0.4:
        return "Positive"
    elif score <= -0.4:
        return "Negative"

    if -0.15 < score < 0.15:
        return "Neutral"

    try:
        result = hf_sentiment(text[:512])[0]
        label = result['label']
        confidence = result['score']

        if confidence > 0.75:
            return "Positive" if label == "POSITIVE" else "Negative"
        else:
            return "Neutral"
    except:
        return "Neutral"

# ====== STREAMLIT UI ======
st.set_page_config(page_title="YouTube Sentiment Dashboard", layout="wide")
st.title("📊 YouTube Sentiment Analysis Dashboard")

url = st.text_input("Enter YouTube Video URL:")

if "page" not in st.session_state:
    st.session_state.page = 0

if url:
    video_id = extract_video_id(url)
    video_title = get_video_title(video_id)

    # 🎯 NEW VIDEO INFO SECTION
    st.markdown("### 🎬 Video Information")
    st.info(f"""
**Title:** {video_title}  
**Video ID:** {video_id}  
**URL:** {url}
""")

    with st.spinner("Fetching comments..."):
        try:
            comments = get_comments(video_id)
        except:
            st.error("Error fetching comments. Check URL or API key.")
            comments = []

    if len(comments) == 0:
        st.warning("⚠ Comments disabled or no comments.")
    else:
        df = pd.DataFrame(comments, columns=["Comment"])
        df["Sentiment"] = df["Comment"].apply(get_sentiment)

        counts = df["Sentiment"].value_counts()
        summary_df = pd.DataFrame({
            "Type": ["Positive", "Negative", "Neutral"],
            "Count": [
                counts.get("Positive", 0),
                counts.get("Negative", 0),
                counts.get("Neutral", 0)
            ]
        })

        # ====== SENTIMENT OVERVIEW ======
        st.subheader("📊 Sentiment Overview")

        pie_fig = px.pie(summary_df, names="Type", values="Count")
        pie_fig.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(
            pie_fig,
            use_container_width=True,
            config={"staticPlot": True, "displayModeBar": False, "scrollZoom": False}
        )

        bar_fig = px.bar(summary_df, x="Type", y="Count")
        bar_fig.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(
            bar_fig,
            use_container_width=True,
            config={"staticPlot": True, "displayModeBar": False, "scrollZoom": False}
        )

        # ====== COMMENTS PAGINATION ======
        st.subheader("💬 Comments Viewer (50 per page)")
        page_size = 50
        total_pages = (len(df) - 1) // page_size + 1
        start = st.session_state.page * page_size
        end = start + page_size

        for _, row in df.iloc[start:end].iterrows():
            st.markdown(f"**{row['Sentiment']}**: {row['Comment']}")

        col1, col2 = st.columns(2)

        if col1.button("⬅ Previous") and st.session_state.page > 0:
            st.session_state.page -= 1

        if col2.button("Next ➡") and st.session_state.page < total_pages - 1:
            st.session_state.page += 1

        st.write(f"Page {st.session_state.page + 1} of {total_pages}")

        # ====== DOWNLOADS ======
        st.subheader("⬇ Download Data")
        st.download_button("Download Comments CSV", df.to_csv(index=False), "youtube_comments.csv")
        st.download_button("Download Summary CSV", summary_df.to_csv(index=False), "sentiment_summary.csv")
