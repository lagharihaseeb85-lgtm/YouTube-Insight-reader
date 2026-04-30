import streamlit as st
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fpdf import FPDF
import requests
from io import BytesIO
from PIL import Image
import isodate

# Page config
st.set_page_config(
    page_title="YouTube Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
  .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF0000 0%, #282828 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
  .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF0000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
  .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
  .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize APIs - HARDCODED KEYS FOR LOCAL TESTING ONLY
@st.cache_resource
def init_apis():
    yt_key = "AIzaSyBf3XQ1_-MBEynmsglRMtnqQeFXePw-pQU"
    gemini_key = "AIzaSyAlxQOAJ8TGp2i8PKz4azEMOqK5Eeb7pto"

    if not yt_key:
        st.error("YOUTUBE_API_KEY not found")
        st.stop()

    youtube = build('youtube', 'v3', developerKey=yt_key)

    if gemini_key:
        genai.configure(api_key=gemini_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    else:
        gemini_model = None

    return youtube, gemini_model

youtube, gemini_model = init_apis()
analyzer = SentimentIntensityAnalyzer()

# Helper Functions
def extract_video_id(url):
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def format_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

def format_duration(duration):
    try:
        dur = isodate.parse_duration(duration)
        total_seconds = int(dur.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"
    except:
        return "N/A"

@st.cache_data(ttl=3600)
def get_video_data(video_id):
    try:
        video_response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=video_id
        ).execute()

        if not video_response['items']:
            return None

        video = video_response['items'][0]
        snippet = video['snippet']
        stats = video['statistics']
        content = video['contentDetails']

        channel_response = youtube.channels().list(
            part='statistics,snippet',
            id=snippet['channelId']
        ).execute()
        channel_stats = channel_response['items'][0]['statistics']

        return {
            'video_id': video_id,
            'title': snippet['title'],
            'channel': snippet['channelTitle'],
            'channel_id': snippet['channelId'],
            'channel_thumb': channel_response['items'][0]['snippet']['thumbnails']['default']['url'],
            'description': snippet['description'],
            'published': snippet['publishedAt'],
            'thumbnail': snippet['thumbnails']['high']['url'],
            'duration': format_duration(content['duration']),
            'duration_raw': content['duration'],
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'comments': int(stats.get('commentCount', 0)),
            'subscribers': int(channel_stats.get('subscriberCount', 0)),
            'tags': snippet.get('tags', [])
        }
    except HttpError as e:
        st.error(f"YouTube API Error: {e}")
        return None

@st.cache_data(ttl=1800)
def get_comments(video_id, max_results=500):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            textFormat='plainText',
            order='relevance'
        )

        while request and len(comments) < max_results:
            response = request.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': comment['textDisplay'],
                    'likes': comment['likeCount'],
                    'author': comment['authorDisplayName'],
                    'published': comment['publishedAt']
                })
            request = youtube.commentThreads().list_next(request, response)

        return pd.DataFrame(comments)
    except:
        return pd.DataFrame()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    return 'Neutral'

def calculate_engagement_metrics(data):
    views = data['views'] if data['views'] > 0 else 1
    engagement_rate = ((data['likes'] + data['comments']) / views) * 100
    like_ratio = (data['likes'] / views) * 100
    comment_ratio = (data['comments'] / views) * 100
    virality = (like_ratio * 0.5 + comment_ratio * 2 + engagement_rate * 0.3)

    return {
        'engagement_rate': round(engagement_rate, 2),
        'like_ratio': round(like_ratio, 3),
        'comment_ratio': round(comment_ratio, 3),
        'virality_score': round(min(virality, 100), 1)
    }

@st.cache_data
def get_gemini_insights(title, description, comments_sample, channel):
    if not gemini_model:
        return None

    prompt = f"""
    Analyze this YouTube video and provide insights in JSON format:

    Video Title: {title}
    Channel: {channel}
    Description: {description[:1000]}
    Sample Comments: {comments_sample[:2000]}

    Return ONLY valid JSON with these keys:
    {{
        "summary": "2-3 sentence video summary",
        "key_points": ["point1", "point2", "point3", "point4", "point5"],
        "topics_covered": ["topic1", "topic2", "topic3"],
        "seo_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "audience_intent": "What viewers want from this video",
        "content_category": "Education/Entertainment/Tech/etc",
        "title_optimization": "Better title suggestion",
        "thumbnail_ideas": ["idea1", "idea2", "idea3"],
        "content_tips": ["tip1", "tip2", "tip3"],
        "seo_strategy": "1 paragraph SEO advice"
    }}
    """

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        import json
        return json.loads(text[json_start:json_end])
    except Exception as e:
        st.warning(f"Gemini AI failed: {e}")
        return None

def generate_pdf_report(data, metrics, insights, sentiment_data, wordcloud_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'YouTube Insight Pro - Analysis Report', 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, f"Video: {data['title'][:60]}", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Channel: {data['channel']} | {format_number(data['subscribers'])} subscribers", 0, 1)
    pdf.cell(0, 6, f"Published: {data['published'][:10]} | Duration: {data['duration']}", 0, 1)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Performance Metrics', 0, 1)
    pdf.set_font('Arial', '', 10)
    col1 = f"Views: {format_number(data['views'])}"
    col2 = f"Likes: {format_number(data['likes'])}"
    col3 = f"Comments: {format_number(data['comments'])}"
    pdf.cell(63, 6, col1, 0, 0)
    pdf.cell(63, 6, col2, 0, 0)
    pdf.cell(63, 6, col3, 0, 1)

    col1 = f"Engagement: {metrics['engagement_rate']}%"
    col2 = f"Like Ratio: {metrics['like_ratio']}%"
    col3 = f"Virality: {metrics['virality_score']}/100"
    pdf.cell(63, 6, col1, 0, 0)
    pdf.cell(63, 6, col2, 0, 0)
    pdf.cell(63, 6, col3, 0, 1)
    pdf.ln(5)

    if insights:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'AI Insights', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, f"Summary: {insights['summary']}")
        pdf.ln(2)
        pdf.cell(0, 6, f"Category: {insights['content_category']} | Intent: {insights['audience_intent']}", 0, 1)
        pdf.ln(3)

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Key Points:', 0, 1)
        pdf.set_font('Arial', '', 9)
        for point in insights['key_points'][:5]:
            pdf.cell(0, 5, f"- {point}", 0, 1)
        pdf.ln(3)

        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'SEO Keywords:', 0, 1)
        pdf.set_font('Arial', '', 9)
        pdf.cell(0, 5, ', '.join(insights['seo_keywords']), 0, 1)

    if wordcloud_img:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Comment Word Cloud', 0, 1)
        pdf.image(wordcloud_img, x=10, w=190)

    return pdf.output(dest='S').encode('latin-1')

# Sidebar
with st.sidebar:
    st.markdown("### 📊 YouTube Insight Pro")
    st.markdown("Paste any YouTube URL to get AI-powered analytics")

    url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Comments to Analyze", 100, 500, 300, 100)
    analyze_btn = st.button("🚀 Analyze Video", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**API Status**")
    st.success("YouTube API: Connected")
    if gemini_model:
        st.success("Gemini AI: Connected")
    else:
        st.warning("Gemini AI: Not configured")

# Main App
st.markdown('<h1 class="main-header">YouTube Insight Pro</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>AI-Powered Video & Comment Analytics Dashboard</p>", unsafe_allow_html=True)

if analyze_btn and url:
    video_id = extract_video_id(url)

    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
        st.stop()

    with st.spinner('Fetching video data...'):
        data = get_video_data(video_id)

    if not data:
        st.error("Could not fetch video. It may be private, deleted, or region-locked.")
        st.stop()

    metrics = calculate_engagement_metrics(data)

    with st.spinner(f'Analyzing {max_comments} comments...'):
        df_comments = get_comments(video_id, max_comments)

    if not df_comments.empty:
        df_comments['sentiment'] = df_comments['text'].apply(analyze_sentiment)
        sentiment_counts = df_comments['sentiment'].value_counts()
    else:
        sentiment_counts = pd.Series()

    with st.spinner('Generating AI insights with Gemini...'):
        comments_sample = ' '.join(df_comments['text'].head(50).tolist()) if not df_comments.empty else ""
        insights = get_gemini_insights(data['title'], data['description'], comments_sample, data['channel'])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💬 Comments", "🧠 AI Insights", "📄 Report"])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(data['thumbnail'], use_container_width=True)
            st.markdown(f"**{data['title']}**")
            st.markdown(f"by **{data['channel']}**")

        with col2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Views", format_number(data['views']))
            m2.metric("Likes", format_number(data['likes']))
            m3.metric("Comments", format_number(data['comments']))
            m4.metric("Subscribers", format_number(data['subscribers']))

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Engagement", f"{metrics['engagement_rate']}%")
            m6.metric("Like Ratio", f"{metrics['like_ratio']}%")
            m7.metric("Virality Score", f"{metrics['virality_score']}/100")
            m8.metric("Duration", data['duration'])

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=metrics['virality_score'],
                title={'text': "Virality Score"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#FF0000"},
                       'steps': [
                           {'range': [0, 40], 'color': "#ffcccc"},
                           {'range': [40, 70], 'color': "#ff8080"},
                           {'range': [70, 100], 'color': "#ff4d4d"}]}
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not sentiment_counts.empty:
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                           title="Comment Sentiment Distribution",
                           color_discrete_map={'Positive':'#28a745', 'Neutral':'#ffc107', 'Negative':'#dc3545'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if df_comments.empty:
            st.warning("No comments found or comments are disabled.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Analyzed", len(df_comments))
            col2.metric("Positive", f"{(sentiment_counts.get('Positive', 0)/len(df_comments)*100):.1f}%")
            col3.metric("Negative", f"{(sentiment_counts.get('Negative', 0)/len(df_comments)*100):.1f}%")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("👍 Top Positive Comments")
                top_pos = df_comments[df_comments['sentiment']=='Positive'].nlargest(5, 'likes')
                for _, row in top_pos.iterrows():
                    st.markdown(f"**{row['author']}** ({row['likes']} likes)")
                    st.caption(row['text'][:150] + "...")

            with col2:
                st.subheader("👎 Top Negative Comments")
                top_neg = df_comments[df_comments['sentiment']=='Negative'].nlargest(5, 'likes')
                for _, row in top_neg.iterrows():
                    st.markdown(f"**{row['author']}** ({row['likes']} likes)")
                    st.caption(row['text'][:150] + "...")

            st.subheader("☁️ Word Cloud")
            text = ' '.join(df_comments['text'].tolist())
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            wordcloud_img = buf

    with tab3:
        if not insights:
            st.warning("Gemini API not configured")
        else:
            st.subheader("📝 AI Summary")
            st.info(insights['summary'])

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🎯 Key Points")
                for point in insights['key_points']:
                    st.markdown(f"- {point}")

                st.subheader("🏷️ Topics Covered")
                st.write(", ".join(insights['topics_covered']))

            with col2:
                st.subheader("🔍 SEO Keywords")
                st.write(", ".join(insights['seo_keywords']))

                st.subheader("👥 Audience Intent")
                st.write(insights['audience_intent'])

            st.markdown("---")
            st.subheader("🚀 Growth Suggestions")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Title Optimization**")
                st.success(insights['title_optimization'])

                st.markdown("**Thumbnail Ideas**")
                for idea in insights['thumbnail_ideas']:
                    st.markdown(f"- {idea}")

            with col2:
                st.markdown("**Content Tips**")
                for tip in insights['content_tips']:
                    st.markdown(f"- {tip}")

                st.markdown("**SEO Strategy**")
                st.write(insights['seo_strategy'])

    with tab4:
        st.subheader("📄 Download Professional PDF Report")
        st.write("Generate a complete analysis report with all metrics, charts, and AI insights.")

        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Creating PDF..."):
                wordcloud_img = buf if 'buf' in locals() else None
                pdf_bytes = generate_pdf_report(data, metrics, insights, sentiment_counts, wordcloud_img)

                st.download_button(
                    label="⬇️ Download Report",
                    data=pdf_bytes,
                    file_name=f"{data['channel']}_{video_id}_report.pdf",
                    mime="application/pdf"
                )
                st.success("PDF generated successfully!")

else:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://www.gstatic.com/youtube/img/branding/youtubelogo/svg/youtubelogo.svg", width=200)
        st.markdown("### Paste a YouTube URL in the sidebar to begin")
        st.markdown("""
        **Features:**
        - 📊 Complete video metadata & engagement metrics
        - 💬 AI sentiment analysis of 500 comments
        - 🧠 Gemini-powered content insights & SEO tips
        - 📈 Virality score & growth suggestions
        - 📄 1-click professional PDF reports
        """)

st.markdown("---")
st.caption("YouTube Insight Pro v1.0 | Built with Streamlit + Gemini AI | Data from YouTube Data API v3")
