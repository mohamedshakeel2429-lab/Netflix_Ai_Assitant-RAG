import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_ollama import ChatOllama
from vector_db import get_vector_store

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CSV_PATH = r"C:\Users\user\Downloads\archive\netflix_titles.csv"

# --- APP CONFIG ---
st.set_page_config(page_title="Netflix AI Insights", layout="wide", page_icon="🍿")

# --- CUSTOM NETFLIX CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #141414; color: white; }
    [data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #E50914; }
    h1, h2, h3 { color: #E50914 !important; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    [data-testid="stMetricValue"] { color: #E50914 !important; font-size: 2rem; }
    .stButton>button { 
        background-color: #E50914; color: white; border-radius: 4px; 
        border: none; font-weight: bold; width: 100%; transition: 0.3s;
        height: 3em;
    }
    .stButton>button:hover { background-color: #ff0a16; transform: scale(1.02); color: white; }
    .glass-card { 
        background: rgba(255, 255, 255, 0.05); padding: 20px; 
        border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
    }
    .chat-bubble { 
        padding: 15px; border-radius: 15px; margin: 10px 0; 
        border-left: 5px solid #E50914; background: #222; 
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. INITIALIZE BACKEND
# ==========================================
@st.cache_resource
def init_resources():
    try:
        # Load Vector Store
        v_store = get_vector_store(CSV_PATH) 
        
        # Initialize Gemma 3
        llm = ChatOllama(model="gemma3:latest", temperature=0.7)
        
        # Load Data
        df = pd.read_csv(CSV_PATH)
        df['country'] = df['country'].fillna("Unknown")
        
        # Combine title + description for vector search
        df['combined_text'] = df['title'].fillna('') + " " + df['description'].fillna('')
        
        return v_store, llm, df
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

vector_store, llm, df = init_resources()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.title("AI Navigator")
    page = st.radio("Go to", ["🏠 Home", "🤖 AI Chatbot (RAG)", "✨ Recommender", "📊 Trend Analyzer"])
    st.divider()
    st.info("Model: Gemma 3 (Local)\nEmbed: mxbai-large")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def is_aggregation_query(query):
    keywords = [
        "most", "count", "highest", "lowest",
        "growth", "compare", "trend",
        "how many", "most frequent",
        "which country produces"
    ]
    return any(word in query.lower() for word in keywords)

def handle_aggregation(query):
    query = query.lower()
    if "most frequent director" in query:
        result = df["director"].value_counts().head(1)
        return f"The most frequent director is {result.index[0]} with {result.values[0]} titles."
    elif "country produces the most comedies" in query:
        comedy_df = df[df["listed_in"].str.contains("Comedy", na=False)]
        result = comedy_df["country"].value_counts().head(1)
        return f"{result.index[0]} produces the most comedies with {result.values[0]} titles."
    else:
        return "This aggregation is not yet implemented."

def retrieve_context(prompt, k=5):
    docs = vector_store.similarity_search(prompt, k=k)
    if not docs:
        return None
    context = "\n".join([d.page_content for d in docs])
    return context

def build_strict_prompt(context, user_question):
    return f"""
You are a dataset-bound assistant.
You MUST follow these rules:
1. Answer ONLY using the information inside <CONTEXT>.
2. If the answer is not found in the context, respond exactly:
   "This information is not available in the dataset."
3. Do NOT use prior knowledge.
4. Do NOT guess.
5. Do NOT invent movie titles, directors, ratings, or years.
6. Be factual and concise.

<CONTEXT>
{context}
</CONTEXT>

User Question:
{user_question}

Answer:
"""

# ==========================================
# 4. PAGE LOGIC
# ==========================================

# --- PAGE 1: HOME ---
if page == "🏠 Home":
    st.title("Netflix Content Intelligence")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Titles", len(df))
    col2.metric("Movies", len(df[df['type'] == 'Movie']))
    col3.metric("TV Shows", len(df[df['type'] == 'TV Show']))
    col4.metric("Countries", df['country'].nunique())
    
    st.markdown("### 📋 Latest Catalog Preview")
    st.dataframe(df.head(20), use_container_width=True)

# --- PAGE 2: AI CHATBOT (RAG) ---
elif page == "🤖 AI Chatbot (RAG)":
    st.title("🔴 Netflix AI Assistant")
    st.markdown("_Query the catalog using natural language._")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the Netflix catalog..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching catalog..."):

                # --- Aggregation Queries ---
                if is_aggregation_query(prompt):
                    answer = handle_aggregation(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.stop()

                # --- Retrieval + RAG ---
                context = retrieve_context(prompt, k=5)
                if not context:
                    answer = "This information is not available in the dataset."
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.stop()

                strict_prompt = build_strict_prompt(context, prompt)
                response = llm.invoke(strict_prompt)
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})

# --- PAGE 3: SMART RECOMMENDER ---
elif page == "✨ Recommender":
    st.title("🎬 Smart Hybrid Recommender")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Define your Mood")
        mood = st.text_input("Mood/Vibe", placeholder="e.g. Heartwarming, gritty, mind-bending")
        genre = st.selectbox("Genre Filter", ["All"] + sorted(list(df['listed_in'].str.split(', ').explode().unique())))
        year_range = st.slider("Year Range", int(df.release_year.min()), int(df.release_year.max()), (2018, 2024))
        content_type = st.radio("Content Type", ["Movie", "TV Show"])
        search_btn = st.button("Generate AI Recommendations")

    with col2:
        if search_btn:
            with st.spinner("Analyzing similarities..."):
                # Semantic Search
                query = f"{mood} {genre} {content_type}"
                results = vector_store.similarity_search(query, k=30)
                
                # Metadata Filtering First
                final_recs = []
                for res in results:
                    m = res.metadata
                    if m['type'] == content_type and year_range[0] <= m['year'] <= year_range[1]:
                        if genre == "All" or genre.lower() in m['genre'].lower():
                            final_recs.append(res)
                
                if not final_recs:
                    st.warning("No perfect matches found. Try broadening your filters.")
                else:
                    for rec in final_recs[:5]:
                        with st.container():
                            st.markdown(f"""
                            <div class="glass-card">
                                <h3 style='margin:0;'>🍿 {rec.metadata['title']}</h3>
                                <p style='color: #E50914; font-weight: bold;'>{rec.metadata['year']} | {rec.metadata['genre']}</p>
                                <p style='font-size: 0.95em;'>{rec.page_content.split('Description: ')[-1]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Grounded AI Insight
                            exp_prompt = f"""
Use ONLY this description:

Title: {rec.metadata['title']}
Description: {rec.page_content}

Explain in ONE short sentence why this matches the mood '{mood}'.
Do NOT add new facts.
"""
                            explanation = llm.invoke(exp_prompt)
                            st.caption(f"✨ **AI Reasoning:** {explanation.content}")

# --- PAGE 4: TREND ANALYZER ---
elif page == "📊 Trend Analyzer":
    st.title("📈 Content Strategy Insights")
    
    tab1, tab2 = st.tabs(["📊 Visual Analytics", "🧠 AI Business Summary"])
    
    with tab1:
        c1, c2 = st.columns(2)
        # Production Trend
        growth = df.groupby('release_year').count()['show_id'].reset_index()
        fig1 = px.area(growth, x='release_year', y='show_id', title="Production Growth", color_discrete_sequence=['#E50914'])
        fig1.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)')
        c1.plotly_chart(fig1, use_container_width=True)
        
        # Distribution
        type_dist = df['type'].value_counts().reset_index()
        type_dist.columns = ['type', 'count']
        fig2 = px.pie(type_dist, values='count', names='type', title="Content Split", hole=0.5, color_discrete_sequence=['#E50914', '#333333'])
        fig2.update_layout(template="plotly_dark")
        c2.plotly_chart(fig2, use_container_width=True)

    with tab2:
        if st.button("Generate AI Business Analysis"):
            with st.spinner("Analyzing dataset patterns..."):
                top_genres = df['listed_in'].value_counts().head(5).to_string()
                stats_context = f"Total: {len(df)} titles. Top Categories: {top_genres}. Most recent year: {df.release_year.max()}."
                
                analysis_prompt = f"""
You are a data analyst.
Use ONLY the statistics below:

{stats_context}

Provide:
1. Three factual insights
2. One future recommendation

Do NOT invent numbers. Be strictly data-driven.
"""
                analysis = llm.invoke(analysis_prompt)
                st.markdown(f"<div class='chat-bubble'>{analysis.content}</div>", unsafe_allow_html=True)
