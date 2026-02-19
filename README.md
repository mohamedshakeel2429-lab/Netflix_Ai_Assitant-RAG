# Netflix_Ai_Assitant-RAG

This is a professional, high-impact README.md designed for a senior-level AI portfolio. It highlights the architecture, the RAG implementation, and the business value of the application.

🍿 Netflix AI Insights: Next-Gen Content Intelligence

![alt text](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)


![alt text](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit)


![alt text](https://img.shields.io/badge/Ollama-Gemma3-white?logo=ollama)


![alt text](https://img.shields.io/badge/VectorDB-ChromaDB-green)

An enterprise-grade AI analytics platform that transforms raw Netflix catalog data into actionable insights. This project utilizes Retrieval-Augmented Generation (RAG), local Large Language Models (LLMs), and semantic search to provide a premium user experience for content discovery and trend analysis.

🚀 Key Features
1. 🤖 Netflix AI Chatbot (RAG-Powered)

A sophisticated conversational agent that goes beyond simple keyword matching. By leveraging Gemma 3 and ChromaDB, the chatbot understands context and nuance.

Semantic Retrieval: Queries the 8k+ catalog using vector embeddings.

Contextual Awareness: Answers questions based strictly on the dataset to minimize hallucinations.

Natural Language Understanding: Handles complex requests like "Find me dark thrillers from the UK after 2018."

2. ✨ Smart Hybrid Recommender

A dual-layer recommendation engine combining metadata filtering with high-dimensional vector similarity.

Mood-Based Search: Uses mxbai-embed-large to match your "vibe" to content descriptions.

Hard Metadata Filters: Real-time Python-side filtering for Year, Genre, and Content Type.

AI Justification: Generates unique reasoning for every recommendation using the LLM.

3. 📊 Content Trend Analyzer

A high-level business intelligence dashboard for content strategists.

Visual Analytics: Interactive Plotly charts showing content growth and distribution.

AI Executive Summary: The LLM acts as a business analyst to interpret charts and suggest future content strategies.

🏗 Architecture Detail

As a senior-led project, the architecture focuses on Local-First AI, ensuring data privacy and zero API costs:

Embedding Engine: mxbai-embed-large (Local via Ollama)

Inference Engine: gemma3:latest (Local via Ollama)

Vector Store: ChromaDB (On-disk persistence)

Data Orchestration: LangChain

Frontend: Streamlit with a custom Glassmorphism Netflix-themed UI.

🛠 Installation & Setup
1. Prerequisites

Python 3.10 or higher.

Ollama installed and running.

2. Pull Local Models
code
Bash
download
content_copy
expand_less
ollama pull gemma3:latest
ollama pull mxbai-embed-large
3. Clone and Install Dependencies
code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/netflix-ai-insights.git
cd netflix-ai-insights
pip install -r requirements.txt
4. Data Setup

Place your netflix_titles.csv in the project root or update the CSV_PATH in main.py and vector_db.py.

5. Launch the Application
code
Bash
download
content_copy
expand_less
streamlit run main.py
📦 Dependencies
Package	Purpose
streamlit	Modern UI/UX Framework
langchain-ollama	Local LLM & Embedding Integration
langchain-chroma	High-performance Vector Store management
pandas	Data manipulation and CSV processing
plotly	Interactive business intelligence charts
chromadb	Local vector database storage
📂 Project Structure
code
Text
download
content_copy
expand_less
├── main.py              # Streamlit UI & Application Orchestration
├── vector_db.py         # Ingestion Logic & Vector Store Management
├── netflix_titles.csv   # Dataset (Source)
├── chroma_netflix_db/   # Local Vector Storage (Generated)
└── requirements.txt     # Project Dependencies
🛡 Architectural Decisions

Local LLM (Gemma 3): Chosen for its state-of-the-art performance on local hardware, ensuring no data leaves the user's machine.

RAG Architecture: We chose RAG over fine-tuning to allow the dataset to be updated instantly without expensive retraining.

Batch Ingestion: Data is processed in chunks of 500 to optimize CPU/GPU utilization during the initial vectorization.

📄 License

Distributed under the MIT License. See LICENSE for more information.

Developed with ❤️ for the AI community.
