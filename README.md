# ✨ BlogCrafterAI — Your Intelligent Blog Writing Assistant 📝🤖

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-orange?logo=streamlit&style=for-the-badge)](https://blogcrafterai-uaxvtnbqqy4d943xexkfrp.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&style=for-the-badge)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](#license)

---

🚀 **BlogCrafterAI** helps you generate high-quality, SEO-friendly blogs using state-of-the-art language models and a sleek, Streamlit-based interface.

## 🌟 Features

- 🧠 AI-powered blog generation using OpenAI
- 📄 Upload PDFs to summarize and blog from
- 🔎 Semantic search with FAISS vector database
- 🛠️ Dynamic prompt tuning for custom creativity
- 💬 Chat-style assistant to guide your content
- 🎨 Modern and responsive UI built on Streamlit

---

## 📸 Demo

> 🔗 [Live Demo](https://blogcrafterai-uaxvtnbqqy4d943xexkfrp.streamlit.app/)  
> Generate blogs from ideas, topics, or documents in seconds!

---

## 🧩 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Core logic & scripting |
| 🧠 LangChain | Language model orchestration |
| 🏗️ FAISS | Vector storage for semantic retrieval |
| 📜 OpenAI | LLM-based generation |
| 🧬 Tavily | Web search integration |
| 🌐 Streamlit | Frontend and deployment |
| 🔐 dotenv | API key management |

---

## ⚙️ Setup

```bash
git clone https://github.com/your-username/BlogCrafterAI.git
cd BlogCrafterAI

# Install dependencies
pip install -r requirements.txt

# Set your environment variables
touch .env
# Add your keys like:
# OPENAI_API_KEY=your_openai_key
# TAVILY_API_KEY=your_tavily_key

# Run the app
streamlit run blog_writer.py


