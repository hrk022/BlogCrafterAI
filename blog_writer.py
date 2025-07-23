import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import streamlit as st
from dotenv import load_dotenv
import requests

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# ───────────────────────────────
# ✅ Load Environment Variables
# ───────────────────────────────
load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"
unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ───────────────────────────────
# ✅ Streamlit UI Setup
# ───────────────────────────────
st.set_page_config(page_title="BlogCrafter AI ✨", layout="centered")
st.title("📝 BlogCrafter AI ✨")

# ───────────────────────────────
# ✅ Streaming Token Callback Handler
# ───────────────────────────────
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

# ───────────────────────────────
# ✅ Session State Initialization
# ───────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ───────────────────────────────
# ✅ Web Scraper for Query Topic using LangChain Tavily Wrapper
# ───────────────────────────────
def web_scrape(query):
    try:
        search = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
        results = search.results(query=query, max_results=3)

        full_text = ""
        for result in results:
            content = result.get("content", "") or result.get("body", "")
            full_text += content + "\n\n"

        return full_text.strip()
    except Exception as e:
        st.error(f"🔌 Tavily API error: {str(e)}")
        return ""

# ───────────────────────────────
# ✅ Unsplash Multiple Images Fetch
# ───────────────────────────────
def fetch_images(topic, per_page=3):
    try:
        url = f"https://api.unsplash.com/search/photos?query={topic}&client_id={unsplash_access_key}&per_page={per_page}"
        response = requests.get(url)
        data = response.json()
        image_urls = [img["urls"]["regular"] for img in data.get("results", [])]
        return image_urls
    except Exception:
        return []

# ───────────────────────────────
# ✅ Vector Store Generator
# ───────────────────────────────
def generate_vectorstore(text):
    if not text or text.strip() == "":
        raise ValueError("❌ No text was extracted from the web to build vectorstore.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])

    if not docs:
        raise ValueError("❌ Document chunking failed. No chunks to process.")

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever()

# ───────────────────────────────
# ✅ Initialize Model & Chain
# ───────────────────────────────
def initialize_model(retriever):
    llm = ChatOpenAI(
        model_name="llama3-70b-8192",
        temperature=0.7,
        openai_api_key=api_key,
        streaming=True,
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, retriever=retriever)

# ───────────────────────────────
# ✅ Main Chat Input Block
# ───────────────────────────────
user_query = st.chat_input("🧠 Ask me to generate a blog on any topic...")

if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append(("user", user_query))

    # Scrape → Validate → Embed → Chain
    scraped_text = web_scrape(user_query)

    if not scraped_text:
        st.error("🚫 Failed to scrape any content from the web. Try a different topic.")
        st.stop()

    st.text_area("🔍 Scraped Content Preview:", scraped_text[:2000])

    try:
        retriever = generate_vectorstore(scraped_text)
    except Exception as e:
        st.error(f"⚠️ Vector generation failed: {str(e)}")
        st.stop()

    st.session_state.qa_chain = initialize_model(retriever)

    with st.chat_message("assistant"):
        response_handler = st.empty()
        stream_handler = StreamHandler(response_handler)

        result = st.session_state.qa_chain.invoke(
            {"question": "Write a beautiful, SEO-optimized blog on this topic."},
            config={"callbacks": [stream_handler]}
        )

        blog_content = result["answer"] if isinstance(result, dict) else result
        st.session_state.chat_history.append(("assistant", blog_content))

        # Show Multiple Unsplash Images
        image_urls = fetch_images(user_query, per_page=3)
        for url in image_urls:
            st.image(url, caption=f"📸 Related to: {user_query}", use_container_width=True)

# ───────────────────────────────
# ✅ Show Previous Chat History
# ───────────────────────────────
for sender, message in st.session_state.chat_history[:-1]:
    st.chat_message(sender).markdown(message)
