import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load environment variables
load_dotenv()

# 1. Cache the embedding model safely
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# 2. Store DB and API clients in session_state to prevent multiple locking instances
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = QdrantClient(path="qdrant_db")

if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Use the persistent state clients
qdrant_client = st.session_state.qdrant_client
groq_client = st.session_state.groq_client

COLLECTION_NAME = "pdf_knowledge_base"

st.title("📚 Production RAG Chatbot")
st.write("Ask questions from your private medical/PDF library.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if user_query := st.chat_input("What would you like to know from the documents?"):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # --- RETRIEVAL STEP ---
    with st.spinner("Searching knowledge base..."):
        query_vector = embedding_model.encode(user_query).tolist()
        
        # Updated for latest Qdrant API (.query_points)
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        search_results = response.points

    # Display retrieved sources in sidebar
    with st.sidebar:
        st.subheader("🔍 Retrieved Source Chunks")
        context_str = ""
        for idx, hit in enumerate(search_results):
            text = hit.payload["text"]
            source = hit.payload["source"]
            page = hit.payload["page"]
            score = hit.score
            
            context_str += f"\nContext from {source} (Page {page}):\n{text}\n---\n"
            
            st.markdown(f"**Chunk {idx+1}** (Match Score: {score:.2f})")
            st.caption(f"Source: {source} | Page: {page}")
            st.text(text[:150] + "...")
            st.write("---")

    # --- GENERATION STEP ---
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        system_prompt = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided text context. "
            "If the answer cannot be found in the context, say 'I cannot find that information in the documents.' "
            "At the end of your answer, you MUST clearly cite the Source filename and Page number provided in the context."
        )
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Context:\n{context_str}"},
                {"role": "user", "content": user_query}
            ],
            stream=True,
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
                
        response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})