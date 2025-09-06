import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
from PIL import Image
from transformers import pipeline # Keep for now, may remove if not used after Gemini integration
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv() # Load environment variables from .env

# Configure Google Gemini API
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("GOOGLE_API_KEY not found in .env. Please add it to use Gemini LLM. Refer to README.md for instructions.")
    st.stop() # Stop the app if API key is missing
else:
    genai.configure(api_key=google_api_key)

# Access an example environment variable
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.warning("OPENAI_API_KEY not found in .env. Please add it for full functionality.")

# You would typically use a library like moviepy for video, and pydub for audio. 
# For simplicity, we'll just handle file uploads and display placeholders.

# 1. Text Processing and Embedding Generation
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def get_text_embedding(text, embedding_model=None):
    if embedding_model is None:
        embedding_model = model
    return embedding_model.encode(text)

# 2. Retrieval Component (FAISS)
class RAGSystem:
    def __init__(self, documents, embedding_function=None):
        self.documents = documents
        self.embedding_function = embedding_function if embedding_function else get_text_embedding
        self.embeddings = [self.embedding_function(doc) for doc in documents]
        self.dimension = self.embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))

    def retrieve(self, query, k=1):
        query_embedding = self.embedding_function(query).astype('float32')
        D, I = self.index.search(np.array([query_embedding]), k)
        return [self.documents[i] for i in I[0]]

# Example Knowledge Base
knowledge_base = [
    "The capital of France is Paris.",
    "Python is a high-level, interpreted programming language.",
    "Streamlit is an open-source app framework for Machine Learning and Data Science.",
    "The Earth revolves around the Sun.",
    "Artificial intelligence is a broad field of computer science.",
    "Gemini is a family of multimodal large language models developed by Google AI."
]

rag_system = RAGSystem(knowledge_base)

# Placeholder for speech synthesis
def text_to_speech(text):
    # In a real application, you'd use a text-to-speech API or library (e.g., gTTS, Amazon Polly)
    st.write(f"Speech synthesis: \"{text}\" would be converted to audio here.")
    # Example: return audio_file_path

# Placeholder for image processing
def process_image(image_file):
    # In a real application, you'd send this to an image analysis API or a local model
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Image", use_container_width=True) # Updated for deprecation
    image_description = "Detailed image description from a vision model."
    ocr_text = "Extracted text from image using OCR."
    st.write(f"Image analysis: {image_description}")
    st.write(f"OCR Text: {ocr_text}")
    return f"Image analysis: {image_description}. OCR: {ocr_text}"

# Placeholder for voice processing
def process_voice(audio_file):
    # In a real application, you'd send this to a speech-to-text API
    st.audio(audio_file, format='audio/wav') # Assuming WAV for demonstration
    return "Voice analysis: Speech transcribed to text, tone and intent analyzed here."

# Placeholder for video processing
def process_video(video_file):
    # In a real application, you'd send this to a video analysis API
    st.video(video_file, use_container_width=True) # Updated for deprecation
    video_summary = "Summary of main actions in the video."
    detected_objects = "Objects detected in video: [object1, object2]."
    speech_transcript = "Transcribed speech from video audio."
    st.write(f"Video summary: {video_summary}")
    st.write(f"Detected objects: {detected_objects}")
    st.write(f"Speech transcript: {speech_transcript}")
    return f"Video analysis: {video_summary}. Objects: {detected_objects}. Speech: {speech_transcript}"

# 3. Response Generation Layer (Google Gemini)
@st.cache_resource
def load_gemini_model(model_name="gemini-pro"):
    return genai.GenerativeModel(model_name)

# Initialize with a default model
gemini_model = load_gemini_model()

def generate_response(query, retrieved_documents, selected_model="gemini-pro", temperature=0.7, top_p=0.9, top_k=1, llm_model=None):
    if llm_model is None:
        llm_model = load_gemini_model(selected_model)
    context = " ".join(retrieved_documents)
    prompt_parts = [
        f"You are a helpful assistant. Based on the following context, answer the query comprehensively.\n\nContext: {context}\n\nQuery: {query}\n\nAnswer:"
    ]

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": 2048, # A reasonable max output
    }

    # Default safety settings (can be customized)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    try:
        response = llm_model.generate_content(prompt_parts, generation_config=generation_config, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        st.error(f"Error generating response with Gemini: {e}")
        return "Could not generate a response with Gemini."


# Streamlit UI
st.set_page_config(layout="wide") # Use wide layout for better aesthetics
st.title("✨ Multimodal RAG System with Google Gemini ✨")

st.markdown("Welcome! This application demonstrates a Retrieval-Augmented Generation (RAG) system capable of processing text queries and handling multimodal inputs (images, voice, video) with Google Gemini.")

# Sidebar for LLM configuration
st.sidebar.header("⚙️ LLM Configuration")
selected_gemini_model = st.sidebar.selectbox(
    "Choose Gemini Model:",
    ["gemini-pro", "gemini-pro-vision"], # Add more models as needed/available
    index=0
)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=1, step=1)

st.sidebar.markdown("--- ✨ ---")


# Main content area
with st.container():
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input")
        user_query = st.text_input("Enter your text query:", key="text_query_input")
        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], key="image_uploader")
        uploaded_audio = st.file_uploader("Upload Voice/Audio", type=["wav", "mp3"], key="audio_uploader")
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "mov"], key="video_uploader")

    with col2:
        st.subheader("📊 Analysis & Output")
        if st.button("Process Inputs"):
            if not (user_query or uploaded_image or uploaded_audio or uploaded_video):
                st.warning("Please provide at least one input (text query, image, audio, or video).")
            else:
                with st.spinner("Processing inputs and generating response..."):
                    # Multimodal Input Analysis
                    combined_input_context = user_query if user_query else ""
                    st.markdown("### Multimodal Input Analysis:")

                    if uploaded_image:
                        st.info("Analyzing Image...")
                        image_analysis_result = process_image(uploaded_image)
                        combined_input_context += " " + image_analysis_result 
                    if uploaded_audio:
                        st.info("Analyzing Audio...")
                        audio_analysis_result = process_voice(uploaded_audio)
                        combined_input_context += " " + audio_analysis_result
                    if uploaded_video:
                        st.info("Analyzing Video...")
                        video_analysis_result = process_video(uploaded_video)
                        combined_input_context += " " + video_analysis_result

                    st.markdown("### Combined Input Context (for RAG):")
                    st.code(combined_input_context)

                    # Retrieval Component (still text-based for now)
                    st.markdown("### Retrieved Documents (Text-based):")
                    retrieved_docs = []
                    if user_query:
                        retrieved_docs = rag_system.retrieve(user_query, k=2) # Retrieve top 2 for better context
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"{i+1}. {doc}")
                    else:
                        st.info("No text query provided for document retrieval. Proceeding with general LLM response if multimodal inputs yield context.")

                    # Response Generation
                    st.markdown("### Generated Response (powered by Google Gemini):")
                    response_text = ""
                    if retrieved_docs or combined_input_context.strip():
                        response_text = generate_response(
                            user_query if user_query else "", # Pass user_query even if empty if combined_input_context is present
                            retrieved_docs,
                            selected_model=selected_gemini_model,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k
                        )
                        st.write(response_text)

                        st.markdown("### Multimodal Outputs:")
                        st.write("**Speech Output:**")
                        text_to_speech(response_text)
                    else:
                        st.warning("No sufficient context or query to generate a response.")
