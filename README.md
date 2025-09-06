# Multimodal RAG System with Streamlit and Google Gemini

## Overview

This project implements a basic Multimodal Retrieval-Augmented Generation (RAG) system using Streamlit for an interactive web interface. It demonstrates how to integrate text processing, information retrieval (using FAISS), and large language model (LLM) response generation (using Google Gemini) with placeholder support for image, voice, and video inputs.

## Features

*   **Text-based RAG**: Utilizes `sentence-transformers` for embedding generation and `FAISS` for efficient similarity search to retrieve relevant text documents from a knowledge base.
*   **Google Gemini Integration**: Employs the `gemini-2.0-flash` (and other flash models) for generating context-aware responses based on user queries and retrieved information.
*   **Multimodal Input Placeholders**: Includes Streamlit widgets for uploading images, audio, and video files, along with placeholder functions to simulate their analysis (object detection, OCR, speech-to-text, video summarization).
*   **Environment Variable Management**: Uses `python-dotenv` for securely managing API keys and other configuration.
*   **Interactive UI**: Built with Streamlit for a user-friendly web application.

## Architecture

The system follows a layered architecture:

1.  **Multimodal Input Processing Layer**: Handles text queries directly and provides placeholders for processing images, voice, and video. In a full implementation, this layer would extract meaningful features or text from multimodal inputs.
2.  **Retrieval Component**: Uses `SentenceTransformer` to create vector embeddings for text documents and `FAISS` for quick similarity search to find the most relevant information.
3.  **Response Generation Layer**: Leverages Google Gemini (`gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-2.5-flash`) to synthesize coherent and contextually relevant answers by combining the user's query with the retrieved documents.
4.  **Streamlit UI Layer**: Provides the interactive interface for users to input queries and upload files, and to display the system's responses and insights.

```mermaid
graph TD;
    A[User Input] --> B{Streamlit UI};
    B --> C{Text Query};
    B --> D{Image Upload};
    B --> E{Audio Upload};
    B --> F{Video Upload};

    C --> G[Text Processor (SentenceTransformer)];
    D --> H[Image Processor (Placeholder)];
    E --> I[Voice Processor (Placeholder)];
    F --> J[Video Processor (Placeholder)];

    G --> K[Text Embeddings];
    H --> L[Image Analysis Results];
    I --> M[Voice Analysis Results];
    J --> N[Video Analysis Results];

    K --> O{FAISS Retrieval (Text DB)};

    subgraph Knowledge Base
        P[Documents]
    end

    O --> P;

    subgraph LLM Response Generation
        Q[Google Gemini (Flash Models)]
    end

    C, L, M, N --> Q;
    O --> Q;

    Q --> R[Text Response];
    Q --> S[Speech Output (Placeholder)];
    H, J --> T[Image/Video Insights];

    R, S, T --> B;
```

## Setup Instructions

Follow these steps to set up and run the application locally.

### 1. Clone the Repository (if applicable)

If this project is part of a repository, clone it first:

```bash
git clone <repository_url>
cd <project_directory>
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a file named `.env` in the root of your project directory.

Obtain a Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/) or the Google Cloud Console. Add it to your `.env` file:

```dotenv
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

(Optional) If you were using other APIs like OpenAI, you might have:

```dotenv
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

## How to Run the Application

Once the setup is complete and your virtual environment is active, run the Streamlit application:

```bash
streamlit run app.py
```

This command will launch the Streamlit application. A new tab should open in your default web browser displaying the RAG system UI.

## Future Enhancements

*   **True Multimodal Embeddings**: Implement a system for generating and retrieving truly multimodal embeddings (e.g., using CLIP or other multimodal models) for cross-modal similarity search.
*   **Integration with Cloud AI Services**: Replace placeholder functions for image, voice, and video processing with actual API calls to services like Google Cloud Vision AI, Speech-to-Text, and Video AI.
*   **Dynamic Knowledge Base**: Implement functionality to load documents from various sources (e.g., databases, cloud storage, web scraping) rather than a hardcoded list.
*   **Advanced Prompt Engineering**: Further refine prompts for the Gemini LLM to improve response quality, summarization, and instruction following.
*   **Chat History and Session Management**: Add features to maintain conversation history and user sessions.
*   **Fine-tuning LLMs**: Explore fine-tuning smaller LLMs on specific domain data for more specialized responses.
*   **User Authentication**: Implement user authentication for secure access to the application and personalized experiences.
