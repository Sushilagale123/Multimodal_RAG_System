import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np
# Import the actual instances directly from app.py
from app import get_text_embedding, RAGSystem, generate_response, model as app_sentence_model, gemini_model as app_gemini_model

# Mock the SentenceTransformer instance in app.py (still needed for test_get_text_embedding directly)
@pytest.fixture
def mock_sentence_transformer_instance():
    with patch('app.model') as mock_model:
        mock_model.encode.return_value = np.array([0.1]*384, dtype='float32')
        yield mock_model

# Test cases for Text Embedding
def test_get_text_embedding(mock_sentence_transformer_instance):
    text = "test sentence"
    embedding = get_text_embedding(text, embedding_model=mock_sentence_transformer_instance) # Inject mock
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    mock_sentence_transformer_instance.encode.assert_called_with(text)

# Test cases for Retrieval Component
@pytest.fixture
def sample_documents():
    return [
        "The capital of France is Paris.",
        "Python is a programming language.",
        "Streamlit is for building web apps."
    ]

@pytest.fixture
def mock_embedding_function():
    # This fixture now simply yields a fresh MagicMock for each test
    yield MagicMock()

@pytest.fixture
def rag_system_instance(sample_documents, mock_embedding_function):
    # Configure side_effect for RAGSystem initialization
    init_embeddings = [
        np.array([0.9, 0.1, 0.1] + [0.0]*(384-3), dtype='float32'), # Doc 1: Paris
        np.array([0.1, 0.9, 0.1] + [0.0]*(384-3), dtype='float32'), # Doc 2: Python
        np.array([0.1, 0.1, 0.9] + [0.0]*(384-3), dtype='float32'), # Doc 3: Streamlit
    ]
    mock_embedding_function.side_effect = init_embeddings
    
    system = RAGSystem(sample_documents, embedding_function=mock_embedding_function)

    # After initialization, reset mock and set a default return_value for retrieval queries
    mock_embedding_function.reset_mock()
    mock_embedding_function.return_value = np.array([0.5]*384, dtype='float32') # Default for query embeddings

    yield system

def test_rag_system_init(rag_system_instance, sample_documents):
    system = rag_system_instance
    assert len(system.documents) == len(sample_documents)
    assert system.dimension == 384

def test_rag_system_retrieve_single_document(rag_system_instance, mock_embedding_function):
    system = rag_system_instance
    query = "France capital"
    # Configure the mock specifically for this retrieval query
    query_embedding_val = np.array([0.9, 0.1, 0.1] + [0.0]*(384-3), dtype='float32')
    mock_embedding_function.return_value = query_embedding_val
    
    retrieved = system.retrieve(query, k=1)
    assert len(retrieved) == 1
    assert retrieved[0] == "The capital of France is Paris."
    mock_embedding_function.assert_called_once_with(query)

def test_rag_system_retrieve_multiple_documents(rag_system_instance, mock_embedding_function):
    system = rag_system_instance
    query = "programming language and web apps"
    # Configure the mock specifically for this retrieval query
    query_embedding_val = np.array([0.1, 0.8, 0.8] + [0.0]*(384-3), dtype='float32')
    mock_embedding_function.return_value = query_embedding_val

    retrieved = system.retrieve(query, k=2)
    assert len(retrieved) == 2
    assert "Python is a programming language." in retrieved
    assert "Streamlit is for building web apps." in retrieved
    mock_embedding_function.assert_called_once_with(query)

# Mock the Gemini model instance in app.py
@pytest.fixture
def mock_gemini_model_instance():
    with patch('app.gemini_model') as mock_gemini:
        mock_gemini.generate_content.return_value.text = "This is a generated response based on the context."
        yield mock_gemini

def test_generate_response(mock_gemini_model_instance):
    query = "What is the capital of France?"
    retrieved_docs = ["The capital of France is Paris."]
    response = generate_response(query, retrieved_docs, llm_model=mock_gemini_model_instance)

    expected_prompt_part = f"You are a helpful assistant. Based on the following context, answer the query comprehensively.\n\nContext: {retrieved_docs[0]}\n\nQuery: {query}\n\nAnswer:"
    
    # Expected generation config and safety settings from app.py defaults
    expected_generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    expected_safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    mock_gemini_model_instance.generate_content.assert_called_once_with(
        [expected_prompt_part],
        generation_config=expected_generation_config,
        safety_settings=expected_safety_settings
    )
    assert response == "This is a generated response based on the context."

def test_generate_response_error_handling(mock_gemini_model_instance):
    mock_gemini_model_instance.generate_content.side_effect = Exception("API Error")

    query = "Test query"
    retrieved_docs = ["Test document."]
    response = generate_response(query, retrieved_docs, llm_model=mock_gemini_model_instance)

    # The expected call will also include generation_config and safety_settings
    expected_prompt_part = f"You are a helpful assistant. Based on the following context, answer the query comprehensively.\n\nContext: {retrieved_docs[0]}\n\nQuery: {query}\n\nAnswer:"
    expected_generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 1,
        "max_output_tokens": 2048,
    }
    expected_safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    mock_gemini_model_instance.generate_content.assert_called_once_with(
        [expected_prompt_part],
        generation_config=expected_generation_config,
        safety_settings=expected_safety_settings
    )
    assert response == "Could not generate a response with Gemini."
