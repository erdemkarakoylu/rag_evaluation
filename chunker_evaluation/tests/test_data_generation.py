import pytest
import json
import os
from unittest.mock import patch, Mock
import configparser
import fitz
from langchain.chat_models import ChatOllama
import data_generation

# --- Fixtures ---

@pytest.fixture
def mock_config():
    """Mocks the configparser to avoid reading from actual config.ini"""
    mock_config = Mock(spec=configparser.ConfigParser)
    mock_config.read.return_value = None
    mock_config.get.side_effect = lambda *args: "mock_value"
    with patch.object(configparser, "ConfigParser", return_value=mock_config):
        yield mock_config

@pytest.fixture
def mock_pdf_document():
    """Mocks a fitz.Document object with some example content."""
    mock_document = Mock(spec=fitz.Document)
    mock_page = Mock()
    mock_page.get_text.return_value = [
        (0, 0, 100, 20, 12, "Header", 0, 0, 0),  # Example header
        (0, 200, 100, 220, 10, "Some text content.", 0, 0, 0),  # Example content
    ]
    mock_document.__iter__.return_value = [mock_page]
    with patch.object(fitz, "open", return_value=mock_document):
        yield mock_document

@pytest.fixture
def mock_llm_response():
    """Mocks the LLM response with predefined questions and answer."""
    questions_response = """
    1. What is the capital of France?
    2. Who painted the Mona Lisa?
    3. What is the highest mountain in the world?
    """
    answer_response = "Answer: Paris"
    with patch.object(ChatOllama, "predict") as mock_predict:
        mock_predict.side_effect = [questions_response, answer_response]
        yield mock_predict

@pytest.fixture
def sample_qa_dataset():
    """Provides a sample QA dataset for testing."""
    return [
        {"question": "Question 1", "answer": "Answer 1", "chunk": "Chunk 1"},
        {"question": "Question 2", "answer": "Answer 2", "chunk": "Chunk 2"},
        {"question": "Question 3", "answer": "Answer 3", "chunk": "Chunk 3"}
    ]

@pytest.fixture
def output_filepath(tmp_path):
    """Creates a temporary file for saving the QA dataset."""
    return os.path.join(tmp_path, "test_qa_dataset.json")

# --- Tests ---

def test_load_pdf(mock_pdf_document):
    """Tests the load_pdf function."""
    filepath = "test.pdf"  # This is just a dummy filepath, the mock handles the rest
    text_content = data_generation.load_pdf(filepath)
    assert "Some text content." in text_content
    assert "Header" not in text_content  # Ensure header is removed

def test_extract_questions():
    """Tests the extract_questions function."""
    response = """
    1. What is the capital of France?
    2. Who painted the Mona Lisa?
    3. What is the highest mountain in the world?
    """
    questions = data_generation.extract_questions(response)
    assert questions == [
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
        "What is the highest mountain in the world?",
    ]

def test_extract_answer():
    """Tests the extract_answer function."""
    response = "Answer: This is the answer."
    answer = data_generation.extract_answer(response)
    assert answer == "This is the answer."

def test_generate_qa_dataset(mock_llm_response, mock_pdf_document):
    """Tests the generate_qa_dataset function."""
    chunker = Mock()
    chunker.split_text.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
    qa_dataset = data_generation.generate_qa_dataset("test.pdf", chunker)
    assert len(qa_dataset) == 9  # 3 chunks * 3 questions per chunk = 9

def test_save_qa_dataset(sample_qa_dataset, output_filepath):
    """Tests the save_qa_dataset function."""
    data_generation.save_qa_dataset(sample_qa_dataset, output_filepath)
    with open(output_filepath, "r") as f:
        loaded_data = json.load(f)
    assert loaded_data == sample_qa_dataset