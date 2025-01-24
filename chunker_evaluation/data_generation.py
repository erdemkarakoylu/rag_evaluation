import json
import os
import time
import fitz
from langchain.chat_models import ChatOllama

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.1"


def load_pdf(filepath):
    """
    Extracts text content from a PDF file, optionally removing headers.

    Parameters
    ----------
    filepath : str
        The path to the PDF file.

    Returns
    -------
    str
        The extracted text content.
    """
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text_content = block[4].strip()
            bbox = block[:4]  # Bounding box coordinates
            font_size = block[3]  # Font size

            # Example: Identify headers based on position and font size
            if bbox[1] < 100 and font_size > 12:
                continue  # Skip this block (header)

            text += text_content + "\n"
    return text


def extract_questions(response):
    """
    Extracts questions from the LLM response.

    Parameters
    ----------
    response : str
        The LLM response containing the questions.

    Returns
    -------
    list[str]
        A list of extracted questions.
    """
    questions = []
    lines = response.split("\n")
    for line in lines:
        if line.startswith(tuple(["1.", "2.", "3."])):
            question = line[line.index(".") + 1 :].strip()
            questions.append(question)
    return questions


def extract_answer(response):
    """
    Extracts the answer from the LLM response.

    Parameters
    ----------
    response : str
        The LLM response containing the answer.

    Returns
    -------
    str
        The extracted answer.
    """
    answer = response.split("Answer:")[-1].strip()
    return answer


def generate_qa_dataset(filepath, chunker):
    """
    Generates a QA dataset for a given PDF file and chunker.

    Parameters
    ----------
    filepath : str
        The path to the PDF file.
    chunker : BaseChunker
        The chunker to use for splitting the document.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary represents a
        question-answer pair and includes the corresponding chunk.
    """
    qa_dataset = []
    text = load_pdf(filepath)
    chunks = chunker.split_text(text)

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)  # Initialize LLM here

    for chunk in chunks:
        questions_prompt = f"""You are a helpful and informative question-answering AI.
        Given the following text chunk from a document, generate 3 insightful questions that can be answered using the information within the chunk.

        Text Chunk:
        \"\"\"
        {chunk}
        \"\"\"

        Questions:
        1. 
        2. 
        3. 
        """
        questions_response = llm.predict(questions_prompt)
        questions = extract_questions(questions_response)

        for question in questions:
            answer_prompt = f"""You are a helpful and informative question-answering AI.
            Answer the following question using only the information provided in the context below.

            Context:
            \"\"\"
            {chunk}
            \"\"\"

            Question: {question}

            Answer:
            """
            answer_response = llm.predict(answer_prompt)
            answer = extract_answer(answer_response)

            qa_dataset.append({"question": question, "answer": answer, "chunk": chunk})

    return qa_dataset


def save_qa_dataset(qa_dataset, output_filepath):
    """
    Saves the QA dataset to a JSON file.

    Parameters
    ----------
    qa_dataset : list[dict]
        The QA dataset to save.
    output_filepath : str
        The path to the output JSON file.
    """
    with open(output_filepath, "w") as f:
        json.dump(qa_dataset, f, indent=4)


def generate_and_save_data(pdf_path, chunker, output_dir):
    """
    Generates and saves QA datasets for each PDF using the given chunker.

    Parameters
    ----------
    pdf_path : str
        Path to the directory containing the PDF files.
    chunker : BaseChunker
        The chunker to use for splitting the documents.
    output_dir : str
        Path to the directory where the generated data will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_path, filename)
            start_time = time.time()

            qa_dataset = generate_qa_dataset(filepath, chunker)

            output_filename = os.path.splitext(filename)[0] + "_qa_dataset.json"
            output_filepath = os.path.join(output_dir, output_filename)

            save_qa_dataset(qa_dataset, output_filepath)

            elapsed_time = time.time() - start_time
            print(
                f"Generated and saved data for {filename} in {elapsed_time:.2f} seconds"
            )