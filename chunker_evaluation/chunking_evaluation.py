import configparser
import json
import os

import numpy as np
from langchain.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from data_generation import load_pdf


# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Define global variables
EMBEDDING_MODEL_NAME = config.get('MODEL', 'embedding_model_name')
TOP_K_CHUNKS = config.getint('MODEL', 'top_k_chunks')

def calculate_metrics(retrieved_chunks, expected_answer):
    """
    Calculates evaluation metrics for a given question-answer pair.

    Parameters
    ----------
    retrieved_chunks : list[str]
        A list of retrieved chunks.
    expected_answer : str
        The expected answer for the question.

    Returns
    -------
    dict[str, float]
        A dictionary containing the calculated metrics:
        - iou: Intersection over Union.
        - recall: Recall.
        - precision_omega: Weighted combination of precision, IoU, and recall.
        - precision: Precision.
    """
    retrieved_text = " ".join(retrieved_chunks)
    intersection = len(set(retrieved_text.split()) & set(expected_answer.split()))
    union = len(set(retrieved_text.split()) | set(expected_answer.split()))
    iou = intersection / union if union else 0
    recall = (
        intersection / len(set(expected_answer.split())) if expected_answer else 0
    )
    precision = intersection / len(set(retrieved_text.split())) if retrieved_text else 0
    precision_omega = precision * (iou + recall) / 2
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (
        precision + recall) if (precision + recall) else 0
    
    return {
        "iou": iou,
        "recall": recall,
        "precision_omega": precision_omega,
        "precision": precision,
        "f1": f1
    }


def retrieve_top_k_chunks(chunks, chunk_embeddings, question_embedding, k=TOP_K_CHUNKS):
    """
    Retrieves the top-k most similar chunks to the question embedding.

    Parameters
    ----------
    chunks : list[str]
        A list of text chunks.
    chunk_embeddings : list[np.ndarray]
        A list of embeddings for the chunks.
    question_embedding : np.ndarray
        The embedding of the question.
    k : int, optional
        The number of top chunks to retrieve, by default 5.

    Returns
    -------
    list[str]
        A list of the top-k most similar chunks.
    """
    similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_k_indices]


def evaluate_chunking_strategy(pdf_path, chunker, output_dir):
    """
    Evaluates a chunking strategy and returns the calculated metrics.

    Parameters
    ----------
    pdf_path : str
        Path to the directory containing the PDF files.
    chunker : BaseChunker
        The chunker to use for splitting the documents.
    output_dir : str
        Path to the directory where the generated data is saved.

    Returns
    -------
    dict[str, float]
        A dictionary containing the aggregated metrics:
        - iou_mean: Mean Intersection over Union.
        - iou_std: Standard deviation of Intersection over Union.
        - recall_mean: Mean Recall.
        - recall_std: Standard deviation of Recall.
        - precision_omega_mean: Mean weighted combination of precision, IoU, and recall.
        - precision_omega_std: Standard deviation of the weighted combination.
        - precision_mean: Mean Precision.
        - precision_std: Standard deviation of Precision.
    """
    results = []
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    for filename in os.listdir(pdf_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_path, filename)

            # Load the corresponding QA dataset
            dataset_filename = os.path.splitext(filename)[0] + "_qa_dataset.json"
            dataset_filepath = os.path.join(
                output_dir, chunker.__class__.__name__, dataset_filename
            )
            with open(dataset_filepath, "r") as f:
                qa_dataset = json.load(f)

            for qa_pair in qa_dataset:
                question = qa_pair["question"]
                expected_answer = qa_pair["answer"]

                # Embed the question
                question_embedding = embeddings.embed_query(question)

                # Embed all chunks for retrieval
                text = load_pdf(filepath)
                chunks = chunker.split_text(text)
                chunk_embeddings = embeddings.embed_documents(chunks)

                # Retrieve the top-k chunks
                retrieved_chunks = retrieve_top_k_chunks(
                    chunks, chunk_embeddings, question_embedding
                )

                # Calculate metrics
                metrics = calculate_metrics(retrieved_chunks, expected_answer)
                results.append(metrics)

    # Aggregate metrics
    iou_mean = np.mean([result["iou"] for result in results])
    iou_std = np.std([result["iou"] for result in results])
    recall_mean = np.mean([result["recall"] for result in results])
    recall_std = np.std([result["recall"] for result in results])
    precision_omega_mean = np.mean([result["precision_omega"] for result in results])
    precision_omega_std = np.std([result["precision_omega"] for result in results])
    precision_mean = np.mean([result["precision"] for result in results])
    precision_std = np.std([result["precision"] for result in results])

    return {
        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "precision_omega_mean": precision_omega_mean,
        "precision_omega_std": precision_omega_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        
    }