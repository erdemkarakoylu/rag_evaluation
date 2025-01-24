import configparser
import os
import time

import pandas as pd

from chunker import chunker_implementations
from data_generation import generate_and_save_data
from chunker_evaluation.chunking_evaluation import evaluate_chunking_strategy
from analysis import analyze_chunk_lengths, count_chunks
from loguru import logger

# Configuration
#OLLAMA_BASE_URL = "http://localhost:11434"
#PDF_DIR = "path/to/your/pdfs"
#OUTPUT_DIR = "path/to/your/output/directory"

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Define global variables
PDF_DIR = config.get('DATA', 'pdf_dir')
OUTPUT_DIR = config.get('DATA', 'output_dir')

if __name__ == "__main__":
    # --- Data Generation ---
    for chunker_name, chunker in chunker_implementations.items():
        logger.info(f"Generating data for chunker: {chunker_name}")
        start_time = time.time()

        generate_and_save_data(
            PDF_DIR, chunker, os.path.join(OUTPUT_DIR, chunker_name)
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Total time for {chunker_name}: {elapsed_time:.2f} seconds")

    # --- Evaluation ---
    df_metrics = pd.DataFrame()

    for chunker_name, chunker in chunker_implementations.items():
        logger.info(f"Evaluating chunker: {chunker_name}")
        start_time = time.time()

        results = evaluate_chunking_strategy(PDF_DIR, chunker, OUTPUT_DIR)

        elapsed_time = time.time() - start_time
        logger.info(f"Evaluation time for {chunker_name}: {elapsed_time:.2f} seconds")
        logger.info(results)

        df_metrics = pd.concat(
            [df_metrics, pd.DataFrame(results, index=[chunker_name])]
        )
        df_metrics.to_csv("chunker_metrics.csv")

    # --- Analysis ---
    analyze_chunk_lengths(OUTPUT_DIR)
    df_chunks = count_chunks(OUTPUT_DIR)
    logger.info("\nNumber of Chunks per Chunker and Document:\n", df_chunks)