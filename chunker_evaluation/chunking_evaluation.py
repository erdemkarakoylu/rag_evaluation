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

