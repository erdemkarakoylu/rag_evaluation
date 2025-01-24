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
    
    return {
        "iou": iou,
        "recall": recall,
        "precision_omega": precision_omega,
        "precision": precision,
    }
