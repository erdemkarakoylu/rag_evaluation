## Explanation of Metrics

### IoU (Intersection over Union)

<u>Calculation</u>:

1. Tokenization: Split both the retrieved chunks and the expected answer into individual words or tokens.
2. Set Intersection: Find the common tokens between the retrieved chunks and the expected answer.
3. Set Union: Find all unique tokens in both the retrieved chunks and the expected answer.
4. IoU Calculation: Divide the number of common tokens (intersection) by the number of unique tokens (union).


<u>Interpretation</u>:

* Measures the overlap between the retrieved information and the expected answer.
* A higher IoU indicates a greater similarity between the retrieved chunks and the expected answer.
* Ranges from 0 (no overlap) to 1 (perfect overlap).
* Useful for assessing the relevance of the retrieved information.


### Recall

<u>Calculation</u>:

1. Tokenization: Split both the retrieved chunks and the expected answer into individual words or tokens.
2. Set Intersection: Find the common tokens between the retrieved chunks and the expected answer.
3. Recall Calculation: Divide the number of common tokens (intersection) by the total number of tokens in the expected answer.

<u>Interpretation</u>:

* Measures the proportion of relevant information that was successfully retrieved.
* A higher recall means that more of the important information from the expected answer was captured in the retrieved chunks.
* Ranges from 0 (no relevant information retrieved) to 1 (all relevant information retrieved).
* Useful for evaluating the completeness of the retrieved information.


### Precision

<u>Calculation</u>:

1. Tokenization: Split both the retrieved chunks and the expected answer into individual words or tokens.
2. Set Intersection: Find the common tokens between the retrieved chunks and the expected answer.
3. Precision Calculation: Divide the number of common tokens (intersection) by the total number of tokens in the retrieved chunks.

<u>Interpretation</u>:

* Measures the proportion of retrieved information that is actually relevant.
* A higher precision means that the retrieved chunks contain more relevant information and less irrelevant information.
* Ranges from 0 (no relevant information retrieved) to 1 (all retrieved information is relevant).
* Useful for evaluating the accuracy or focus of the retrieved information.

### Precision Omega

<u>Calculation</u>:

1. Calculated as:
    ```precision * (iou + recall) / 2 in the code```
2. This is a weighted combination of precision, IoU, and recall.

<u>Interpretation</u>:

* Provides a balanced measure that considers both the accuracy and completeness of the retrieved information.
* The weighting gives equal importance to IoU and recall when adjusting the precision.
* A higher precision omega suggests a better overall retrieval performance, considering both relevance and comprehensiveness.

### F1-score

<u>Calculation</u>:

1. F1 Calculation: The F1-score is the harmonic mean of precision and recall. 
2. It is calculated as:

```F1 = 2 * (precision * recall) / (precision + recall)```

<u>Interpretation</u>:

* Provides a single metric that balances precision and recall.
* A higher F1-score indicates a better balance between accuracy (precision) and completeness (recall).
* Ranges from 0 (worst) to 1 (best).
* Useful when you need a single metric to compare different chunking strategies, especially when there is a trade-off between precision and recall.