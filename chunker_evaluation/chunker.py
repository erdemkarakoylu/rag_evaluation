import re
from chunking_evaluation import BaseChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter


class SimpleSplitter(BaseChunker):
    """Splits text into paragraphs using two or more newlines as delimiter."""

    def split_text(self, text):
        paragraphs = re.split(r"\n\s*\n", text)
        return [para.strip() for para in paragraphs if para.strip() != ""]


# Define the chunker implementations
chunker_implementations = {
    "RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)": RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    ),
    "RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)": RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    ),
    "RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)": RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ),
    "SimpleSplitter": SimpleSplitter(),
}