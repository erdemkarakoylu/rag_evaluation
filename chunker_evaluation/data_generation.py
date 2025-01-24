import fitz
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
