from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


def parse_pdf(file_path: str) -> str:
    try:
        text = extract_text(file_path)
        if not text or not text.strip():
            raise ValueError("PDF appears to be empty or scanned (no extractable text).")
        return text.strip()
    except PDFSyntaxError:
        raise ValueError("Invalid or corrupted PDF file.")