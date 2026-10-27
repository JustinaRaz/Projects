import logging
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from src.core.settings import get_settings

settings = get_settings()
logger = logging.getLogger("src")

_converter = DocumentConverter()

def extract_document(contents: bytes, filename: str) -> DoclingDocument:
    """Parse raw file bytes into a structured DoclingDocument.

    Args:
        contents: Raw bytes of the uploaded file.
        filename: Original filename, used by docling for format detection.

    Returns:
        The parsed DoclingDocument, containing text, tables and layout.

    Raises:
        DocumentParsingError: If docling cannot parse the file (corrupt
            file, unsupported format, etc.).
    """
    source = DocumentStream(name=filename, stream=BytesIO(contents)) # Packages file

    try:
        result = _converter.convert(source)
    except Exception as exc:
        error_message = f"Failed to parse document '{filename}'"
        logger.exception(error_message)
        raise RuntimeError(error_message) from exc

    document = result.document # From raw file to DoclingDocument

    markdown = document.export_to_markdown()
    output_path = settings.text_upload_dir / f"{Path(filename).stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    logger.info(f"You can inspect the document at {output_path}.")

    return document
