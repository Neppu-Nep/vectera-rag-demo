import io
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage
from pypdf import PdfReader, PdfWriter
from reducto import Reducto

from src.core.config import settings
from src.core.logger import get_logger
from src.utils.file_utils import temp_file_from_bytes

load_dotenv()

logger = get_logger(__name__)


def llama_parse_financial_pdf(file_bytes: bytes, filename: str = "document") -> list[str]:
    """Upload and parse a PDF synchronously using LlamaCloud.

    Args:
        file_bytes: File bytes in memory to parse.
        filename: The filename to use for the temp file prefix.

    Returns:
        A list of per-page markdown strings.
    """
    with temp_file_from_bytes(file_bytes, prefix=filename) as tmp_path:
        logger.info("Initializing LlamaCloud for parsing...")

        try:
            # The client automatically picks up LLAMA_CLOUD_API_KEY from the environment
            client = LlamaCloud()

            logger.info("Uploading document to LlamaCloud...")
            file = client.files.create(file=tmp_path, purpose="parse")

            logger.info("Awaiting Agentic parse job...")
            result = client.parsing.parse(
                file_id=file.id,
                tier="agentic_plus",
                version="latest",
                output_options={
                    "markdown": {
                        "inline_images": True,
                        "tables": {
                            "output_tables_as_markdown": True,
                            "merge_continued_tables": True,
                        },
                    }
                },
                processing_control={
                    "timeouts": {
                        "base_in_seconds": 120,
                        "extra_time_per_page_in_seconds": 2,
                    }
                },
                expand=["markdown"],
            )

            logger.info("Parsing Complete! Fetching results...")
            pages: list[str] = []
            if not result.markdown:
                logger.error("No markdown found in result")
                return []

            for page in result.markdown.pages:
                if (
                    isinstance(page, MarkdownPageMarkdownResultPage)
                    and page.markdown.strip()
                ):
                    pages.append(page.markdown.strip())

            return pages

        except Exception as e:
            logger.error(f"LlamaCloud API Error: {e}", exc_info=True)
            return []


def reducto_parse_financial_pdf(file_bytes: bytes, filename: str = "document") -> list[str]:
    """Parse a PDF into markdown chunks using Reducto.

    Args:
        file_bytes: File bytes in memory to parse.
        filename: The original filename.

    Returns:
        A list of markdown strings per chunk.
    """
    logger.info("Initializing Reducto for spatial financial parsing (Single-Page Sequential Mode)...")

    try:
        client = Reducto(api_key=settings.reducto_api_key)

        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        pages_markdown: list[str] = []

        base_name = filename.replace('.pdf', '')

        for i in range(total_pages):
            logger.info(f"Processing page {i + 1}/{total_pages} of {filename} with Reducto...")

            writer = PdfWriter()
            writer.add_page(reader.pages[i])

            single_page_io = io.BytesIO()
            writer.write(single_page_io)
            single_page_bytes = single_page_io.getvalue()

            # Inject the original filename and page number into the temp file prefix
            page_prefix = f"{base_name}_pg{i+1}"

            with temp_file_from_bytes(single_page_bytes, prefix=page_prefix) as tmp_path:
                upload_response = client.upload(file=Path(tmp_path))
                result = client.parse.run(
                    input=upload_response,
                    enhance={
                        "agentic": [
                            {
                                "scope": "figure",
                                "prompt": "Extract all underlying data from charts, graphs, and figures into clean, structured Markdown tables. Use logical column headers dynamically derived from the axes, legends, or data labels. Explicitly exclude all visual metadata describing the image. Strictly preserve all numerical values, negative signs, parentheses, percentages, dates, and superscript footnotes exactly as they appear. If multiple charts on the same page share the same context or subjects, consolidate them into a single unified relational table.",
                                "advanced_chart_agent": True,
                            },
                            {
                                "scope": "table",
                            },
                        ],
                        "intelligent_ordering": True,
                    },
                    retrieval={
                        "chunking": {
                            "chunk_mode": "page",
                            "chunk_size": 1000,
                        },
                        "embedding_optimized": True,
                    },
                    formatting={
                        "table_output_format": "md",
                        "merge_tables": True,
                    },
                    spreadsheet={
                        "split_large_tables": {
                            "enabled": False,
                        }
                    },
                    settings={
                        "timeout": 300,
                    },
                )  # type: ignore

                page_content: list[str] = []
                if result.result and result.result.chunks:
                    for chunk in result.result.chunks:
                        if chunk.content and chunk.content.strip():
                            page_content.append(chunk.content.strip())

                if page_content:
                    pages_markdown.append('\n\n'.join(page_content))
                else:
                    pages_markdown.append('')

            time.sleep(1.5)

        return pages_markdown

    except Exception as e:
        logger.error(f"Reducto API Error: {e}", exc_info=True)
        return []
