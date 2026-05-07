from dotenv import load_dotenv
from llama_cloud import LlamaCloud

from src.core.logger import get_logger
from src.utils.file_utils import temp_file_from_bytes

load_dotenv()

logger = get_logger(__name__)


def parse_financial_pdf(file_bytes: bytes) -> str:
    """Upload and parse a PDF synchronously using LlamaCloud.

    Args:
        file_bytes: File bytes in memory to parse.

    Returns:
        The markdown string result from LlamaCloud.
    """
    with temp_file_from_bytes(file_bytes) as tmp_path:
        logger.info("Initializing LlamaCloud for parsing...")

        try:
            # The client automatically picks up LLAMA_CLOUD_API_KEY from the environment
            client = LlamaCloud()

            logger.info("Uploading document to LlamaCloud...")
            file = client.files.create(file=tmp_path, purpose="parse")

            logger.info("Awaiting Agentic parse job...")
            result = client.parsing.parse(
                file_id=file.id,
                tier="agentic",
                version="latest",
                output_options={
                    "markdown": {
                        "inline_images": False,
                        "tables": {
                            "output_tables_as_markdown": True,
                            "merge_continued_tables": True,
                        },
                    }
                },
                processing_options={
                    "cost_optimizer": {
                        "enable": True
                    },
                    "ignore": {
                        "ignore_hidden_text": True
                    },
                },
                processing_control={
                    "timeouts": {
                        "base_in_seconds": 120,
                        "extra_time_per_page_in_seconds": 2,
                    }
                },
                expand=["markdown_full"],
            )

            logger.info("Parsing Complete! Fetching results...")
            full_text = result.markdown_full or ""
            logger.debug(f"Preview of extracted text: {full_text[:500]}...")
            return full_text

        except Exception as e:
            logger.error(f"LlamaCloud API Error: {e}", exc_info=True)
            return ""
