from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from llama_cloud.types.parsing_get_response import MarkdownPageMarkdownResultPage

from src.core.logger import get_logger
from src.utils.file_utils import temp_file_from_bytes

load_dotenv()

logger = get_logger(__name__)


def parse_financial_pdf(file_bytes: bytes) -> list[str]:
    """Upload and parse a PDF synchronously using LlamaCloud.

    Args:
        file_bytes: File bytes in memory to parse.

    Returns:
        A list of per-page markdown strings.
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
                        "inline_images": True,
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
                expand=["markdown"],
            )

            logger.info("Parsing Complete! Fetching results...")
            pages: list[str] = []
            if not result.markdown:
                logger.error("No markdown found in result")
                return []

            for page in result.markdown.pages:
                if isinstance(page, MarkdownPageMarkdownResultPage) and page.markdown.strip():
                    pages.append(page.markdown.strip())

            return pages

        except Exception as e:
            logger.error(f"LlamaCloud API Error: {e}", exc_info=True)
            return []
