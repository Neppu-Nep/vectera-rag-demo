import time
import streamlit as st

from src.core.logger import get_logger
from src.generator import generate_answer
from src.ingestion import ingest_pdf
from src.retriever import retrieve_context

logger = get_logger(__name__)


def _format_score(value: object) -> float | None:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def init_session() -> None:
    """Initialize the Streamlit session state fields if not present.

    Sets up the default conversational chat history and a unique client ID.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "client_id" not in st.session_state:
        st.session_state.client_id = "Vectera_Capital_Fund"
        logger.info("Initialized new session with default Tenant: Vectera_Capital_Fund")
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "files_to_process" not in st.session_state:
        st.session_state.files_to_process = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False


def handle_tenant_change() -> None:
    """Reset chat history when tenant changes."""
    st.session_state.messages = []
    logger.info("Tenant changed; chat history reset.")


def render_sidebar() -> None:
    """Render the sidebar UI components for PDF ingestion.

    Allows the user to upload one or more PDFs, parses them via LlamaCloud,
    and enqueues their text chunks for Supabase insertion.
    """
    with st.sidebar:
        st.header("Settings")
        st.selectbox(
            "Active Tenant Scope", 
            options=["Vectera_Capital_Fund", "Rival_Private_Equity", "Default_Client"],
            key="client_id",
            on_change=handle_tenant_change,
            help="Demonstrating database-level row segregation. Documents uploaded to one tenant cannot be retrieved by another."
        )
        st.toggle("Enable Retrieval Debug Mode", key="debug_mode")

        st.header("Upload Document")
        uploaded_files = st.file_uploader(
            "Choose a PDF document", 
            type=["pdf"], 
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if st.button("Process Documents", disabled=st.session_state.is_processing) and uploaded_files:
            st.session_state.files_to_process = [
                {"name": f.name, "bytes": f.read()} for f in uploaded_files
            ]
            st.session_state.is_processing = True
            st.session_state.uploader_key += 1
            st.rerun()


def render_chat() -> None:
    """Render the central chat UI, retrieving context and generating responses."""
    st.header("Chat with your Data")

    # Render historical messages
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander(f"View {len(msg['sources'])} sources used"):
                    for chunk in msg["sources"]:
                        doc_name = chunk.get("document_name", "Unknown Document")
                        version = chunk.get("document_version", "Unknown")
                        page_number = chunk.get("page_number", "N/A")
                        text_full = chunk.get("raw_content") or chunk.get("chunk_text", "")
                        with st.expander(f"**{doc_name} | {version} | Page {page_number}**"):
                            st.text(text_full)

    # Wait for user input
    user_query = st.chat_input("Ask a question about your documents...", disabled=st.session_state.is_processing)
    if user_query:
        # Add user query to history and render it immediately
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching knowledge base..."):
                    client_id = st.session_state.client_id
                    # Retrieve from Supabase and rerank
                    logger.info(f"Processing query from {client_id}: {user_query}")
                    chunks, filters = retrieve_context(
                        user_query, client_id, rerank_with_model=True
                    )

                filter_parts = []
                if filters.years:
                    filter_parts.append(f"Years: {filters.years}")
                if filters.quarters:
                    filter_parts.append(f"Quarters: {filters.quarters}")
                if filters.companies:
                    filter_parts.append(f"Companies: {filters.companies}")

                if filter_parts:
                    st.info("Active Filters Applied: " + " | ".join(filter_parts))
                else:
                    st.info("Active Filters Applied: None")

                if not chunks:
                    answer = "I could not find any relevant information in the uploaded documents."
                else:
                    with st.spinner("Thinking..."):
                        answer = generate_answer(user_query, chunks)

                st.markdown(answer)
                st.feedback("thumbs")

                if chunks:
                    with st.expander(f"View {len(chunks)} sources used"):
                        for chunk in chunks:
                            doc_name = chunk.get("document_name", "Unknown Document")
                            version = chunk.get("document_version", "Unknown")
                            page_number = chunk.get("page_number", "N/A")
                            text_full = chunk.get("raw_content") or chunk.get("chunk_text", "")
                            with st.expander(f"**{doc_name} | {version} | Page {page_number}**"):
                                st.text(text_full)

                if st.session_state.debug_mode:
                    with st.expander("⚙️ Debug: Retrieval Metrics"):
                        if not chunks:
                            st.write("No retrieval metrics available.")
                        else:
                            rows = []
                            for chunk in chunks:
                                chunk_id = str(chunk.get("id", ""))[:8]
                                rows.append(
                                    {
                                        "Chunk ID": chunk_id,
                                        "Document": chunk.get(
                                            "document_name", "Unknown Document"
                                        ),
                                        "Similarity": _format_score(
                                            chunk.get("similarity")
                                        ),
                                        "RRF Score": _format_score(
                                            chunk.get("rrf_score")
                                        ),
                                    }
                                )
                            st.dataframe(rows, use_container_width=True)

                # Store the final assistant answer in session history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": chunks}
                )

            except Exception as e:
                logger.error(f"Failed to answer query: {e}", exc_info=True)
                error_msg = "An unexpected error occurred while generating your answer."
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )


def process_pending_files() -> None:
    """Process queued PDF files and store results in session state."""
    if not st.session_state.files_to_process:
        return

    client_id = st.session_state.client_id

    status_placeholder = st.empty()
    for f in st.session_state.files_to_process:
        file_bytes = f["bytes"]
        filename = f["name"]
        with status_placeholder.status(f"Processing '{filename}'..."):
            try:
                # progress_cb writes into the status expander
                result = ingest_pdf(file_bytes, filename, client_id, progress_cb=lambda msg: st.write(f" - {msg}"))
                
                if result.get("skipped"):
                    # If duplicate, treat as silent success
                    if result.get("reason") in ["file_sha256_exists", "text_sha256_exists"]:
                        msg = f"Successfully processed {filename}"
                        st.toast(msg, icon="✅")
                    else:
                        msg = f"Skipped {filename}: {result.get('reason')}"
                        st.warning(msg)
                else:
                    inserted = result.get("inserted", 0)
                    msg = f"Successfully processed {filename}: {inserted} chunks inserted."
                    st.toast(msg, icon="✅")
            except Exception as e:
                logger.error(f"Failed to ingest document {filename}: {e}", exc_info=True)
                st.error(f"Error processing '{filename}'")
            finally:
                # Clear the status expander before starting the next file or finishing
                status_placeholder.empty()

    st.session_state.files_to_process = None
    st.session_state.is_processing = False
    
    # Give the user time to see the last toast before the app reruns and refreshes the UI
    time.sleep(2)
    st.rerun()


def main() -> None:
    """Orchestrate the application layout and logic."""
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    init_session()
    render_sidebar()
    render_chat()
    process_pending_files()


if __name__ == "__main__":
    main()
