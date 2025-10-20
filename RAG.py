from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http import models as rest
from qdrant_client import QdrantClient
import os
import io
import re
import json
import time
import uuid
import base64
import hashlib
import requests
import streamlit as st
from pathlib import Path
from typing import List, Tuple
import datetime
import pandas as pd
import hmac

# OCR / PDF
import fitz  # PyMuPDF
from PIL import Image
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
# Qdrant + Embeddings

load_dotenv()  # auto-load variables from .env


# ===============================
# ENV / CONFIG
# ===============================
st.set_page_config(page_title="BRA Team Contract Document Handling",
                   page_icon="📚", layout="wide")

# Authentication
# Google OAuth configuration will be in secrets.toml
AUTHORIZED_DOMAINS = os.getenv("AUTHORIZED_DOMAINS", "")  # Comma-separated list of domains
# Password authentication
AUTH_EMAIL = os.getenv("AUTH_EMAIL")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")

# Set your envs once (use your real values or .env)
# You can also keep them in OS env already set in your system.
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

DEKA_BASE = os.getenv("DEKA_BASE_URL")
DEKA_KEY = os.getenv("DEKA_KEY")
OCR_MODEL = "meta/llama-4-maverick-instruct"
EMBED_MODEL = os.getenv("EMBED_MODEL", "baai/bge-multilingual-gemma2")

ALLOWED_LANGS = {"en", "id"}
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Supabase
# ---- Supabase (for Index Table)
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. "https://xyzcompany.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "Indexing"

# ===============================
# CONNECTORS
# ===============================
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
deka_client = OpenAI(api_key=DEKA_KEY, base_url=DEKA_BASE)


def ensure_collection_and_indexes(dim: int):
    # Create collection if missing
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(
                size=dim, distance=rest.Distance.COSINE),
        )
    # Ensure payload indexes for common filters
    for field in ["metadata.source", "metadata.company", "metadata.doc_id"]:
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            # ignore "already exists"
            if "already exists" not in str(e).lower():
                st.warning(f"Index create failed for {field}: {e}")


def build_embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=DEKA_KEY,
        base_url=DEKA_BASE,
        model=EMBED_MODEL,
        encoding_format="float",
    )

# ===============================
# HELPERS
# ===============================


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\u200b|\u200c|\u200d|\ufeff", "", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def keep_language(text: str, allowed_langs=ALLOWED_LANGS) -> bool:
    try:
        lang = detect(text[:1000])
        return lang in allowed_langs
    except Exception:
        return True


def deterministic_doc_hash(full_path: Path, content_bytes: bytes) -> str:
    """
    Hash that is stable across runs; if file path is unknown, use content hash.
    """
    try:
        stat = full_path.stat()
        blob = f"{full_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
    except Exception:
        # fallback on content
        blob = hashlib.sha1(content_bytes).hexdigest()
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def page_image_base64(pdf_doc, page_index: int, zoom: float = 3.0) -> str:
    page = pdf_doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_meta_header(meta: dict) -> str:
    company = (meta or {}).get("company", "N/A")
    source = (meta or {}).get("source", "N/A")
    page = (meta or {}).get("page", "N/A")
    return f"Company: {company}\nDocument: {source}\nPage: {page}\n---\n"

# ===============================
# SUPABASE HELPERS
# ===============================
def add_to_supabase(company_name: str, document_name: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.warning("Supabase credentials missing — skipping index insert.")
        return

    payload = {"Company Name": company_name, "Contract Title": document_name}
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            },
            json=payload
        )
        if r.status_code not in (200, 201, 204):
            st.warning(f"⚠️ Supabase insert failed: {r.status_code} {r.text}")
    except Exception as e:
        st.warning(f"Supabase insert error: {e}")


def delete_from_supabase(document_name: str):
    if not SUPABASE_URL or not SUPABASE_KEY:
        st.warning("Supabase credentials missing — skipping index delete.")
        return
    try:
        r = requests.delete(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            },
            params={"Contract Title": f"eq.{document_name}"}
        )
        if r.status_code not in (200, 204):
            st.warning(f"⚠️ Supabase delete failed: {r.status_code} {r.text}")
    except Exception as e:
        st.warning(f"Supabase delete error: {e}")

# ===============================
# OCR (DEKA Maverick) per page
# ===============================


def ocr_pdf_with_deka(pdf_path: Path, company: str, source_name: str, progress_ocr, status_ocr) -> List[dict]:
    """
    Returns a list of dicts:
    { "page": int, "text": str, "lang_mismatch": bool, "words": int }
    """
    pages_out = []
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    success_pages = 0

    for i in range(total_pages):
        status_ocr.write(f"🖼️ OCR page {i+1}/{total_pages}")
        b64_image = page_image_base64(doc, i, zoom=3.0)

        # Call DEKA OCR
        resp = deka_client.chat.completions.create(
            model=OCR_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR engine specialized in Indonesian/English legal and technical contracts. "
                        "Your task is to extract text *exactly as it appears* in the document image, without rewriting or summarizing.\n\n"
                        "Guidelines:\n"
                        "- Preserve all line breaks, numbering, and indentation.\n"
                        "- Keep all headers, footers, and notes if they appear in the image.\n"
                        "- Preserve tables as text: keep rows and columns aligned with | separators. output it in Markdown table format Pad cells so that columns align visually.\n"
                        "- Do not translate text — output exactly as in the document.\n"
                        "- If a cell or field is blank, or contains only dots/dashes (e.g., '.....', '—'), write N/A.\n"
                        "- Keep units, percentages, currency (e.g., m², kVA, %, Rp.) exactly as written.\n"
                        "- If text is unclear, output it as ??? instead of guessing."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Extract the text from this page {i+1} of the PDF."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
            max_tokens=8000,
            temperature=0,
            timeout=120
        )
        text = (resp.choices[0].message.content or "").strip()
        text = _clean_text(text)

        # language check + counts
        lang_ok = keep_language(text, allowed_langs=ALLOWED_LANGS)
        words = len(text.split())

        pages_out.append({
            "page": i + 1,
            "text": text,
            "lang_mismatch": not lang_ok,
            "words": words,
        })
        success_pages += 1
        progress_ocr.progress(int((success_pages / total_pages) * 100))

    doc.close()
    status_ocr.write("✅ OCR complete")
    return pages_out

# ===============================
# Ingestion runner (append mode)
# ===============================


def run_ingestion(company: str, document_name: str, pdf_bytes: bytes,
                  progress_ocr, status_ocr,
                  progress_embed, status_embed,
                  progress_upload, status_upload):
    # Save uploaded PDF (organized by company)
    save_dir = Path("uploads") / company
    save_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = save_dir / document_name
    pdf_path.write_bytes(pdf_bytes)

    # Create doc hash for stable IDs
    doc_id = deterministic_doc_hash(pdf_path, pdf_bytes)
    
    # Capture upload time in ISO format for consistency
    

    # Get current time in that timezone
    
    upload_time = (datetime.datetime.utcnow() + datetime.timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S")


    # 1) OCR per page
    ocr_pages = ocr_pdf_with_deka(
        pdf_path, company, document_name, progress_ocr, status_ocr)

    # 2) Build chunks (here: 1 chunk per page + header as you do)
    chunks = []
    for page_info in ocr_pages:
        t = page_info["text"]
        if not t:
            continue
        header = build_meta_header(
            {"company": company, "source": document_name, "page": page_info["page"]})
        full_text = (header + t).strip()

        chunks.append({
            "id_raw": f"{doc_id}:{page_info['page']}",
            "text": full_text,
            "meta": {
                "company": company,
                "source": document_name,
                "page": page_info["page"],
                # "path": str(pdf_path.resolve()),
                "doc_id": doc_id,
                "words": page_info["words"],
                "lang_mismatch": page_info["lang_mismatch"],
                "upload_time": upload_time,
            }
        })

    # 3) Embeddings
    status_embed.write(f"🔎 Building embeddings for {len(chunks)} chunks")
    embedder = build_embedder()
    # detect dim
    dim = len(embedder.embed_query("hello world"))

    # Ensure collection exists + indexes (append mode)
    ensure_collection_and_indexes(dim)

    vectors = []
    ids = []
    payloads = []

    total = len(chunks)
    done = 0
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs = embedder.embed_documents(texts)
        vectors.extend(vecs)

        for c in batch:
            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, c["id_raw"]))
            ids.append(pid)
            payloads.append({
                "content": c["text"],
                "metadata": c["meta"]
            })

        done += len(batch)
        progress_embed.progress(int((done / total) * 100))

    status_embed.write("✅ Embedding complete")

    # 4) Upsert to Qdrant (append)
    status_upload.write(
        f"☁️ Uploading {len(ids)} points to Qdrant (append mode)")
    n = len(ids)
    uploaded = 0
    for i in range(0, n, BATCH_SIZE):
        pts = [
            rest.PointStruct(
                id=ids[j],
                vector=vectors[j],
                payload=payloads[j]
            )
            for j in range(i, min(i + BATCH_SIZE, n))
        ]
        client.upsert(collection_name=QDRANT_COLLECTION, points=pts, wait=True)
        uploaded += len(pts)
        progress_upload.progress(int((uploaded / n) * 100))

    status_upload.write("✅ Upload complete")
    
    # aDD TO SUPABASE
    add_to_supabase(company, document_name)
    
    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "uploaded": len(ids),
        "collection": QDRANT_COLLECTION,
        # ✅ return all extracted text
        "chunk_texts": [c["text"] for c in chunks]
    }


# ===============================
# OCR Chunk Review Helper
# ===============================

def review_ocr_chunks(chunks, status_review):
    """
    Display OCR chunks to user for review and editing.
    Returns the reviewed chunks or None if cancelled.
    """
    status_review.write(f"📝 Review {len(chunks)} OCR chunks before embedding")
    
    reviewed_chunks = []
    edited_count = 0
    
    # Create a container for the review UI
    review_container = st.container()
    
    with review_container:
        st.subheader(f"Review OCR Chunks ({len(chunks)} pages)")
        st.caption("You can edit the text in each chunk. Click 'Proceed with Embedding' when done.")
        
        # Store edited chunks in session state to persist across reruns
        if "reviewed_chunks" not in st.session_state:
            st.session_state.reviewed_chunks = chunks.copy()
            
        # Display each chunk in an expander
        for i, chunk in enumerate(st.session_state.reviewed_chunks):
            with st.expander(f"📄 Page {chunk['meta']['page']} ({chunk['meta']['words']} words)", expanded=(i==0)):
                # Display metadata
                cols = st.columns(4)
                cols[0].write(f"**Page:** {chunk['meta']['page']}")
                cols[1].write(f"**Words:** {chunk['meta']['words']}")
                cols[2].write(f"**Language OK:** {'✅' if not chunk['meta']['lang_mismatch'] else '❌'}")
                cols[3].write(f"**Doc ID:** {chunk['meta']['doc_id'][:8]}...")
                
                # Editable text area for the chunk content
                edited_text = st.text_area(
                    "Content (editable)",
                    value=chunk["text"],
                    height=300,
                    key=f"chunk_edit_{i}",
                    help="Edit the OCR text if needed. Changes will be preserved for embedding."
                )
                
                # Update the chunk in session state if edited
                if edited_text != chunk["text"]:
                    st.session_state.reviewed_chunks[i]["text"] = edited_text
                    st.session_state.reviewed_chunks[i]["meta"]["words"] = len(edited_text.split())
                    edited_count += 1
        
        # Summary and action buttons
        st.divider()
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"📝 Edited {edited_count} chunk{'s' if edited_count != 1 else ''}")
            
        with col2:
            if st.button("↩️ Reset All Edits"):
                st.session_state.reviewed_chunks = chunks.copy()
                st.success("All edits reset!")
                time.sleep(1)
                st.rerun()
                
        with col3:
            proceed = st.button("✅ Insert to Database", type="primary")
            cancel = st.button("❌ Cancel Ingestion")
            
        if proceed:
            # Clean up session state
            reviewed_chunks = st.session_state.reviewed_chunks.copy()
            if "reviewed_chunks" in st.session_state:
                del st.session_state.reviewed_chunks
            return reviewed_chunks
            
        if cancel:
            # Clean up session state
            if "reviewed_chunks" in st.session_state:
                del st.session_state.reviewed_chunks
            return None
    
# ===============================
# OCR Chunk Review Helper
# ===============================

def display_ocr_review_ui():
    """Display the OCR review UI if there are chunks in session state"""
    if "ocr_chunks_for_review" not in st.session_state:
        return False
    
    chunks = st.session_state.ocr_chunks_for_review
    if not chunks:
        return False
        
    st.subheader(f"🔍 Review OCR Chunks ({len(chunks)} pages)")
    st.caption("Review and edit the OCR results before proceeding with embedding.")
    
    edited_count = 0
    
    # Display each chunk in an expander
    for i, chunk in enumerate(chunks):
        with st.expander(f"📄 Page {chunk['meta']['page']} ({chunk['meta']['words']} words)", expanded=(i==0)):
            # Display metadata
            cols = st.columns(4)
            cols[0].write(f"**Page:** {chunk['meta']['page']}")
            cols[1].write(f"**Words:** {chunk['meta']['words']}")
            cols[2].write(f"**Language OK:** {'✅' if not chunk['meta']['lang_mismatch'] else '❌'}")
            cols[3].write(f"**Doc ID:** {chunk['meta']['doc_id'][:8]}...")
            
            # Editable text area for the chunk content
            edited_text = st.text_area(
                "Content (editable)",
                value=chunk["text"],
                height=300,
                key=f"chunk_edit_{i}",
                help="Edit the OCR text if needed. Changes will be preserved for embedding."
            )
            
            # Update the chunk in session state if edited
            if edited_text != chunk["text"]:
                st.session_state.ocr_chunks_for_review[i]["text"] = edited_text
                st.session_state.ocr_chunks_for_review[i]["meta"]["words"] = len(edited_text.split())
                edited_count += 1
    
    # Summary and action buttons (outside the form to avoid conflicts)
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"📝 Edited {edited_count} chunk{'s' if edited_count != 1 else ''}")
        
    with col2:
        if st.button("↩️ Reset All Edits", key="reset_edits"):
            # Reset to original OCR results
            st.session_state.ocr_chunks_for_review = st.session_state.original_ocr_chunks.copy()
            st.success("All edits reset!")
            time.sleep(1)
            st.rerun()
            
    with col3:
        # Use st.button outside of any form to avoid conflicts
        if st.button("✅ Proceed with Embedding", key="proceed_embedding"):
            # Clean up session state and proceed
            reviewed_chunks = st.session_state.ocr_chunks_for_review.copy()
            if "ocr_chunks_for_review" in st.session_state:
                del st.session_state.ocr_chunks_for_review
            if "original_ocr_chunks" in st.session_state:
                del st.session_state.original_ocr_chunks
            if "awaiting_review" in st.session_state:
                del st.session_state.awaiting_review
                
            # Store the reviewed chunks for the next step
            st.session_state.reviewed_chunks = reviewed_chunks
            st.session_state.ready_for_embedding = True
            st.rerun()
        
        if st.button("❌ Cancel Ingestion", key="cancel_ingestion"):
            # Clean up session state
            for key in ["ocr_chunks_for_review", "original_ocr_chunks", "awaiting_review", "reviewed_chunks", "ready_for_embedding"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.warning("Ingestion cancelled by user.")
            time.sleep(1)
            st.rerun()
        
    return True

def format_datetime(dt_str):
    """Format datetime string for display"""
    if dt_str == "Unknown Time":
        return dt_str
    try:
        # Try to parse ISO format
        dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        # Return as is if parsing fails
        return dt_str

# ===============================
# OCR Chunk Review Helper
# ===============================

def display_ocr_review_ui():
    """Display the OCR review UI if there are chunks in session state"""
    if "ocr_chunks_for_review" not in st.session_state:
        return False
    
    chunks = st.session_state.ocr_chunks_for_review
    st.subheader(f"🔍 Review OCR Chunks ({len(chunks)} pages)")
    st.caption("Review and edit the OCR results before proceeding with embedding.")
    
    edited_count = 0
    
    # Display each chunk in an expander
    for i, chunk in enumerate(chunks):
        with st.expander(f"📄 Page {chunk['meta']['page']} ({chunk['meta']['words']} words)", expanded=(i==0)):
            # Display metadata
            cols = st.columns(4)
            cols[0].write(f"**Page:** {chunk['meta']['page']}")
            cols[1].write(f"**Words:** {chunk['meta']['words']}")
            cols[2].write(f"**Language OK:** {'✅' if not chunk['meta']['lang_mismatch'] else '❌'}")
            cols[3].write(f"**Doc ID:** {chunk['meta']['doc_id'][:8]}...")
            
            # Editable text area for the chunk content
            edited_text = st.text_area(
                "Content (editable)",
                value=chunk["text"],
                height=300,
                key=f"chunk_edit_{i}",
                help="Edit the OCR text if needed. Changes will be preserved for embedding."
            )
            
            # Update the chunk in session state if edited
            if edited_text != chunk["text"]:
                st.session_state.ocr_chunks_for_review[i]["text"] = edited_text
                st.session_state.ocr_chunks_for_review[i]["meta"]["words"] = len(edited_text.split())
                edited_count += 1
    
    # Summary and action buttons
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"📝 Edited {edited_count} chunk{'s' if edited_count != 1 else ''}")
        
    with col2:
        if st.button("↩️ Reset All Edits", key="reset_edits"):
            # Reset to original OCR results
            st.session_state.ocr_chunks_for_review = st.session_state.original_ocr_chunks.copy()
            st.success("All edits reset!")
            time.sleep(1)
            st.rerun()
            
    with col3:
        if st.button("✅ Insert to Database", type="primary", key="proceed_embedding"):
            # Clean up session state and proceed
            reviewed_chunks = st.session_state.ocr_chunks_for_review.copy()
            if "ocr_chunks_for_review" in st.session_state:
                del st.session_state.ocr_chunks_for_review
            if "original_ocr_chunks" in st.session_state:
                del st.session_state.original_ocr_chunks
            if "awaiting_review" in st.session_state:
                del st.session_state.awaiting_review
                
            # Store the reviewed chunks for the next step
            st.session_state.reviewed_chunks = reviewed_chunks
            st.session_state.ready_for_embedding = True
            st.rerun()
            
        if st.button("❌ Cancel Ingestion", key="cancel_ingestion"):
            # Clean up session state
            for key in ["ocr_chunks_for_review", "original_ocr_chunks", "awaiting_review", "reviewed_chunks", "ready_for_embedding"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.warning("Ingestion cancelled by user.")
            time.sleep(1)
            st.rerun()
        
    return True


def list_documents(limit: int = 1000):
    pts, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )
    docs = {}
    for p in pts or []:
        meta = (p.payload or {}).get("metadata", {})
        source = meta.get("source", "Unknown Source")
        comp = meta.get("company", "Unknown Company")
        doc_id = meta.get("doc_id", "-")
        # Handle missing upload_time more gracefully
        upload_time = meta.get("upload_time")
        if not upload_time:
            # Try to get a creation timestamp from Qdrant if available
            upload_time = "Unknown Time"
        if source not in docs:
            docs[source] = {"company": comp, "doc_id": doc_id, "chunks": 0, "upload_time": upload_time}
        docs[source]["chunks"] += 1
    return docs


def delete_document_by_source(source_name: str):
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=rest.FilterSelector(
            filter=rest.Filter(
                must=[rest.FieldCondition(
                    key="metadata.source", match=rest.MatchValue(value=source_name))]
            )
        )
    )
    delete_from_supabase(source_name)


# ===============================
# AUTHENTICATION
# ===============================
def is_authorized_domain(email):
    """Check if the user's email domain is in the authorized domains list."""
    if not AUTHORIZED_DOMAINS:
        return True  # If no domains specified, allow all
    
    authorized_domains = [domain.strip() for domain in AUTHORIZED_DOMAINS.split(",")]
    user_domain = email.split("@")[1] if "@" in email else ""
    return user_domain in authorized_domains

def is_authorized_account(email):
    """Check if the user's email is in the authorized accounts list."""
    # You can set AUTHORIZED_ACCOUNTS as a comma-separated list of emails in environment variables
    authorized_accounts = os.getenv("AUTHORIZED_ACCOUNTS", "")
    if not authorized_accounts:
        return True  # If no accounts specified, allow all (when domains are restricted)
    
    authorized_emails = [email.strip() for email in authorized_accounts.split(",")]
    return email in authorized_emails

def check_password():
    """Returns `True` if the user has entered the correct email and password."""
    
    # Return True if the user is already authenticated with password
    if st.session_state.get("password_correct", False):
        return True

    # Show input for email and password.
    st.title("🔐 Password Required")
    st.caption("You've successfully signed in with Google. Please enter your application password to access the RAG application.")
    
    # Callback function to check credentials
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (hmac.compare_digest(st.session_state["email"], AUTH_EMAIL) and
                hmac.compare_digest(st.session_state["password"], AUTH_PASSWORD)):
            st.session_state["password_correct"] = True
            st.success("Password authentication successful!")
        else:
            st.session_state["password_correct"] = False
            st.error("😕 Email or password incorrect")

    # Input fields for email and password
    st.text_input("Email", key="email", autocomplete="email")
    st.text_input("Password", type="password", key="password")
    st.button("Authenticate", on_click=password_entered)
    
    # Always return False when showing the login form
    # This allows the form to be displayed without calling st.stop()
    return False

def check_authentication():
    """Handle Google OAuth authentication, domain restrictions, and password authentication."""
    # Check if user is logged in with Google
    if st.user.is_logged_in:
        # Check domain restrictions if configured
        user_email = getattr(st.user, 'email', '')
        if user_email and not is_authorized_domain(user_email):
            st.error(f"Access denied. Only users from authorized domains can access this application.")
            st.button("Logout", on_click=st.logout)
            st.stop()
        
        # Check account restrictions if configured
        if user_email and not is_authorized_account(user_email):
            st.error(f"Access denied. Your account is not authorized to access this application.")
            st.button("Logout", on_click=st.logout)
            st.stop()
        
        # User is authenticated with Google and authorized
        # Now check password authentication
        if not AUTH_EMAIL or not AUTH_PASSWORD:
            st.warning("⚠️ Password authentication not configured. Set AUTH_EMAIL and AUTH_PASSWORD environment variables to enable password protection.")
            return True
        else:
            return check_password()
    else:
        # Show Google login screen
        st.title("🔐 Login")
        st.caption("Please sign in with your Google account to access the RAG application.")
        
        # Show domain restriction info if configured
        if AUTHORIZED_DOMAINS:
            domains = ", ".join([domain.strip() for domain in AUTHORIZED_DOMAINS.split(",")])
            st.info(f"Note: Only users from the following domains are authorized: {domains}")
            
        # Show account restriction info if configured
        if os.getenv("AUTHORIZED_ACCOUNTS"):
            st.info("Note: Only specific accounts are authorized to access this application.")
        
        # Google login button
        if st.button("Sign in with Google"):
            st.login()
        
        return False

# Check authentication
auth_result = check_authentication()
if not auth_result:
    st.stop()

# ===============================
# UI
# ===============================
# Show user info and logout button
col1, col2 = st.columns([4, 1])
with col1:
    st.title("📚 BRA Team Contract Document Handling")
with col2:
    if st.user.is_logged_in:
        user_name = getattr(st.user, 'name', 'User')
        user_email = getattr(st.user, 'email', '')
        st.markdown(f"**{user_name}**")
        st.caption(user_email)
        if st.button("Logout"):
            # Clear password authentication state when logging out
            if "password_correct" in st.session_state:
                del st.session_state.password_correct
            st.logout()

st.caption(f"Collection: `{QDRANT_COLLECTION}` · Qdrant: {QDRANT_URL}")

# Check for successful ingestion and display success message
if st.session_state.get("ingestion_success", False):
    doc_id = st.session_state.get("success_doc_id", "")
    collection = st.session_state.get("success_collection", "")
    chunks = st.session_state.get("success_chunks", 0)
    
    st.success(f"✅ Document successfully ingested! {chunks} chunks upserted to `{collection}` (doc_id={doc_id}…)")
    
    # Clear the success flags
    del st.session_state.ingestion_success
    if "success_doc_id" in st.session_state:
        del st.session_state.success_doc_id
    if "success_collection" in st.session_state:
        del st.session_state.success_collection
    if "success_chunks" in st.session_state:
        del st.session_state.success_chunks

# ===============================
# 📚 Unified Vertical Layout
# ===============================

st.subheader("➕ Ingest New PDF")
# Use session state to manage form inputs
if "company_input" not in st.session_state:
    st.session_state.company_input = ""
if "docname_input" not in st.session_state:
    st.session_state.docname_input = ""

with st.form("ingest_form", clear_on_submit=True):
    company = st.text_input(
        "🏢 Company Name", 
        value=st.session_state.company_input,
        placeholder="e.g., PT Lintasarta",
        key="company_input_field")
    docname = st.text_input("📄 Document Name (filename)",
                            value=st.session_state.docname_input,
                            placeholder="e.g., Contract_ABC.pdf",
                            key="docname_input_field")
    uploaded = st.file_uploader("📎 Upload PDF", type=["pdf"], key="pdf_uploader")
    go = st.form_submit_button("🚀 Ingest")

# Check if we're ready for embedding (after review) - moved outside form handler
if st.session_state.get("ready_for_embedding", False):
    # Get the reviewed chunks and proceed with embedding
    reviewed_chunks = st.session_state.get("reviewed_chunks", [])
    company = st.session_state.get("stored_company", "")
    docname = st.session_state.get("stored_docname", "")
    
    if "reviewed_chunks" in st.session_state:
        del st.session_state.reviewed_chunks
    if "ready_for_embedding" in st.session_state:
        del st.session_state.ready_for_embedding
        if "stored_company" in st.session_state:
            del st.session_state.stored_company
        if "stored_docname" in st.session_state:
            del st.session_state.stored_docname
        
    # Show loading spinner during processing
    with st.spinner("Processing document..."):
        st.info("Starting embedding and upload... (append mode)")

        # Progress sections
        with st.expander("🧠 Embedding Progress", expanded=True):
            progress_embed = st.progress(0)
            status_embed = st.empty()

        with st.expander("☁️ Upload Progress", expanded=True):
            progress_upload = st.progress(0)
            status_upload = st.empty()

        try:
            # Create a temporary function to handle the embedding/upload part
            def run_embedding_and_upload(chunks):
                # This is a simplified version of the embedding/upload process
                # Build embeddings
                status_embed.write(f"🔎 Building embeddings for {len(chunks)} chunks")
                embedder = build_embedder()
                # detect dim
                dim = len(embedder.embed_query("hello world"))

                # Ensure collection exists + indexes (append mode)
                ensure_collection_and_indexes(dim)

                vectors = []
                ids = []
                payloads = []

                total = len(chunks)
                done = 0
                for i in range(0, total, BATCH_SIZE):
                    batch = chunks[i:i + BATCH_SIZE]
                    texts = [c["text"] for c in batch]
                    vecs = embedder.embed_documents(texts)
                    vectors.extend(vecs)

                    for c in batch:
                        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, c["id_raw"]))
                        ids.append(pid)
                        payloads.append({
                            "content": c["text"],
                            "metadata": c["meta"]
                        })

                    done += len(batch)
                    progress_embed.progress(int((done / total) * 100))

                status_embed.write("✅ Embedding complete")

                # 4) Upsert to Qdrant (append)
                status_upload.write(
                    f"☁️ Uploading {len(ids)} points to Qdrant (append mode)")
                n = len(ids)
                uploaded_count = 0
                for i in range(0, n, BATCH_SIZE):
                    pts = [
                        rest.PointStruct(
                            id=ids[j],
                            vector=vectors[j],
                            payload=payloads[j]
                        )
                        for j in range(i, min(i + BATCH_SIZE, n))
                    ]
                    client.upsert(collection_name=QDRANT_COLLECTION, points=pts, wait=True)
                    uploaded_count += len(pts)
                    progress_upload.progress(int((uploaded_count / n) * 100))

                status_upload.write("✅ Upload complete")
                
                # Add to Supabase
                add_to_supabase(company, docname)
                
                return {
                    "doc_id": chunks[0]["meta"]["doc_id"] if chunks else "unknown",
                    "chunks": len(chunks),
                    "uploaded": len(ids),
                    "collection": QDRANT_COLLECTION,
                }
            
            result = run_embedding_and_upload(reviewed_chunks)

            # Clear the form inputs after successful ingestion
            st.session_state.company_input = ""
            st.session_state.docname_input = ""
            
            st.success(
                f"✅ Done! {result['uploaded']} chunks upserted to `{result['collection']}` "
                f"(doc_id={result['doc_id'][:8]}…)."
            )
            
            # Set success flag in session state
            st.session_state.ingestion_success = True
            st.session_state.success_doc_id = result['doc_id'][:8]
            st.session_state.success_collection = result['collection']
            st.session_state.success_chunks = result['uploaded']
            
            # Force a rerun to refresh the document list
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"🚫 Ingestion failed: {e}")
            st.error(f"Error details: {str(e)}")

# Handle form submission
if go:
    if not company or not docname or not uploaded:
        st.warning("⚠️ Please fill all fields and upload a PDF.")
    else:
        # Store form values in session state for later use
        st.session_state.stored_company = company
        st.session_state.stored_docname = docname
        
        # Show loading spinner during OCR processing
        with st.spinner("Processing document..."):
            st.info("Starting OCR processing…")

            # Progress sections
            with st.expander("🔎 OCR Progress", expanded=True):
                progress_ocr = st.progress(0)
                status_ocr = st.empty()

            try:
                # Save uploaded PDF (organized by company)
                save_dir = Path("uploads") / company
                save_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = save_dir / docname
                pdf_path.write_bytes(uploaded.getvalue())

                # Create doc hash for stable IDs
                doc_id = deterministic_doc_hash(pdf_path, uploaded.getvalue())
                
                # Capture upload time in ISO format for consistency
                upload_time = datetime.datetime.now().isoformat()

                # 1) OCR per page
                ocr_pages = ocr_pdf_with_deka(
                    pdf_path, company, docname, progress_ocr, status_ocr)

                # 2) Build chunks (here: 1 chunk per page + header as you do)
                chunks = []
                for page_info in ocr_pages:
                    t = page_info["text"]
                    if not t:
                        continue
                    header = build_meta_header(
                        {"company": company, "source": docname, "page": page_info["page"]})
                    full_text = (header + t).strip()

                    chunks.append({
                        "id_raw": f"{doc_id}:{page_info['page']}",
                        "text": full_text,
                        "meta": {
                            "company": company,
                            "source": docname,
                            "page": page_info["page"],
                            "path": str(pdf_path.resolve()),
                            "doc_id": doc_id,
                            "words": page_info["words"],
                            "lang_mismatch": page_info["lang_mismatch"],
                            "upload_time": upload_time,
                        }
                    })

                # Store chunks in session state for review
                st.session_state.ocr_chunks_for_review = chunks
                st.session_state.original_ocr_chunks = chunks.copy()
                st.session_state.awaiting_review = True
                
                st.info("OCR complete. Please review the chunks below before proceeding.")
                time.sleep(1)
                st.rerun()

            except Exception as e:
                st.error(f"🚫 OCR processing failed: {e}")
                st.error(f"Error details: {str(e)}")


# Display OCR review UI if there are chunks to review
display_ocr_review_ui()

st.markdown("---")
st.subheader("📄 Documents Stored in Qdrant")

# Add auto-refresh toggle
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Refresh Document List"):
        st.rerun()
with col2:
    st.session_state.auto_refresh = st.checkbox("Auto-refresh every 30 seconds", st.session_state.auto_refresh)

docs = list_documents(limit=1000)
st.write(f"Found **{len(docs)}** documents")

if docs:
    # Initialize session state for document selections if not exists
    if "selected_documents" not in st.session_state:
        st.session_state.selected_documents = set()
    
    # Convert docs to a list of dictionaries for the dataframe
    docs_list = []
    for k, v in docs.items():
        is_selected = k in st.session_state.selected_documents
        docs_list.append({
            "Select": is_selected,
            "Source": k, 
            "Company": v["company"], 
            "Doc ID": v["doc_id"], 
            "Chunks": v["chunks"], 
            "Upload Time": format_datetime(v["upload_time"])
        })
    
    # Show documents in a dataframe with selection checkboxes
    st.write("**Select documents for deletion:**")
    
    # Display the dataframe in a scrollable container
    df_container = st.container(height=400)
    with df_container:
        edited_df = st.data_editor(
            docs_list,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select documents for deletion",
                    default=False,
                )
            },
            disabled=["Source", "Company", "Doc ID", "Chunks", "Upload Time"],
            num_rows="fixed",
            key="documents_table"
        )
    
    # Automatically update session state with current selections
    current_selections = {row["Source"] for row in edited_df if row["Select"]}
    st.session_state.selected_documents = current_selections
    
    # Show currently selected documents
    if st.session_state.selected_documents:
        st.write(f"Selected {len(st.session_state.selected_documents)} document(s) for deletion:")
        
        # Show selected documents (limit to first 10 for space)
        selected_list = list(st.session_state.selected_documents)
        for doc_source in selected_list[:10]:
            st.write(f"- {doc_source}")
        if len(selected_list) > 10:
            st.write(f"... and {len(selected_list) - 10} more")
        
        st.markdown("---")
        if st.button("🗑️ Delete Selected Documents", type="primary"):
            try:
                deleted_count = 0
                for doc_source in list(st.session_state.selected_documents):
                    delete_document_by_source(doc_source)
                    deleted_count += 1
                
                # Clear selections after deletion
                st.session_state.selected_documents.clear()
                # Also reset the dataframe state
                if "docs_df_state" in st.session_state:
                    del st.session_state.docs_df_state
                
                st.success(
                    f"✅ Deleted all chunks for {deleted_count} document(s). Refreshing list…")
                time.sleep(1.0)
                st.rerun()
            except Exception as e:
                st.error(f"Deletion failed: {e}")
    else:
        st.info("Check boxes in the table to select documents for deletion.")
else:
    st.info("No points yet. Ingest a PDF above to start populating.")

# Auto-refresh every 30 seconds if enabled
if st.session_state.auto_refresh:
    time.sleep(30)
    st.rerun()
