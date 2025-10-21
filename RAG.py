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

# TOTP MFA Libraries
import pyotp
import qrcode
from PIL import Image

# OCR / PDF
import fitz  # PyMuPDF
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

load_dotenv()  # auto-load variables from .env

# ===============================
# ENV / CONFIG
# ===============================
st.set_page_config(page_title="BRA Team Contract Document Handling",
                   page_icon="📚", layout="wide")

# Authentication Configuration (Multi-User Support)
# Load multiple user credentials from environment variables
def load_user_credentials():
    """Load user credentials from environment variables"""
    users = {}
    
    # Check for multiple users (USER1_EMAIL, USER2_EMAIL, etc.)
    i = 1
    while True:
        email_key = f"USER{i}_EMAIL"
        password_key = f"USER{i}_PASSWORD"
        
        email = os.getenv(email_key)
        password = os.getenv(password_key)
        
        if email and password:
            users[email] = password
            i += 1
        else:
            break
    
    # Fallback to single user if no multi-users defined
    if not users:
        auth_email = os.getenv("AUTH_EMAIL")
        auth_password = os.getenv("AUTH_PASSWORD")
        if auth_email and auth_password:
            users[auth_email] = auth_password
    
    return users

# Load user credentials
USER_CREDENTIALS = load_user_credentials()

# Qdrant Configuration
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
SUPABASE_URL = os.getenv("SUPABASE_URL")
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
# TOTP MFA FUNCTIONS
# ===============================

# Simple file-based storage for TOTP secrets (in production, use a database)
SECRETS_FILE = "user_secrets.json"

def load_totp_secrets():
    """Load TOTP secrets from file"""
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_totp_secrets(secrets):
    """Save TOTP secrets to file"""
    with open(SECRETS_FILE, 'w') as f:
        json.dump(secrets, f, indent=2)

def generate_totp_secret():
    """Generate a new TOTP secret"""
    return pyotp.random_base32()

def generate_qr_code(provisioning_uri):
    """Generate QR code for TOTP setup"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(provisioning_uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes for Streamlit
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def verify_totp(secret, token):
    """Verify TOTP token"""
    totp = pyotp.TOTP(secret)
    return totp.verify(token)

# ===============================
# AUTHENTICATION
# ===============================

def check_password():
    """Returns `True` if the user has entered the correct email and password."""
    
    # Return True if the user is already authenticated with password
    if st.session_state.get("password_correct", False):
        return True

    # Show input for email and password.
    st.title("🔐 Login")
    st.caption("Please enter your email and password to access the application.")
    
    # Callback function to check credentials
    def password_entered():
        """Checks whether email and password are correct."""
        email = st.session_state["email"]
        password = st.session_state["password"]
        
        # Check if user exists and password matches
        if email in USER_CREDENTIALS and hmac.compare_digest(password, USER_CREDENTIALS[email]):
            st.session_state["password_correct"] = True
            st.session_state["user_email"] = email
            st.success("Password authentication successful!")
        else:
            st.session_state["password_correct"] = False
            st.error("😕 Email or password incorrect")

    # Input fields for email and password
    st.text_input("Email", key="email", autocomplete="email")
    st.text_input("Password", type="password", key="password")
    st.button("Authenticate", on_click=password_entered)
    
    # Always return False when showing the login form
    return False

def setup_totp():
    """Setup TOTP MFA for the user"""
    st.title("📱 Multi-Factor Authentication Setup")
    st.caption("Set up your authenticator app for enhanced security.")
    
    user_email = st.session_state.get("user_email", "")
    if not user_email:
        st.error("Unable to get user email for MFA setup")
        return False
    
    # Load existing secrets
    secrets = load_totp_secrets()
    
    # Check if user already has MFA set up
    if user_email in secrets:
        st.info("MFA is already set up for your account.")
        st.session_state["totp_setup_complete"] = True
        return True
    
    st.info("To set up MFA, please scan the QR code below with your authenticator app (Google Authenticator, Authy, etc.)")
    
    # Generate a new secret for the user if not already in session
    if "totp_secret" not in st.session_state:
        secret = generate_totp_secret()
        st.session_state["totp_secret"] = secret
        # Save the secret temporarily
        secrets[user_email] = secret
        save_totp_secrets(secrets)
    
    secret = st.session_state["totp_secret"]
    
    # Generate provisioning URI and QR code
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user_email,
        issuer_name="RAG Application"
    )
    
    qr_code_bytes = generate_qr_code(provisioning_uri)
    
    # Display QR code
    st.image(qr_code_bytes, caption="Scan this QR code with your authenticator app", width=300)
    
    # Show secret key for manual entry
    st.info("If you can't scan the QR code, enter this key manually in your authenticator app:")
    st.code(secret, language=None)
    
    # Verification form
    st.subheader("Verify Setup")
    st.caption("Enter a 6-digit code from your authenticator app to complete setup.")
    
    with st.form("totp_verification"):
        verification_code = st.text_input("6-digit code", max_chars=6, key="verification_code")
        submitted = st.form_submit_button("Verify and Enable MFA")
        
        if submitted:
            if verification_code and len(verification_code) == 6 and verification_code.isdigit():
                if verify_totp(secret, verification_code):
                    st.session_state["totp_setup_complete"] = True
                    st.success("✅ MFA setup successful! You're now protected with two-factor authentication.")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Invalid code. Please make sure you're entering the current code from your authenticator app.")
            else:
                st.error("Please enter a valid 6-digit code.")
    
    return False

def check_totp():
    """Verify TOTP code for MFA"""
    user_email = st.session_state.get("user_email", "")
    if not user_email:
        st.error("Authentication error: User email not found")
        return False
    
    # Load secrets
    secrets = load_totp_secrets()
    
    # Check if user has completed setup but hasn't verified yet
    if st.session_state.get("totp_setup_complete", False) and not st.session_state.get("totp_verified", False):
        st.success("✅ MFA setup complete! Please enter a code from your authenticator app to verify.")
    
    # Check if user has MFA set up
    if user_email not in secrets:
        return setup_totp()
    
    # If already verified, return True
    if st.session_state.get("totp_verified", False):
        return True
    
    # Show TOTP verification form
    st.title("📱 Two-Factor Authentication")
    st.caption("Enter the 6-digit code from your authenticator app.")
    
    def verify_totp_code():
        """Verify the TOTP code entered by the user"""
        code = st.session_state.get("totp_code", "")
        if code and len(code) == 6 and code.isdigit():
            secret = secrets[user_email]
            if verify_totp(secret, code):
                st.session_state["totp_verified"] = True
                st.success("Authentication successful!")
            else:
                st.session_state["totp_error"] = "Invalid code. Please try again."
        else:
            st.session_state["totp_error"] = "Please enter a valid 6-digit code."
    
    # Input for TOTP code
    st.text_input("6-digit code", 
                  key="totp_code", 
                  max_chars=6,
                  placeholder="Enter code from your authenticator app",
                  on_change=verify_totp_code)
    
    if "totp_error" in st.session_state:
        st.error(st.session_state["totp_error"])
        del st.session_state["totp_error"]
    
    st.button("Verify", on_click=verify_totp_code)
    
    # Option to reset MFA
    st.caption("Having trouble with MFA?")
    if st.button("Reset MFA Setup"):
        if user_email in secrets:
            del secrets[user_email]
            save_totp_secrets(secrets)
        # Clear session state
        for key in ["totp_secret", "totp_setup_complete", "totp_verified"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    return False

def check_authentication():
    """Handle HMAC password authentication and TOTP MFA."""
    # First check password authentication
    password_auth_passed = check_password()
    
    # If password authentication passed, check TOTP MFA
    if password_auth_passed:
        return check_totp()
    
    return False

# ===============================
# OCR REVIEW UI FUNCTION
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

# ===============================
# OTHER HELPER FUNCTIONS
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
    st.markdown(f"**{st.session_state['user_email']}**")
    if st.button("Logout"):
        # Clear all authentication state when logging out
        for key in list(st.session_state.keys()):
            if key.startswith(("password_", "totp_", "user_")):
                del st.session_state[key]
        # Rerun to show login screen
        st.rerun()

st.caption(f"Collection: `{QDRANT_COLLECTION}` · Qdrant: {QDRANT_URL}")

# MFA Management Section
with st.expander("🔒 Multi-Factor Authentication Settings"):
    secrets = load_totp_secrets()
    user_email = st.session_state.get("user_email", "")
    
    if user_email in secrets:
        st.success("✅ MFA is enabled for your account")
        if st.button("Disable MFA"):
            if user_email in secrets:
                del secrets[user_email]
                save_totp_secrets(secrets)
                # Also clear session state
                for key in ["totp_secret", "totp_setup_complete", "totp_verified"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("MFA has been disabled for your account")
                time.sleep(1)
                st.rerun()
    else:
        st.warning("⚠️ MFA is not enabled for your account")
        if st.button("Enable MFA"):
            st.session_state["totp_setup_complete"] = False
            st.rerun()

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
    
    # Update session state with current selections only when delete button is pressed
    # For now, just show the count of currently selected items in the UI
    current_selections = {row["Source"] for row in edited_df if row["Select"]}
    selection_changed = current_selections != st.session_state.selected_documents
    
    if selection_changed:
        st.info(f"Selection updated: {len(current_selections)} document(s) selected. Click 'Delete Selected Documents' to proceed.")
    
    # Show currently selected documents (from the data editor, not session state)
    if current_selections:
        st.write(f"Selected {len(current_selections)} document(s) for deletion:")
        
        # Show selected documents (limit to first 10 for space)
        selected_list = list(current_selections)
        for doc_source in selected_list[:10]:
            st.write(f"- {doc_source}")
        if len(selected_list) > 10:
            st.write(f"... and {len(selected_list) - 10} more")
        
        st.markdown("---")
        if st.button("🗑️ Delete Selected Documents", type="primary"):
            try:
                # Update session state with current selections when delete is pressed
                st.session_state.selected_documents = current_selections
                
                deleted_count = 0
                for doc_source in list(current_selections):
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
        # Clear session state if no selections
        st.session_state.selected_documents.clear()
else:
    st.info("No points yet. Ingest a PDF above to start populating.")

# Auto-refresh every 30 seconds if enabled
if st.session_state.auto_refresh:
    time.sleep(30)
    st.rerun()