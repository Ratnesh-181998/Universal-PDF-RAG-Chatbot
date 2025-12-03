import streamlit as st
import os
import tempfile
import time
import logging
import datetime
import shutil
import html
from pathlib import Path
from typing import List
import textwrap

# Logging Setup
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message: str, level: str = "INFO"):
    """Log event to file and session state"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    if level == "INFO":
        logging.info(message)
    elif level == "WARNING":
        logging.warning(message)
    elif level == "ERROR":
        logging.error(message)
        
    if "app_logs" not in st.session_state:
        st.session_state["app_logs"] = []
    
    # Use HTML entities for emojis in logs to avoid syntax errors
    emoji = '&#128308;' if level=='ERROR' else ('&#9888;' if level=='WARNING' else '&#128313;')
    st.session_state["app_logs"].insert(0, f"{emoji} {log_entry}")
    
    if len(st.session_state["app_logs"]) > 100:
        st.session_state["app_logs"] = st.session_state["app_logs"][:100]

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Configuration
INDEX_DIR = "faiss_index_storage"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 10

GROQ_MODELS = [
    "llama-3.3-70b-versatile",     # Latest powerful model
    "llama-3.1-70b-versatile",     # Stable 70B
    "llama-3.1-8b-instant",        # Fast & efficient
    "mixtral-8x7b-32768",          # Long context
    "gemma2-9b-it",                # Google Gemma 2
    "llama3-70b-8192",             # Original Llama 3 70B
    "llama3-8b-8192",              # Original Llama 3 8B
]

# Page Configuration
st.set_page_config(
    page_title="Universal RAG Chatbot",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Yulu-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    .block-container {
        background: rgba(17, 24, 39, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #f472b6 50%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    h2 { 
        color: #f3f4f6 !important; 
        border-bottom: 3px solid #8b5cf6; 
        padding-bottom: 0.5rem; 
        font-weight: 700 !important;
    }
    h3 { color: #e5e7eb !important; font-weight: 600 !important; }
    p, li, span, div { color: #cbd5e1; }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 12px; 
        background-color: rgba(17, 24, 39, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(244, 114, 182, 0.1) 100%);
        color: #a78bfa; 
        border-radius: 10px; 
        padding: 12px 24px; 
        font-weight: 600; 
        transition: all 0.3s ease; 
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .stTabs [data-baseweb="tab"]:hover { 
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(244, 114, 182, 0.2) 100%);
        transform: translateY(-2px); 
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important; 
        color: white !important; 
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
    }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); 
        border-right: 1px solid rgba(139, 92, 246, 0.3);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { 
        color: white !important; 
        -webkit-text-fill-color: white !important; 
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); 
        color: white; 
        border-radius: 12px; 
        padding: 0.75rem 2rem; 
        font-weight: 600; 
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4); 
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6); 
    }
    
    .card {
        padding: 1rem; 
        border-radius: 16px; 
        text-align: center; 
        color: white; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.4); 
        transition: all 0.4s ease; 
        border: 1px solid rgba(255,255,255,0.1);
    }
    .card:hover { 
        transform: translateY(-8px) scale(1.02); 
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.4);
    }
    
    @keyframes fadeInDown { 
        from { opacity: 0; transform: translateY(-30px); } 
        to { opacity: 1; transform: translateY(0); } 
    }
    
    .stExpander {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(17, 24, 39, 0.6) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style='position: fixed; top: 3.5rem; right: 1.5rem; z-index: 9999;'>
    <div style='background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%); 
                border-radius: 20px; padding: 0.6rem 1.2rem; 
                box-shadow: 0 4px 20px rgba(139, 92, 246, 0.5);'>
        <span style='color: white; font-weight: 700; font-size: 0.9rem; letter-spacing: 1.5px;'>
            &#10024; By RATNESH SINGH(Data Scientist)
        </span>
    </div>
</div>
<div style='text-align: center; padding: 2rem 0 1rem 0;'>
    <h1 style='font-size: 4rem; margin-bottom: 0;'>&#128218; Universal RAG Chatbot</h1>
    <p style='font-size: 1.3rem; background: linear-gradient(135deg, #a78bfa 0%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; margin-top: 0.5rem;'>
        &#127919; AI-Powered Document Q&A System
    </p>
</div>
""", unsafe_allow_html=True)

# How to Use Workflow
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(244, 114, 182, 0.1) 100%); 
            padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h3 style='text-align: center; margin-top: 0; color: #a78bfa !important; font-size: 1.2rem;'>&#128640; Quick Guide</h3>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; text-align: center;'>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128194;</div>
            <div style='font-weight: 600; color: #fff;'>1. Upload PDF</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Use sidebar/uploader</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#9203;</div>
            <div style='font-weight: 600; color: #fff;'>2. Wait 20-30s</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Processing docs</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128172;</div>
            <div style='font-weight: 600; color: #fff;'>3. Ask Question</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>In Chat tab</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#129302;</div>
            <div style='font-weight: 600; color: #fff;'>4. Wait 20-30s</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>AI generating</div>
        </div>
        <div style='font-size: 1.2rem; color: #666;'>➔</div>
        <div style='flex: 1; min-width: 120px;'>
            <div style='font-size: 1.5rem; margin-bottom: 5px;'>&#128218;</div>
            <div style='font-weight: 600; color: #fff;'>5. View Sources</div>
            <div style='font-size: 0.8rem; color: #cbd5e1;'>Check citations</div>
        </div>
    </div>
    <div style='margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: #a78bfa;'>
        &#128161; <strong>Tip:</strong> If answers aren't found, click <strong>"&#128296; Rebuild Index"</strong> in sidebar & re-upload.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## &#9881; Settings")
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(139, 92, 246, 0.3);'>
        <h3 style='color: #a78bfa !important; margin-top: 0;'>&#128202; Configuration</h3>
        <p><strong>Embedding:</strong> all-mpnet-base-v2</p>
        <p><strong>Chunk Size:</strong> 800</p>
        <p><strong>Top-K:</strong> 10</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Check API keys
    has_groq = bool(os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None))
    has_openai = bool(os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None))
    
    use_groq = st.checkbox("Prefer Groq LLM", value=GROQ_AVAILABLE and has_groq)
    force_rebuild = st.button("&#128296; Rebuild Index", help="Force rebuild FAISS index")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style='background: rgba(56, 239, 125, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(56, 239, 125, 0.3);'>
        <h3 style='color: #38ef7d !important; margin-top: 0;'>&#8505; Status</h3>
        <p>{'&#9989;' if GROQ_AVAILABLE and has_groq else '&#10060;'} Groq</p>
        <p>{'&#9989;' if OPENAI_AVAILABLE and has_openai else '&#10060;'} OpenAI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Download Chat History
    if "messages" in st.session_state and st.session_state["messages"]:
        chat_history_text = f"Chat History - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        chat_history_text += "="*50 + "\n\n"
        
        for role, content in st.session_state["messages"]:
            # Simple cleanup of HTML tags for better readability in text file
            clean_content = content.replace("<b>", "**").replace("</b>", "**")
            # Remove the hidden debug chunks for the text file
            if "|||DEBUG_CHUNKS" in clean_content:
                clean_content = clean_content.split("|||DEBUG_CHUNKS")[0]
            if "|||FOOTER" in clean_content:
                clean_content = clean_content.split("|||FOOTER")[0]
                
            chat_history_text += f"[{role.upper()}]:\n{clean_content.strip()}\n\n{'-'*50}\n\n"
            
        st.download_button(
            label="&#128190; Download Chat History",
            data=chat_history_text,
            file_name=f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    st.markdown("**Build By RATNESH SINGH(Data Scientist)**")

# Helper Functions
def save_uploaded_files(uploaded_files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(dest_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_documents_langchain(dir_path: str) -> List[Document]:
    docs = []
    pdf_files = list(Path(dir_path).glob("*.pdf"))
    
    from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
    
    for pdf_path in pdf_files:
        try:
            # Try PyMuPDF first (fast for text-based PDFs)
            loader = PyMuPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Check if we got meaningful text
            total_text = "".join([p.page_content for p in pages])
            
            # If very little text extracted, use OCR mode
            if len(total_text.strip()) < 100:
                log_event(f"Low text content detected in {pdf_path.name}, using OCR mode...")
                try:
                    # Use Unstructured with OCR for image-based PDFs
                    loader = UnstructuredPDFLoader(
                        str(pdf_path),
                        mode="elements",
                        strategy="hi_res"  # High resolution for better OCR
                    )
                    pages = loader.load()
                    log_event(f"OCR extraction completed for {pdf_path.name}")
                except Exception as ocr_error:
                    log_event(f"OCR failed for {pdf_path.name}: {ocr_error}", "WARNING")
                    # Continue with whatever we got from PyMuPDF
            
            # Log sample text to verify extraction
            if pages:
                sample_text = pages[0].page_content[:500].replace('\n', ' ')
                log_event(f"Extracted text sample from {pdf_path.name}: {sample_text}...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(pages)
            log_event(f"Created {len(chunks)} chunks from {pdf_path.name}")
            
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk.page_content,
                    metadata={**chunk.metadata, "source": pdf_path.name}
                ))
        except Exception as e:
            st.error(f"Failed to load {pdf_path.name}: {e}")
            log_event(f"Error loading {pdf_path.name}: {e}", "ERROR")
    
    return docs

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_or_load_faiss(docs: List[Document], index_dir: str, force_rebuild: bool = False):
    embeddings = get_embeddings()
    
    if os.path.exists(index_dir) and not force_rebuild:
        try:
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            st.success("&#9989; Loaded existing FAISS index")
            log_event("Loaded existing FAISS index")
            return db
        except Exception as e:
            st.warning(f"&#9888; Failed to load index: {e}. Rebuilding...")
            log_event(f"Failed to load index: {e}", "WARNING")
            shutil.rmtree(index_dir, ignore_errors=True)
    
    with st.spinner("&#128296; Building FAISS index..."):
        log_event(f"Building FAISS index from {len(docs)} documents")
        db = FAISS.from_documents(docs, embeddings)
        os.makedirs(index_dir, exist_ok=True)
        db.save_local(index_dir)
        st.success("&#9989; FAISS index built and saved")
        log_event("FAISS index built successfully")
    
    return db

def get_groq_llm_instance(api_key, model_name):
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.0,
        max_tokens=4096
    )

# File Upload
st.header("&#128193; Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    accept_multiple_files=True,
    type=["pdf"],
    help="Upload PDF documents to create your knowledge base"
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["&#128172; Chat", "&#8505; Info & Tech Stack", "&#128220; Logs", "&#128221; Notes"])

NOTES_CONTENT = """
<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>
    
    <div style='background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(139, 92, 246, 0.3);'>
        <h4 style='color: #a78bfa !important; margin-top: 0;'>&#9889; Groq Integration</h4>
        <p style='font-size: 0.95rem;'>
            If you set <code>GROQ_API_KEY</code> in your environment, the app will automatically use Groq's high-speed inference via <code>langchain_groq</code>. 
            <br><br>
            <strong>Fallback:</strong> If Groq is unavailable, the app falls back to OpenAI (if configured) or handles the error gracefully. For production, ensure robust fallback logic is in place.
        </p>
    </div>

    <div style='background: rgba(236, 72, 153, 0.1); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(236, 72, 153, 0.3);'>
        <h4 style='color: #f472b6 !important; margin-top: 0;'>&#128190; Index Persistence</h4>
        <p style='font-size: 0.95rem;'>
            The FAISS vector index is saved locally in the <code>faiss_index_storage</code> directory. This speeds up repeated runs by avoiding re-indexing.
            <br><br>
            <strong>Action:</strong> Use the <strong>"&#128296; Rebuild Index"</strong> button in the sidebar to force a fresh rebuild if your documents change.
        </p>
    </div>

    <div style='background: rgba(56, 239, 125, 0.1); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(56, 239, 125, 0.3);'>
        <h4 style='color: #38ef7d !important; margin-top: 0;'>&#129504; Memory & Context</h4>
        <p style='font-size: 0.95rem;'>
            This implementation uses a simple <strong>session-state chat history</strong>. 
            <br><br>
            <strong>Customization:</strong> You can plug in LangChain's advanced memory classes (e.g., <code>ConversationBufferMemory</code>) to maintain longer context windows or persist chat history to a database (Redis, SQL) for production apps.
        </p>
    </div>

    <div style='background: rgba(251, 146, 60, 0.1); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(251, 146, 60, 0.3);'>
        <h4 style='color: #fb923c !important; margin-top: 0;'>&#128737; Safety & Instructions</h4>
        <p style='font-size: 0.95rem;'>
            The LLM is strictly instructed to answer <strong>ONLY</strong> from the provided sources.
            <br><br>
            <strong>Advisory:</strong> Always review AI-generated outputs before using them in critical production environments. Hallucinations are reduced but possible.
        </p>
    </div>

    <div style='background: rgba(96, 165, 250, 0.1); padding: 1.5rem; border-radius: 16px; border: 1px solid rgba(96, 165, 250, 0.3); grid-column: 1 / -1;'>
        <h4 style='color: #60a5fa !important; margin-top: 0;'>&#128257; LlamaIndex Alternative</h4>
        <p style='font-size: 0.95rem;'>
            If you prefer <strong>LlamaIndex</strong>, this app can be adapted to build a <code>VectorStoreIndex</code> and use its <code>as_query_engine()</code> method. The architecture is modular, allowing you to swap the LangChain retrieval logic with LlamaIndex's powerful indexing capabilities easily.
        </p>
    </div>

</div>
"""

with tab4:
    st.markdown("### &#128221; Developer Notes & Customization")
    st.html(NOTES_CONTENT)

with tab2:
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### &#128640; How to Use")
        st.markdown("""
        1. **Upload PDFs** using the file uploader above
        2. **Wait max 20-30 Sec** for documents to be processed
        3. **Ask questions** in the Chat tab
        4. **Wait max 20-30 Sec** for model gives you answer
        5. **View sources** by expanding the sources section
        
        **Troubleshooting:**
        - If answers are not found, click **"&#128296; Rebuild Index"** in the sidebar and re-upload files.
        - Check **"&#128269; Debug Context"** in the chat to see what the model is reading.
        """)
        
        st.markdown("### &#9889; Features")
        st.markdown("""
        - &#128196; **Multi-PDF Support** - Upload multiple documents
        - &#128269; **Semantic Search** - FAISS vector similarity
        - &#129302; **Multiple LLM Options** - Groq (fast) or OpenAI
        - &#128218; **Source Citations** - See where answers come from
        - &#128190; **Persistent Index** - Faster subsequent queries
        - &#128259; **Auto Fallback** - Switches models if one fails
        """)

    with col_info2:
        st.markdown("### &#128736; Tech Stack")
        st.markdown("""
        <div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid rgba(139, 92, 246, 0.3);'>
            <p><strong>Frontend:</strong> Streamlit</p>
            <p><strong>Orchestration:</strong> LangChain & LlamaIndex</p>
            <p><strong>Vector DB:</strong> FAISS (Facebook AI Similarity Search)</p>
            <p><strong>Embeddings:</strong> HuggingFace (all-mpnet-base-v2)</p>
            <p><strong>LLM:</strong> Groq (Llama 3) & OpenAI (GPT-3.5)</p>
            <p><strong>Language:</strong> Python 3.9+</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### &#128273; API Keys")
        st.markdown("""
        Set in `.streamlit/secrets.toml`:
        - `GROQ_API_KEY` (Recommended)
        - `OPENAI_API_KEY` (Fallback)
        """)

with tab3:
    st.markdown("### &#128220; Application Logs")
    
    # Toolbar
    col_tools1, col_tools2, col_tools3 = st.columns([1.5, 1.5, 4])
    with col_tools1:
        if st.button("&#128259; Refresh Logs", use_container_width=True):
            st.rerun()
            
    with col_tools2:
        # Prepare log data for download
        log_content = ""
        try:
            if os.path.exists('app.log'):
                with open('app.log', 'r') as f:
                    log_content = f.read()
        except:
            pass
            
        if not log_content and "app_logs" in st.session_state:
            log_content = "\n".join(st.session_state["app_logs"])
            
        st.download_button(
            label="&#11015; Download",
            data=log_content,
            file_name=f"rag_logs_{datetime.datetime.now().strftime('%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Interactive Log Viewer
    st.markdown("""
    <style>
        .log-container {
            background-color: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
        }
        .log-entry {
            padding: 4px 8px;
            border-bottom: 1px solid #2d2d2d;
            display: flex;
            align-items: flex-start;
        }
        .log-entry:hover {
            background-color: #2d2d2d;
        }
        .log-icon {
            margin-right: 10px;
            min-width: 20px;
        }
        .log-time {
            color: #888;
            margin-right: 10px;
            font-size: 0.8rem;
            min-width: 150px;
        }
        .log-msg-info { color: #4ec9b0; }
        .log-msg-warn { color: #ce9178; }
        .log-msg-error { color: #f44747; }
    </style>
    """, unsafe_allow_html=True)
    
    # Collect all logs from both sources
    all_logs = []
    
    # Read from app.log file
    if os.path.exists('app.log'):
        try:
            with open('app.log', 'r', encoding='utf-8') as f:
                file_logs = f.readlines()
                all_logs.extend([log.strip() for log in file_logs if log.strip()])
        except Exception as e:
            st.warning(f"Could not read app.log: {e}")
    
    # Add session state logs
    if "app_logs" in st.session_state and st.session_state["app_logs"]:
        all_logs.extend(st.session_state["app_logs"])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_logs = []
    for log in all_logs:
        if log not in seen:
            seen.add(log)
            unique_logs.append(log)
    
    # Reverse to show most recent first
    unique_logs.reverse()
    
    log_html = '<div class="log-container">'
    
    if unique_logs:
        for log_entry in unique_logs:
            # Parse entry
            icon = "🔹"
            msg_class = "log-msg-info"
            
            # Check for log level
            if "ERROR" in log_entry or "🔴" in log_entry or "&#128308;" in log_entry:
                icon = "🔴"
                msg_class = "log-msg-error"
            elif "WARNING" in log_entry or "⚠️" in log_entry or "&#9888;" in log_entry:
                icon = "⚠️"
                msg_class = "log-msg-warn"
            
            # Clean message - remove HTML entities and emojis
            clean_msg = log_entry.replace("&#128308;", "").replace("&#9888;", "").replace("&#128313;", "")
            clean_msg = clean_msg.replace("🔴", "").replace("⚠️", "").replace("🔵", "").strip()
            
            # Extract timestamp (format: YYYY-MM-DD HH:MM:SS,mmm or [HH:MM:SS])
            timestamp = ""
            if " - " in clean_msg:
                # Format from logging module: "2024-12-03 01:35:18,123 - INFO - message"
                parts = clean_msg.split(" - ", 2)
                if len(parts) >= 3:
                    timestamp = parts[0]
                    clean_msg = parts[2].strip()
            elif clean_msg.startswith("[") and "]" in clean_msg:
                # Format from session state: "[HH:MM:SS] message"
                parts = clean_msg.split("]", 1)
                timestamp = parts[0] + "]"
                clean_msg = parts[1].strip() if len(parts) > 1 else ""
            
            log_html += f"""
            <div class="log-entry">
                <span class="log-icon">{icon}</span>
                <span class="log-time">{html.escape(timestamp)}</span>
                <span class="{msg_class}">{html.escape(clean_msg)}</span>
            </div>
            """
    else:
        log_html += '<div style="color: #666; text-align: center; padding: 20px;">No logs available</div>'
    
    log_html += '</div>'
    st.html(log_html)

with tab1:
    if uploaded_files:
        temp_dir = tempfile.mkdtemp(prefix="rag_chatbot_")
        
        try:
            # Only process if not already processed or new files
            if "processed_files" not in st.session_state or st.session_state.get("processed_files") != [f.name for f in uploaded_files]:
                log_event(f"Processing {len(uploaded_files)} uploaded files")
                saved_paths = save_uploaded_files(uploaded_files, temp_dir)
                st.success(f"&#9989; Uploaded {len(saved_paths)} file(s)")
                
                with st.spinner("&#128214; Loading documents..."):
                    docs = load_documents_langchain(temp_dir)
                    
                    if not docs:
                        st.error("&#10060; No documents loaded")
                        st.stop()
                    
                    st.info(f"&#128196; Loaded {len(docs)} document chunks")
                
                db = build_or_load_faiss(docs, INDEX_DIR, force_rebuild)
                # Use regular similarity search for better accuracy
                st.session_state["retriever"] = db.as_retriever(
                    search_kwargs={"k": TOP_K}
                )
                st.session_state["processed_files"] = [f.name for f in uploaded_files]
            
            if "retriever" in st.session_state:
                retriever = st.session_state["retriever"]
                
                # Get API keys
                groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
                openai_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
                
                # Create enhanced prompt
                prompt = ChatPromptTemplate.from_template("""
You are an expert AI assistant helping users understand their documents.

Context from the uploaded documents:
{context}

Question: {input}

Instructions:
- First, carefully review the Context above for relevant information
- If the Context contains information about the question, provide a detailed answer based on it
- If the Context has partial information, use it and supplement with general knowledge where appropriate
- If the Context doesn't contain relevant information, provide a helpful general answer and note that specific details weren't found in the uploaded documents
- Be clear, accurate, and comprehensive in your response

Answer:
""")
                
                st.markdown("### &#128172; Chat with Your Documents")
                
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state["messages"] = []
                
                # Display chat history
                for role, message in st.session_state["messages"]:
                    if role == "user":
                        st.markdown(f"""
                        <div style='display: flex; gap: 12px; margin-bottom: 1rem;'>
                            <div style='width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #FF0099 0%, #493240 100%); display: flex; align-items: center; justify-content: center; font-size: 20px;'>&#128100;</div>
                            <div style='flex: 1; padding: 16px 22px; border-radius: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
                                {html.escape(message)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Extract debug chunks
                        debug_chunks = []
                        if "|||DEBUG_CHUNKS:" in message:
                            parts_debug = message.split("|||DEBUG_CHUNKS:")
                            message = parts_debug[0]
                            debug_info = parts_debug[1]
                            # Parse chunks
                            if "|||CHUNK" in debug_info:
                                chunk_parts = debug_info.split("|||CHUNK")
                                for chunk in chunk_parts:
                                    if ":" in chunk:
                                        chunk_num, content = chunk.split(":", 1)
                                        debug_chunks.append(f"**Chunk {chunk_num}**: {content.strip()}")

                        # Extract footer and sources
                        footer_text = ""
                        if "|||FOOTER:" in message:
                            parts_footer = message.split("|||FOOTER:")
                            message = parts_footer[0]
                            footer_text = parts_footer[1]
                        
                        parts = message.split("**&#128218; Sources:**")
                        main_answer = parts[0].strip()
                        
                        st.markdown(f"""
                        <div style='display: flex; gap: 12px; margin-bottom: 1rem;'>
                            <div style='width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #00F260 0%, #0575E6 100%); display: flex; align-items: center; justify-content: center; font-size: 20px;'>&#129302;</div>
                            <div style='flex: 1; padding: 16px 22px; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: #e0e0e0;'>
                                {html.escape(main_answer).replace(chr(10), '<br>')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if len(parts) > 1:
                            sources_raw = parts[1].strip().split("\n")
                            with st.expander("&#128218; View Sources", expanded=False):
                                for line in sources_raw:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    if line.startswith("**[") and "]**" in line:
                                        st.markdown(f"**{line.replace('**', '')}**")
                                    elif line.startswith(">"):
                                        st.markdown(f"> {line[1:].strip()}")
                        
                        if debug_chunks:
                            with st.expander("&#128269; Debug Context", expanded=False):
                                for chunk in debug_chunks:
                                    st.markdown(chunk)
                                    st.markdown("---")

                        if footer_text:
                            st.markdown(f"<small style='color:#888;'>&#129302; {footer_text}</small>", unsafe_allow_html=True)
                
                # Chat input
                # Chat input area
                def submit_query():
                    if st.session_state.get("query_input"):
                        st.session_state["messages"].append(("user", st.session_state["query_input"]))
                        st.session_state["query_input"] = ""

                st.text_area("Ask a question about your documents...", key="query_input")
                
                # Chat controls
                col_actions1, col_actions2 = st.columns(2)
                with col_actions1:
                    st.button("&#10140; Enter Query", on_click=submit_query, use_container_width=True)
                
                with col_actions2:
                    if st.button("&#128465; Clear Conversation", use_container_width=True):
                        st.session_state["messages"] = []
                        st.rerun()
                
                # Process latest message
                if st.session_state["messages"] and st.session_state["messages"][-1][0] == "user":
                    with st.spinner("&#129504; Thinking..."):
                        latest_query = st.session_state["messages"][-1][1]
                        log_event(f"Processing query: {latest_query[:50]}...")
                        
                        
                        success = False
                        models_to_try = []
                        all_errors = []  # Collect all errors
                        
                        if use_groq and GROQ_AVAILABLE and groq_key:
                            models_to_try.extend([(name, "groq") for name in GROQ_MODELS])
                        
                        if OPENAI_AVAILABLE and openai_key:
                            models_to_try.append(("gpt-3.5-turbo", "openai"))
                        
                        if not models_to_try:
                            st.error("&#10060; No LLM configured. Please set API keys.")
                            st.stop()
                        
                        for model_name, provider in models_to_try:
                            try:
                                if provider == "groq":
                                    current_llm = get_groq_llm_instance(groq_key, model_name)
                                else:
                                    current_llm = OpenAI(temperature=0.0, openai_api_key=openai_key)
                                
                                document_chain = create_stuff_documents_chain(current_llm, prompt)
                                qa_chain = create_retrieval_chain(retriever, document_chain)
                                
                                result = qa_chain.invoke({"input": latest_query})
                                
                                answer = result.get("answer", "No answer generated")
                                source_docs = result.get("context", [])
                                
                                # Log retrieved chunks for debugging
                                log_event(f"Retrieved {len(source_docs)} chunks for query")
                                
                                full_answer = answer
                                full_answer += f"|||FOOTER:Generated with: {model_name}"
                                
                                # Add debug info about retrieved chunks
                                if source_docs:
                                    sources_text = "\n\n**&#128218; Sources:**\n"
                                    for i, doc in enumerate(source_docs, 1):
                                        source = doc.metadata.get("source", f"Document {i}")
                                        raw_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                        snippet = " ".join(raw_snippet.split())
                                        sources_text += f"\n**[{i}]** {source}\n> {snippet}\n"
                                    full_answer += sources_text
                                    
                                    # Add full retrieved context for debugging (hidden in metadata)
                                    full_answer += f"\n\n|||DEBUG_CHUNKS:{len(source_docs)}"
                                    for i, doc in enumerate(source_docs, 1):
                                        chunk_preview = doc.page_content[:500].replace('\n', ' ')
                                        full_answer += f"\n|||CHUNK{i}:{chunk_preview}"
                                else:
                                    log_event("WARNING: No source documents retrieved!", "WARNING")
                                
                                st.session_state["messages"].append(("assistant", full_answer))
                                log_event(f"Generated answer using {model_name}")
                                st.rerun()
                                success = True
                                break
                                
                            except Exception as e:
                                error_msg = str(e)
                                all_errors.append(f"{model_name}: {error_msg}")
                                st.warning(f"&#9888; {model_name} failed: {error_msg[:100]}...")
                                log_event(f"Model {model_name} failed: {error_msg}", "WARNING")
                                continue
                        
                        if not success:
                            st.error("&#10060; All models failed. Please check your API keys or try again later.")
                            with st.expander("&#128269; View Error Details"):
                                for error in all_errors:
                                    st.code(error, language="text")
                            st.session_state["messages"].append(("assistant", "&#10060; Sorry, I couldn't generate an answer."))
                            st.rerun()
                
                # Clear chat button

        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            log_event(f"App error: {e}", "ERROR")

    else:
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: rgba(139, 92, 246, 0.1); border-radius: 20px; border: 2px solid rgba(139, 92, 246, 0.3);'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>&#128193;</div>
            <h2 style='color: #a78bfa !important;'>Upload Your PDFs to Get Started</h2>
            <p style='font-size: 1.1rem; color: #cbd5e1;'>Use the file uploader above to upload one or more PDF files</p>
            <p style='color: #94a3b8;'>Limit: 200MB per file • Supports multiple PDFs</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Build By RATNESH SINGH(Data Scientist)</p>
    <p>Universal RAG Chatbot • Built with LangChain, FAISS & Streamlit</p>
    <p>Powered by Groq LLM & HuggingFace Embeddings</p>
</div>
""", unsafe_allow_html=True)
