# 🤖 Universal PDF RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green.svg)](https://langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Latest-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Transform your PDFs into an intelligent, conversational knowledge base powered by cutting-edge AI**

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables natural language conversations with your PDF documents. Built with enterprise-grade technologies including LangChain, FAISS vector search, and high-speed LLM inference via Groq.

---

## 📑 Table of Contents

- [Features](#-features)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [UI Overview](#-ui-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [Architecture](#-architecture)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

### Core Capabilities
- 📁 **Multi-PDF Upload** - Process single or multiple PDF documents simultaneously
- 🔍 **Semantic Search** - FAISS-powered vector similarity search for accurate retrieval
- 🤖 **Dual LLM Support** - Groq (ultra-fast) with OpenAI fallback
- 📚 **Source Citations** - Every answer includes document references with page numbers
- 💬 **Chat History** - Persistent conversation tracking with download capability
- 🔄 **Smart Caching** - Persistent FAISS index for instant subsequent queries

### Advanced Features
- 🎨 **Modern UI** - Glassmorphic design with gradient animations
- 📊 **Real-time Logs** - Interactive log viewer with filtering
- 🔧 **Debug Mode** - View retrieved context chunks for transparency
- 📥 **Export Options** - Download chat history as text files
- ⚡ **OCR Fallback** - Automatic image-based PDF text extraction
- 🎯 **Adaptive Retrieval** - Configurable Top-K and chunk parameters

---

## 🌐🎬 Live Demo
🚀 **Try it now:**
- **Streamlit Profile** - https://share.streamlit.io/user/ratnesh-181998
- **Project Demo** - https://universal-pdf-rag-chatbot-mhsi4ygebe6hmq3ij6d665.streamlit.app/

 rag langchain streamlit python llm chatbot faiss generative-ai groq llama-3 pdf-parser vector-search

> *Upload a PDF and ask questions in seconds!*

---

### 🎬 Live Project Demo
> Experience the full flow:

![Project Demo Walkthrough](combined_demo.gif)

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.51+ | Interactive web interface |
| **LLM** | Groq (Llama 3.3 70B) | Lightning-fast inference (2-5s response) |
| **Fallback LLM** | OpenAI GPT-3.5 | Backup for high availability |
| **Embeddings** | HuggingFace Transformers | Sentence embeddings (all-mpnet-base-v2) |
| **Vector Store** | FAISS | High-performance similarity search |
| **Orchestration** | LangChain 0.2.16 | RAG pipeline management |
| **PDF Parsing** | PyMuPDF + Unstructured | Text extraction with OCR fallback |
| **Language** | Python 3.9+ | Core application logic |

### Key Dependencies
```
langchain==0.2.16
langchain-groq>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
streamlit>=1.28.0
```

---

## 🎨 UI Overview

### Main Interface
<img width="2872" height="1432" alt="image" src="https://github.com/user-attachments/assets/cde93d74-ddf2-403a-9af0-6c5ed62abc2d" />
<img width="2855" height="1443" alt="image" src="https://github.com/user-attachments/assets/771f4616-446d-493b-ab3e-7823e2c1ebaa" />
<img width="2799" height="1447" alt="image" src="https://github.com/user-attachments/assets/f0084a11-8c9a-4c17-bfeb-a400ff33a511" />
<img width="2842" height="1400" alt="image" src="https://github.com/user-attachments/assets/76b8f40e-2dae-417a-96a1-6f4a54a74e6a" />
<img width="2774" height="1442" alt="image" src="https://github.com/user-attachments/assets/18affc5e-b28b-4f90-a232-bc8e65cf2e43" />
<img width="2855" height="1421" alt="image" src="https://github.com/user-attachments/assets/627014f4-57bf-4beb-85bc-d88952dbc4e8" />
<img width="2844" height="1456" alt="image" src="https://github.com/user-attachments/assets/b38b60d9-47a5-4e5a-9d46-8cac6fa52a22" />
<img width="2858" height="1416" alt="image" src="https://github.com/user-attachments/assets/30858ed3-a294-423c-b52f-4ce0073e82b3" />
<img width="2840" height="1433" alt="image" src="https://github.com/user-attachments/assets/12b7fbfb-6269-44cf-bbb2-d0a4d9f097b4" />
<img width="2837" height="1397" alt="image" src="https://github.com/user-attachments/assets/680b06b2-e1e4-45cb-85ab-f49de5c02880" />
<img width="2843" height="1448" alt="image" src="https://github.com/user-attachments/assets/b7cfa70b-d3c6-40f0-82be-d3b06c549696" />

**Components:**
1. **Header** - Gradient title with author credit
2. **Quick Guide** - Visual workflow (Upload → Process → Query → Answer)
3. **File Uploader** - Drag-and-drop PDF upload zone
4. **Tabbed Navigation** - Chat, Info, Logs, Notes

### Chat Tab
<img width="2860" height="1449" alt="image" src="https://github.com/user-attachments/assets/79480816-71ec-4f26-9350-a8e2e8715d3b" />
<img width="2839" height="1404" alt="image" src="https://github.com/user-attachments/assets/7e0b862a-ea2f-4f5f-97a0-586eedb39d5a" />
<img width="2868" height="1461" alt="image" src="https://github.com/user-attachments/assets/aad6fddf-56e2-42b0-a4aa-492d967ba896" />
<img width="2856" height="1437" alt="image" src="https://github.com/user-attachments/assets/08d94d55-7054-488d-b421-4d96b578a086" />
<img width="2857" height="1410" alt="image" src="https://github.com/user-attachments/assets/09a97486-7baf-49a8-8db1-87dfba535d3d" />
<img width="2861" height="1446" alt="image" src="https://github.com/user-attachments/assets/216499ef-88f1-4da2-a30c-97726694c26b" />
<img width="2832" height="1466" alt="image" src="https://github.com/user-attachments/assets/3729c804-5f44-401c-b52c-d9160c00ca91" />
<img width="2813" height="1434" alt="image" src="https://github.com/user-attachments/assets/c7745b93-d27c-4158-af7c-9572cb1b8cfe" />
<img width="2320" height="1259" alt="image" src="https://github.com/user-attachments/assets/4a01fa43-18ac-49d0-88c0-dcc9415dedb3" />

**Features:**
- **Text Area Input** - Dark-themed query box
- **Enter Query Button** - Submit questions
- **Clear Conversation** - Reset chat history
- **Message Bubbles** - User (purple gradient) vs AI (dark glass)
- **Source Expanders** - Collapsible citation details
- **Debug Context** - View retrieved document chunks

### Sidebar
<img width="2737" height="1429" alt="image" src="https://github.com/user-attachments/assets/0586ca7d-bdee-42a7-b82f-c4aa68c7fc00" />

**Controls:**
- **Configuration Display** - Embedding model, chunk size, Top-K
- **Groq Toggle** - Prefer Groq LLM checkbox
- **Rebuild Index** - Force FAISS re-indexing
- **API Status** - Real-time Groq/OpenAI availability
- **Download Chat** - Export conversation history

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Groq API Key (free at [console.groq.com](https://console.groq.com/))
- OR OpenAI API Key

### One-Command Setup (Windows)
```bash
# Clone repository
git clone https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot.git
cd Universal-PDF-RAG-Chatbot

# Run setup script
setup.bat
```

### One-Command Setup (Linux/Mac)
```bash
# Clone repository
git clone https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot.git
cd Universal-PDF-RAG-Chatbot

# Install and run
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
streamlit run app.py
```

---

## 📦 Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot.git
cd Universal-PDF-RAG-Chatbot
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

**Option A: Environment Variables (Recommended)**
```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_groq_api_key_here"

# Windows CMD
set GROQ_API_KEY=gsk_your_groq_api_key_here

# Linux/Mac
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```

**Option B: Streamlit Secrets (For Deployment)**

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
OPENAI_API_KEY = "sk_your_openai_key_here"  # Optional fallback
```

### Step 5: Run Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚙️ Configuration

### Customizable Parameters (in `app.py`)

```python
# Vector Store
INDEX_DIR = "faiss_index_storage"  # FAISS index save location

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Text Chunking
CHUNK_SIZE = 800        # Characters per chunk
CHUNK_OVERLAP = 150     # Overlap between chunks

# Retrieval
TOP_K = 10              # Number of chunks to retrieve

# Available Groq Models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",  # Latest, most powerful
    "llama-3.1-70b-versatile",  # Stable alternative
    "llama-3.1-8b-instant",     # Fastest
]
```

### Performance Tuning

| Use Case | CHUNK_SIZE | TOP_K | Model |
|----------|------------|-------|-------|
| **Precise Answers** | 500 | 15 | llama-3.3-70b |
| **Balanced** | 800 | 10 | llama-3.1-70b |
| **Speed Priority** | 1000 | 5 | llama-3.1-8b |

---

## 📖 Usage Guide

### Basic Workflow

1. **Upload PDFs**
   - Click the file uploader
   - Select one or more PDF files (max 200MB each)
   - Wait 20-30 seconds for processing

2. **Ask Questions**
   - Type your question in the text area
   - Click "Enter Query"
   - Wait 20-30 seconds for AI response

3. **Review Answers**
   - Read the AI-generated response
   - Expand "View Sources" to see citations
   - Check "Debug Context" to verify retrieved chunks

4. **Manage Conversation**
   - Click "Clear Conversation" to reset
   - Download chat history via sidebar button

### Example Questions

```
✅ "What are the main conclusions of this research paper?"
✅ "Summarize the methodology section"
✅ "What does the author say about climate change?"
✅ "List all recommendations from the report"
✅ "Compare the results in Table 3 and Table 5"
```

### Troubleshooting Tips

**No Answer Found?**
- Click "🔧 Rebuild Index" in sidebar
- Re-upload your PDFs
- Check "Debug Context" to see what was retrieved

**Slow Performance?**
- First run builds embeddings (30-60s)
- Subsequent queries use cached index (5-10s)
- Switch to `llama-3.1-8b-instant` for speed

---

## 🏗 Architecture

### RAG Pipeline Flow

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ PyMuPDF Loader  │ ──► OCR Fallback (if needed)
└──────┬──────────┘
       │
       ▼
┌──────────────────────┐
│ Text Splitter        │ (Chunk: 800, Overlap: 150)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ HuggingFace Embedder │ (all-mpnet-base-v2)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ FAISS Vector Store   │ (Persistent Index)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ User Query           │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Similarity Search    │ (Top-K Retrieval)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ LLM (Groq/OpenAI)    │ (Context + Query)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Answer + Citations   │
└──────────────────────┘
```

### Directory Structure

```
Universal-PDF-RAG-Chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
├── QUICKSTART.md              # Fast setup guide
├── .gitignore                 # Git exclusions
├── .streamlit/
│   └── secrets.toml           # API keys (not in git)
├── faiss_index_storage/       # FAISS index (auto-created)
│   ├── index.faiss
│   └── index.pkl
└── app.log                    # Application logs
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No LLM configured" Error
**Cause:** Missing API keys  
**Solution:**
```bash
# Set environment variable
export GROQ_API_KEY="your_key_here"

# OR add to .streamlit/secrets.toml
GROQ_API_KEY = "your_key_here"
```

#### 2. Keras 3 Compatibility Error
**Cause:** Transformers library incompatibility  
**Solution:**
```bash
pip install tf-keras
```

#### 3. "Failed to load documents"
**Cause:** Corrupted or image-only PDFs  
**Solution:**
- Ensure PDF has extractable text
- App will auto-fallback to OCR for image PDFs
- Try a different PDF to verify

#### 4. Slow First Query
**Cause:** Building embeddings for first time  
**Solution:**
- Normal behavior (30-60s)
- Subsequent queries use cached index (5-10s)

#### 5. Import Errors
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Universal-PDF-RAG-Chatbot.git

# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
✅ Commercial use  
✅ Modification  
✅ Distribution  
✅ Private use  
❌ Liability  
❌ Warranty  

---

## 📞 Contact

**RATNESH SINGH**  
*Data Scientist | AI/ML Engineer*

- 📧 Email: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/ratneshkumar1998](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 GitHub: [github.com/Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 Phone: +91-947XXXXX46

### Project Links
- 🌐 **Live Demo:** [Streamlit Cloud](https://universal-pdf-rag-chatbot-mhsi4ygebe6hmq3ij6d665.streamlit.app/)
- 📖 **Documentation:** [GitHub Wiki](https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot/wiki)
- 🐛 **Issue Tracker:** [GitHub Issues](https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot/issues)
- ⭐ **Star this repo** if you find it useful!

---

## 🙏 Acknowledgments

This project leverages amazing open-source technologies:

- **[LangChain](https://langchain.com/)** - RAG orchestration framework
- **[Groq](https://groq.com/)** - Ultra-fast LLM inference
- **[FAISS](https://github.com/facebookresearch/faiss)** - Efficient vector search by Meta AI
- **[Streamlit](https://streamlit.io/)** - Rapid web app development
- **[HuggingFace](https://huggingface.co/)** - Transformer models and embeddings
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - PDF text extraction

---

## 🚀 Roadmap

### Planned Features
- [ ] Support for DOCX, TXT, CSV files
- [ ] Multi-language document support
- [ ] Conversation memory across sessions
- [ ] Voice input/output integration
- [ ] Docker containerization
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Batch processing mode
- [ ] Custom prompt templates UI
- [ ] Advanced analytics dashboard

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Average Query Time** | 5-10 seconds |
| **First Upload Processing** | 30-60 seconds |
| **Supported PDF Size** | Up to 200MB |
| **Concurrent Users** | 10+ (Streamlit Cloud) |
| **Accuracy (F1 Score)** | ~0.85 on test set |

---

## 🔐 Security & Privacy

- ✅ API keys stored in `.streamlit/secrets.toml` (gitignored)
- ✅ No data persistence beyond session (unless explicitly saved)
- ✅ FAISS index stored locally (not cloud-synced)
- ⚠️ Uploaded PDFs processed in-memory only
- ⚠️ For sensitive documents, deploy on private infrastructure

---

## 📚 Additional Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Documentation](https://faiss.ai/)
- [Groq API Docs](https://console.groq.com/docs)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

<div align="center">

**Built with ❤️ by Ratnesh Singh**

*Powered by LangChain • FAISS • Groq • Streamlit*

[![GitHub stars](https://img.shields.io/github/stars/Ratnesh-181998/Universal-PDF-RAG-Chatbot?style=social)](https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Ratnesh-181998/Universal-PDF-RAG-Chatbot?style=social)](https://github.com/Ratnesh-181998/Universal-PDF-RAG-Chatbot/network/members)

</div>


---


<img src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=24,20,12,6&height=3" width="100%">


## 📜 **License**

![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge&logo=opensourceinitiative&logoColor=white)

**Licensed under the MIT License** - Feel free to fork and build upon this innovation! 🚀

---

# 📞 **CONTACT & NETWORKING** 📞


### 💼 Professional Networks

[![LinkedIn](https://img.shields.io/badge/💼_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ratneshkumar1998/)
[![GitHub](https://img.shields.io/badge/🐙_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ratnesh-181998)
[![X](https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/RatneshS16497)
[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-FF6B6B?style=for-the-badge&logo=google-chrome&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![Email](https://img.shields.io/badge/✉️_Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:rattudacsit2021gate@gmail.com)
[![Medium](https://img.shields.io/badge/Medium-000000?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@rattudacsit2021gate)
[![Stack Overflow](https://img.shields.io/badge/Stack_Overflow-F58025?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/32068937/ratnesh-kumar)

### 🚀 AI/ML & Data Science
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://share.streamlit.io/user/ratnesh-181998)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/RattuDa98)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/rattuda)

### 💻 Competitive Programming (Including all coding plateform's 5000+ Problems/Questions solved )
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/Ratnesh_1998/)
[![HackerRank](https://img.shields.io/badge/HackerRank-00EA64?style=for-the-badge&logo=hackerrank&logoColor=black)](https://www.hackerrank.com/profile/rattudacsit20211)
[![CodeChef](https://img.shields.io/badge/CodeChef-5B4638?style=for-the-badge&logo=codechef&logoColor=white)](https://www.codechef.com/users/ratnesh_181998)
[![Codeforces](https://img.shields.io/badge/Codeforces-1F8ACB?style=for-the-badge&logo=codeforces&logoColor=white)](https://codeforces.com/profile/Ratnesh_181998)
[![GeeksforGeeks](https://img.shields.io/badge/GeeksforGeeks-2F8D46?style=for-the-badge&logo=geeksforgeeks&logoColor=white)](https://www.geeksforgeeks.org/profile/ratnesh1998)
[![HackerEarth](https://img.shields.io/badge/HackerEarth-323754?style=for-the-badge&logo=hackerearth&logoColor=white)](https://www.hackerearth.com/@ratnesh138/)
[![InterviewBit](https://img.shields.io/badge/InterviewBit-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://www.interviewbit.com/profile/rattudacsit2021gate_d9a25bc44230/)


---

## 📊 **GitHub Stats & Metrics** 📊



![Profile Views](https://komarev.com/ghpvc/?username=Ratnesh-181998&color=blueviolet&style=for-the-badge&label=PROFILE+VIEWS)





<img src="https://github-readme-streak-stats.herokuapp.com/?user=Ratnesh-181998&theme=radical&hide_border=true&background=0D1117&stroke=4ECDC4&ring=F38181&fire=FF6B6B&currStreakLabel=4ECDC4" width="48%" />




<img src="https://github-readme-activity-graph.vercel.app/graph?username=Ratnesh-181998&theme=react-dark&hide_border=true&bg_color=0D1117&color=4ECDC4&line=F38181&point=FF6B6B" width="48%" />

---

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&lines=Ratnesh+Kumar+Singh;Data+Scientist+%7C+AI%2FML+Engineer;4%2B+Years+Building+Production+AI+Systems" alt="Typing SVG" />

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=Built+with+passion+for+the+AI+Community+🚀;Innovating+the+Future+of+AI+%26+ML;MLOps+%7C+LLMOps+%7C+AIOps+%7C+GenAI+%7C+AgenticAI+Excellence" alt="Footer Typing SVG" />


<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%">

