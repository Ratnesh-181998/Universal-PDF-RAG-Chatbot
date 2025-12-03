# 🚀 Quick Start Guide

## Get Started in 5 Minutes!

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will install:
- Streamlit (UI framework)
- LangChain (orchestration)
- LlamaIndex (document loading)
- FAISS (vector store)
- Groq/OpenAI clients
- HuggingFace transformers

### Step 2: Get Your API Key

**Option A: Groq (Recommended - FREE & FAST)**

1. Go to [https://console.groq.com/](https://console.groq.com/)
2. Sign up for a free account
3. Create an API key
4. Copy your key

**Option B: OpenAI**

1. Go to [https://platform.openai.com/](https://platform.openai.com/)
2. Create an API key
3. Copy your key (requires payment)

### Step 3: Set Environment Variable

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY="your_key_here"
```

**Windows CMD:**
```cmd
set GROQ_API_KEY=your_key_here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="your_key_here"
```

### Step 4: Run the App

```bash
streamlit run app.py
```

### Step 5: Use the Chatbot

1. **Upload PDFs** - Click the file uploader and select your PDF files
2. **Wait** - First time takes ~30 seconds to build the index
3. **Ask Questions** - Type your question and click Send
4. **Get Answers** - Receive answers with source citations!

## 🎯 Example Questions to Try

Once you've uploaded a PDF, try asking:

- "What is the main topic of this document?"
- "Summarize the key points"
- "What are the conclusions?"
- "Explain [specific concept] from the document"
- "What does the author say about [topic]?"

## 💡 Tips

- **First run is slower** - Building embeddings takes time
- **Subsequent runs are fast** - Index is cached
- **Use Groq for speed** - Much faster than OpenAI
- **Upload multiple PDFs** - Build a knowledge base
- **Check sources** - Always verify the citations

## ❓ Common Issues

**"No LLM configured"**
→ Set your API key (see Step 3)

**"Failed to load documents"**
→ Ensure PDF has extractable text

**Slow performance**
→ Use Groq instead of OpenAI

**Import errors**
→ Run `pip install -r requirements.txt` again

## 🎉 You're Ready!

Enjoy your RAG chatbot! 🚀
