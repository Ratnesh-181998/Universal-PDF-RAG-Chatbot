@echo off
echo ========================================
echo Starting RAG Chatbot...
echo ========================================

if not exist venv (
    echo Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Checking API Keys...
if "%GROQ_API_KEY%"=="" (
    if "%OPENAI_API_KEY%"=="" (
        echo WARNING: No API keys found in environment variables.
        echo You may need to set GROQ_API_KEY or OPENAI_API_KEY.
        echo.
    )
)

echo Launching Streamlit...
streamlit run app.py

pause
