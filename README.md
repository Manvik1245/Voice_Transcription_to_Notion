An AI-powered academic assistant designed for real-time lecture transcription, automated summarization, and seamless integration with Notion. Built specifically for students to capture every detail of complex STEM lectures (like CICS or Chemistry) without the friction of manual note-taking.

🚀 Features
Dual-Model Transcription Strategy: * Live View: Uses Whisper-Tiny for low-latency, real-time visual feedback during lectures.

Batch Correction: Automatically re-processes the entire lecture in the background using Whisper-Large-v3-Turbo for maximum accuracy once recording stops.

Intelligent Summarization: Leverages Meta-Llama 4 (128-expert MoE) via Groq to generate exam-ready study points, definitions, and key takeaways.

RAM-First Architecture: Audio is stored as NumPy arrays in system memory—no messy temporary files or disk clutter.

Notion Integration: Automatically creates a new page in your specified Notion database, complete with:

Chronological Date Stamping

Class/Subject Tagging

AI-Generated Summary

Full Chunker-Optimized Transcript (handles the 2,000-character Notion API limit).

🛠️ Technical Stack
Language: Python 3.10+

UI Framework: Tkinter

AI Models: * Transcription: faster-whisper (Large-v3-Turbo & Tiny)

LLM: Llama-4-Maverick-17B (via Groq API)

API: Notion Client (REST)

Hardware Optimization: Specifically tuned for Apple Silicon (M1/M2/M3) using int8 quantization for efficient CPU/Neural Engine performance.

Setup Environment Variables:
Create a .env file or replace the constants in the script with your keys:

GROQ_API_KEY: Get from Groq Console

NOTION_TOKEN: Get from Notion Integrations

NOTION_DATABASE_ID: The ID of your specific Notion DatabaseRun the App: python main.py

Select Class: Choose your subject (e.g., CICS 160 or CHEM 111) from the dropdown.

Enter Topic: Type the name of the today's lesson.

Record: Hit Start Recording. You will see a live "rough draft" transcription.

Finish: Hit Stop Recording.

Wait ~60 seconds for the program to perform the high-quality batch re-transcription.

The app will turn Green when your notes are safely synced to Notion.

This tool was built to handle the specific vocabulary of Computer Science and Chemistry. By using the large-v3-turbo model in batch mode, it accurately captures:

Programming syntax and logic descriptions.

Chemical formulas and reaction names.

Mathematical terminology often missed by standard real-time dictation tools.
