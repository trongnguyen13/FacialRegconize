# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Company Staff Directory app built with Python/Streamlit. Staff register their face with name, role, and department. New employees can take a photo of a colleague to instantly identify them. Uses DeepFace (ArcFace model, 512-dim) for face embeddings and Pinecone for vector similarity search. Python 3.13+.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application (opens at http://localhost:8501)
streamlit run app.py

# Configure environment (copy .env.example, then set PINECONE_API_KEY)
cp .env.example .env
```

No test framework is currently configured.

## Architecture

**Entry point:** `app.py` — Streamlit app with component-based UI. Each page is a `render_*()` function. Navigation via sidebar with 4 pages: Home, Find Staff, Register Staff, Staff Directory. Uses `MODEL_NAME = "ArcFace"` globally (no user-facing model selection). Streamlit session state holds the Pinecone connection.

**Utils layer (`utils/`):**
- `deepface_helper.py` — Wrapper over DeepFace SDK. Only `extract_embedding()` is used by the app. Other functions (verify, analyze, detect) exist but are unused.
- `pinecone_helper.py` — `PineconeHelper` class wrapping Pinecone SDK. Core operations: `register_face()` (upsert), `search_faces()` (query), `delete_face()`, `list_all_faces()` (paginated list + fetch). Serverless index on AWS us-east-1 with cosine similarity.
- `image_utils.py` — Image I/O helpers. Saves uploads to system temp dir (`face_recognition_temp/`), displays images in Streamlit.

**Data flow:** Photo (upload or webcam) → temp file → DeepFace embedding → Pinecone upsert (register) or query (find). Metadata stored per vector: `{name, role, department, registered_at}`.

## Key Patterns

- **Dual input:** Every feature supports both file upload and webcam capture via radio button selection.
- **Graceful degradation:** App shows Home page without Pinecone; other pages show a config warning.
- **Configuration:** Environment variables via python-dotenv (`.env`). Key vars: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`.
