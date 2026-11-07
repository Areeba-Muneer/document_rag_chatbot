# Document-Based AI Chatbot using RAG

A local Retrieval-Augmented Generation (RAG) chatbot. It extracts text from PDF, TXT and DOCX files, splits the text into chunks, creates real embeddings with Sentence Transformers, stores vectors in FAISS, retrieves the top three relevant chunks, and asks a local Ollama LLM to answer using that context.

## Technology
- Python
- Streamlit
- PyPDF
- python-docx
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- Google Gemini API (`gemini-2.5-flash`)
- python-dotenv

## Requirements
- Python 3.10 or newer
- A Google AI Studio API key
- Internet access for the first download of the embedding model and Python packages

## Setup (Windows / VS Code)

1. Extract the project ZIP and open the folder in VS Code.
2. Open the VS Code terminal.
3. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

5. Get an API key from https://aistudio.google.com/app/apikey.
6. Copy `.env.example` to `.env` and paste your key after `GEMINI_API_KEY=`. Never commit `.env` or put API keys in source code.
7. Put at least five documents on one topic in the `docs` folder. Five sample Python-learning documents are included.
8. Start the chatbot:

   ```powershell
   streamlit run app.py
   ```

9. In the browser sidebar, click **Build / Refresh Index**. Wait for the embedding/indexing process to finish.
10. Ask questions in the chat box. The interface displays retrieved source filenames and chunks.

## RAG workflow
1. **Load:** Read PDF, TXT and DOCX documents.
2. **Chunk:** Split each document into smaller word-based passages while retaining source filename and chunk number.
3. **Embed:** Use the pretrained `all-MiniLM-L6-v2` embedding model to convert every chunk into a numerical vector.
4. **Store:** Save normalized vectors in a FAISS inner-product index and save chunk text/metadata alongside it.
5. **Query:** Convert the user's question into an embedding and retrieve the top three nearest chunks.
6. **Generate:** Send the question and retrieved passages to the configured Gemini model with instructions to use only the supplied context.
7. **Grounding:** If no retrieved chunk meets the configured similarity threshold, return a not-found response rather than guessing.

## Suggested test questions
Use these questions after building the index:
1. What is Python?
2. What are Python's built-in data types?
3. What is a function in Python?
4. How do variables, data types and functions work together?
5. Who invented Java programming language?

Question 5 is intentionally outside the sample documents. The expected behavior is a not-found response, not a guessed answer.

## Testing record
Run each question in the actual application and record the observed answer and source filename(s). Do not claim a test passed until you have run it.

| # | Question | Generated answer | Source file(s) | Supported? |
|---|---|---|---|---|
| 1 | What is Python? | Fill after running | Fill after running | Yes/No |
| 2 | What are Python's built-in data types? | Fill after running | Fill after running | Yes/No |
| 3 | What is a function in Python? | Fill after running | Fill after running | Yes/No |
| 4 | How do variables, data types and functions work together? | Fill after running | Fill after running | Yes/No |
| 5 | Who invented Java programming language? | Fill after running | None expected | Yes/No |

## Notes and limitations
- The first run downloads the embedding model and may take time.
- A valid Google AI Studio API key and internet connection are required for Gemini answer generation.
- Scanned/image-only PDFs may not contain extractable text; OCR is not included.
- The similarity threshold is configurable in `.env`. Tune it against your own documents and test questions; a fixed threshold is not universally correct.
- The FAISS index is rebuilt from the contents of `docs` and `uploaded_docs` when the sidebar button is clicked.
