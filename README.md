# 📄 DocuChat — RAG-Based Document Q&A App

DocuChat is a Retrieval-Augmented Generation (RAG) application that lets you
upload documents (PDF or TXT) and ask natural-language questions about them.
Instead of relying on a language model's memorized knowledge, DocuChat
retrieves the most relevant passages from your own documents and grounds
every answer in that retrieved context — reducing hallucination and citing
exactly which source was used.

## Why this project

Most LLM demos just call an API and hope for the best. DocuChat implements
the actual retrieval pipeline that powers real-world AI products — internal
knowledge bases, customer support bots, research assistants — end to end:
chunking, embedding, vector search, and grounded generation.

## Architecture

```
 ┌────────────┐    ┌───────────┐    ┌─────────────┐    ┌──────────────┐
 │  PDF / TXT │───▶│  Chunking  │───▶│  Embeddings  │───▶│ FAISS Vector │
 │  Documents │    │ (LangChain)│    │(sentence-    │    │    Store     │
 └────────────┘    └───────────┘    │ transformers)│    └──────┬───────┘
                                     └──────────────┘           │
                                                                  ▼
 ┌────────────┐    ┌──────────────────┐    ┌─────────────────────────┐
 │ User Query │───▶│ Similarity Search │───▶│  Claude generates answer │
 └────────────┘    │  (top-k chunks)   │    │  grounded in retrieved   │
                    └──────────────────┘    │  context, with sources  │
                                             └─────────────────────────┘
```

## Tech Stack

| Component        | Tool                                              |
|-------------------|---------------------------------------------------|
| Orchestration     | LangChain                                         |
| Embeddings        | sentence-transformers (`all-MiniLM-L6-v2`, local, free) |
| Vector store      | FAISS                                             |
| Generation (LLM)  | Ollama (`llama3.2`)                              |
| UI                | Streamlit                                         |
| PDF parsing       | pypdf                                             |

Embeddings run locally and are free — the only API key you need is for the
generation step (Claude).

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/SUJAL006/docuchat.git
   cd docuchat
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama**
   Install Ollama, then make sure the Ollama service is running and pull the model:
   ```bash
   ollama pull llama3.2
   ```
   The app uses Ollama locally, so no Anthropic API key is required.

5. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The app opens at `http://localhost:8501`.

6. **Try it out**
   Upload `sample_docs/sample.txt` (included in this repo) and ask something
   like *"How much is the home office stipend?"* to see grounded, cited
   answers.

## How it works (pipeline detail)

1. **Load** — PDFs/TXT files are parsed into text using `pypdf` / LangChain loaders.
2. **Chunk** — Text is split into ~1000-character overlapping chunks so context isn't cut mid-sentence.
3. **Embed** — Each chunk is converted into a vector using a local sentence-transformers model (no API cost).
4. **Store** — Vectors are indexed in FAISS for fast similarity search.
5. **Retrieve** — On each question, the query is embedded and the top-4 most similar chunks are pulled from the index.
6. **Generate** — Ollama receives the question + retrieved chunks and is explicitly instructed to answer only from that context, preventing hallucinated answers.
7. **Cite** — The UI shows exactly which document and page each answer came from.

## Possible extensions

- Swap FAISS for ChromaDB with persistent storage across sessions
- Add conversation memory for multi-turn follow-up questions
- Support additional file types (.docx, .csv)
- Add a re-ranking step (e.g., cross-encoder) before generation for higher precision
- Deploy with authentication for multi-user use

## Deployment

This app deploys for free on
[Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push this repo to GitHub.
2. Connect the repo on Streamlit Community Cloud.
3. Add `ANTHROPIC_API_KEY` under app settings → Secrets.
4. Deploy — you'll get a public URL to share (e.g., in your resume/portfolio).

## License

MIT — free to use and adapt.
