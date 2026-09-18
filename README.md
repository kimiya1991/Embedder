<<<<<<< HEAD
# Embedder Lab

Compare embedding models on `sample.txt`. Each `##` heading is one chunk. Labeled questions measure whether a model retrieves the right section (Recall@k, MRR, nDCG). Sentence pairs score STS.

## Models

Local, no key (first run downloads weights):

- Hash bag-of-words (baseline, not neural)
- MiniLM-L6-v2
- BGE small English
- Nomic embed text v1.5
- Jina v2 small English

Needs a key in `.env` or the **API keys** tab:

- OpenAI 3-small / 3-large (`OPENAI_API_KEY`)
- Cohere embed-english-v3 (`COHERE_API_KEY`)
- Google Gemini (`GOOGLE_API_KEY`)
- Voyage voyage-3 (`VOYAGE_API_KEY`)
- Hugging Face Inference (`HF_TOKEN`)
- Mistral embed (`MISTRAL_API_KEY`)

Extra setup:

- MPNet base v2 (`sentence-transformers`)
- Ollama nomic-embed-text (local Ollama server)

Download local FastEmbed weights in PowerShell (use Docker if that is how you run the UI):

```powershell
docker compose exec app python lab.py download
```

Without Docker:

```powershell
.\.venv\Scripts\python lab.py download
```

Then run the UI. MiniLM, BGE, Nomic, and Jina will already be cached.

```bash
pip install -r requirements.txt
streamlit run ui.py
```

Or `docker compose up --build` and open http://localhost:8701.

`lab.py` holds embedders and metrics. `ui.py` is the UI. `python lab.py minilm` runs a model from the command line.
=======
# Embedder
An Embedder Lab is all you need
>>>>>>> 5ecfa5df7f88df5d3bff9628917b7efdb686fb8a
