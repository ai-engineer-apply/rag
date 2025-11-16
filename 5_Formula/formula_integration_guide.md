# Formula: Integrating Semantic Search into AI Chat Queries

This document provides a step-by-step guide on how to port the existing semantic search functionality to a new project and integrate it as a Retrieval Augmented Generation (RAG) component for AI-based chat queries.

## 1. Understanding the Core Components

The semantic search system relies on three main components:
-   **Sentence Transformer Model**: Converts text (documents and queries) into numerical vector embeddings.
-   **FAISS Index**: Stores these embeddings and allows for fast similarity search.
-   **File Paths Mapping**: A simple text file (`filepaths.txt`) that maps the index IDs back to their original document paths.

## 2. Step-by-Step Integration Guide

### Step 2.1: Copy Core Files to Your New Project

Start by copying the essential files from this project into your new AI chat project's directory structure.

1.  **`index.py`**: This script is used to build or update your vector index.
2.  **`search.py`**: This script contains the logic for performing a semantic search. We will modify this to be a reusable function.
3.  **`requirements.txt`**: This file lists all necessary Python dependencies.
4.  **`docs/` folder**: Create a `docs/` folder in your new project and place all your markdown documents (or any other text-based documents you want to search) inside it.

### Step 2.2: Install Dependencies

Navigate to your new project's root directory in your terminal and install the required Python packages. It's highly recommended to use a virtual environment.

```bash
# Create a virtual environment (if you haven't already)
python3 -m venv venv_chat_project

# Activate the virtual environment
source venv_chat_project/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2.3: Initial Indexing of Your Documents

Before you can search, you need to create the FAISS index and the file paths map for your documents.

```bash
# Run the indexer (assuming your documents are in ./docs)
python3 index.py --folder ./docs
```

This will generate `faiss_index.bin` and `filepaths.txt` in your project's root directory. These files will be used by your chat application.

### Step 2.4: Modify `search.py` for Programmatic Access

To integrate the search functionality into your chat application, you'll want to call it as a function rather than running it as a standalone script.

**Conceptual Change to `search.py`:**

```python
# /path/to/your/new_project/search_module.py (or similar)

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os # Added for path joining

# Global variables to load resources once
_index = None
_filepaths = None
_model = None

def _load_resources():
    """Loads FAISS index, file paths, and sentence transformer model."""
    global _index, _filepaths, _model
    if _index is None:
        print("Loading FAISS index from 'faiss_index.bin'...")
        _index = faiss.read_index('faiss_index.bin')
    if _filepaths is None:
        print("Loading file paths from 'filepaths.txt'...")
        with open('filepaths.txt', 'r', encoding='utf-8') as f:
            _filepaths = [line.strip() for line in f.readlines()]
    if _model is None:
        print("Loading sentence transformer model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(query_text: str, k: int = 5) -> list[tuple[str, float]]:
    """
    Performs a semantic search and returns a list of (filepath, distance) tuples.
    """
    _load_resources() # Ensure resources are loaded

    query_embedding = _model.encode([query_text], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype('float32')

    distances, indices = _index.search(query_embedding, k)

    results = []
    if len(indices[0]) > 0:
        for i, idx in enumerate(indices[0]):
            if idx < len(_filepaths):
                results.append((_filepaths[idx], distances[0][i]))
            else:
                print(f"Warning: Index {idx} is out of bounds for filepaths list.")
    return results

# The original main() function can be removed or kept for standalone testing
# if __name__ == "__main__":
#     # ... original main() logic ...
```

### Step 2.5: Integrate with Your AI Chat Query Logic (RAG Pattern)

Now, in your main AI chat application script (e.g., `chat_app.py`), you can import and use the `semantic_search` function. This implements a basic Retrieval Augmented Generation (RAG) pattern.

```python
# /path/to/your/new_project/chat_app.py

from search_module import semantic_search # Assuming you saved the modified search.py as search_module.py
import os

# Placeholder for your AI model interaction (e.g., OpenAI, Gemini API)
def get_ai_response(prompt: str) -> str:
    # This is where you would call your AI model API
    # Example: response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=150)
    # For demonstration, we'll just echo the prompt
    return f"AI Response (based on context):\n{prompt}\n---"

def chat_with_rag(user_query: str):
    print(f"User: {user_query}")

    # 1. Retrieve relevant documents
    search_results = semantic_search(user_query, k=3) # Get top 3 relevant docs

    context_text = ""
    if search_results:
        print("\nRetrieving context from relevant documents:")
        for filepath, distance in search_results:
            print(f"- {filepath} (Distance: {distance:.4f})")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    context_text += f.read() + "\n\n---\n\n" # Append document content
            except Exception as e:
                print(f"Could not read {filepath}: {e}")
    else:
        print("No relevant documents found for context.")

    # 2. Augment the prompt with retrieved context
    if context_text:
        prompt = (
            f"Based on the following context, answer the user's question:\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question: {user_query}\n\n"
            f"Answer:"
        )
    else:
        prompt = f"User Question: {user_query}\n\nAnswer:"

    # 3. Generate response using the AI model
    ai_response = get_ai_response(prompt)
    print(f"\nAI: {ai_response}")

if __name__ == "__main__":
    # Example usage
    chat_with_rag("what is semantic search?")
    print("\n" + "="*50 + "\n")
    chat_with_rag("tell me about databases")
    print("\n" + "="*50 + "\n")
    chat_with_rag("who is mehmet?")
```

### Step 2.6: Run Your Chat Application

```bash
# Make sure your virtual environment is active
source venv_chat_project/bin/activate

# Run your chat application
python3 chat_app.py
```

This setup allows your AI chat queries to first retrieve relevant information from your local markdown documents, and then use that information to provide more accurate and context-aware responses.

## Step 2.7: Worked example — how this repository indexed and searched

The steps below document a concrete run I performed on this repository so you can reproduce the process exactly. It includes the environment creation, dependency installation, indexing, sample searches, and small repository housekeeping I performed.

1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade packaging tools and install the repository dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r 6_Symbols/requirements.txt
```

Notes: installing `sentence-transformers`, `torch`, and `faiss-cpu` will download model files and native wheels. The SentenceTransformers model used by the scripts is `all-MiniLM-L6-v2` and will be cached by Hugging Face.

3. Create the FAISS index from markdown files (indexing)

```bash
# From repo root; pass the folder to scan (I scanned the repo root to include all .md files)
python3 6_Symbols/index.py --folder .
```

What this did:
- Scanned the passed folder recursively for `.md` files.
- Extracted inline text from Markdown using `markdown-it-py`.
- Created embeddings for each document using `SentenceTransformer('all-MiniLM-L6-v2')`.
- Built a FAISS `IndexFlatL2` (wrapped with `IndexIDMap`) and stored it to `faiss_index.bin`.
- Wrote `filepaths.txt` mapping index ids to file paths.

Files produced:
- `faiss_index.bin` — binary FAISS index (generated)
- `filepaths.txt` — newline list mapping index ids → document paths

4. Run sample semantic searches

```bash
# Example queries I ran
python3 6_Symbols/search.py --query "retrieval augmented generation"
python3 6_Symbols/search.py --query "people"
python3 6_Symbols/search.py --query "who is"
```

Each search loads `faiss_index.bin`, `filepaths.txt`, and the `SentenceTransformer` model, encodes the query, and returns the top-k matching document paths and distances.

Sample person-related results found during the runs (top unique matches):
- `6_Symbols/docs/jane.md`
- `6_Symbols/docs/john.md`
- `6_Symbols/docs/mehmet.md`
- `2_Real/README.md`
- `4_UI/README.md`
- `7_Semblance/README.md`

5. Re-index after adding documents

If you add or update markdown files, re-run the indexer to refresh `faiss_index.bin` and `filepaths.txt`:

```bash
python3 6_Symbols/index.py --folder .
```

6. Repository housekeeping I performed (optional but recommended)

- Added common environment and generated file patterns to `.gitignore` so large or environment-specific files are not committed. Notable entries added:

```gitignore
.venv
.venv/
venv/
faiss_index.bin
filepaths.txt
__pycache__/
*.py[cod]
```

- If the virtual environment directory was already committed, remove it from the index and commit the change:

```bash
git rm -r --cached .venv
git commit -m "Remove .venv from repo and update .gitignore"
git push
```

7. Programmatic integration (quick recap)

- To reuse the search behavior inside another service (for RAG), load the FAISS index and model once and expose a `semantic_search(query, k)` function that:
    - encodes the query with the SentenceTransformer,
    - runs `index.search(query_embedding, k)`,
    - maps returned IDs to file paths using `filepaths.txt`, and
    - returns (path, distance) pairs.  (See the conceptual `search_module.py` example earlier in this guide.)

8. Next steps you can take

- Create a small `chat_app.py` that calls the `semantic_search` function, retrieves the document contents for the top results, and uses them as context for your LLM prompt.
- Add snippet extraction to the search results so each hit returns a short excerpt for quick preview.
- Persist embeddings and metadata in a more featureful store (e.g., Qdrant or Milvus) if you need advanced filtering or larger scale.

If you'd like, I can add a short `6_Symbols/README.md` with the exact commands, or modify `6_Symbols/search.py` to return snippets alongside file paths and distances. Tell me which and I'll implement it.

