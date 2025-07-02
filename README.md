# Semantic Book Recommender

A smart, AI-powered book recommendation system that understands **meaning**, not just keywords. Built using **Hugging Face**, **LangChain**, **Gradio**, and **Python**, this tool helps users discover books that match their thoughts, feelings, or themes — even vague ones.

---

## Features

-  **Semantic search** using text embeddings  
-  Powered by **Hugging Face sentence-transformers**  
-  Uses **LangChain + FAISS** for vector storage and querying  
-  Clean and easy-to-use **Gradio interface**  
-  Optional **OpenAI** support (via `.env`) — defaults to **free Hugging Face model**  
-  Fast, lightweight, and runs locally

---

## How It Works

1. The book descriptions are encoded using a transformer model like `all-MiniLM-L6-v2`.
2. A user types in a natural language query like:
   > "Books about self-discovery and ancient cultures"
3. The system calculates the **cosine similarity** between the query and book vectors.
4. Top matches are displayed via a simple Gradio web app.

---

## Tech Stack

| Tool         | Use                          |
|--------------|-------------------------------|
| Python       | Core development              |
| Hugging Face | Sentence embeddings           |
| LangChain    | Vector store/search logic     |
| FAISS        | Fast Approximate Nearest Neighbors search |
| Gradio       | Interactive front-end         |
| OpenAI (opt) | Embedding alternative (via `.env`) |

---

## File Structure

## semantic-book-recommender/
│
├── Book_Recommender.ipynb # Main logic and interface
├── Vector_Search.ipynb # Vector storage & retrieval (LangChain + FAISS)
├── Text_classification.ipynb # (Optional) additional classification
├── books_cleaned.csv # Dataset with book titles & descriptions
├── .gitignore # Ignores .env and unnecessary files
├── .env # (Ignored) for API keys
└── README.md # Project documentation

---

## How to Run Locally

1. **Clone the repo:**
```bash
git clone https://github.com/your-username/semantic-book-recommender.git
cd semantic-book-recommender
2. pip install -r requirements.txt
3. python app.py

---

## PS:
- If you want to use OpenAI embeddings instead of the default Hugging Face model:
  OPENAI_API_KEY=your_openai_key
