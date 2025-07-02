import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from difflib import get_close_matches

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Load your book data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Load and split your tagged descriptions
raw_documents = TextLoader("tagged_description.txt").load()

# Smarter text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

# Use Hugging Face sentence embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embeddings)

def retrieve_semantic_recommendations(query, category=None, tone=None,
                                      initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)

    matched_isbn13 = []
    for rec, _ in recs:
        possible_id = rec.page_content.split()[0]
        if possible_id.isdigit():
            matched_isbn13.append(int(possible_id))

    books_recs = books[books["isbn13"].isin(matched_isbn13)]

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category]

    if tone == "Happy":
        books_recs = books_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        books_recs = books_recs.sort_values(by="surprising", ascending=False)
    elif tone == "Angry":
        books_recs = books_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        books_recs = books_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        books_recs = books_recs.sort_values(by="sadness", ascending=False)

    return books_recs.head(final_top_k)


# Display results
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row.get("description", "")
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row.get("authors", "").split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_split[0] if authors_split else "Unknown"

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# UI dropdown options
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Gradio UI
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description:",
                                placeholder="e.g., A story about courage and friendship")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## 🔍 Recommended Books")
    output = gr.Gallery(label="Books", columns=4, rows=4, show_label=False)

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    dashboard.launch()



