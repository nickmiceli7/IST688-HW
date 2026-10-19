import streamlit as st
import pandas as pd
from pathlib import Path
import chromadb
from openai import OpenAI

st.title("HW7: News Information Bot")

EMBED_MODEL = "text-embedding-3-small"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

@st.cache_resource(show_spinner="Building the news database (first run only)...")
def get_news_collection():
    base = Path(__file__).parent
    chroma = chromadb.PersistentClient(path=str(base / "ChromaDB_for_HW7"))
    collection = chroma.get_or_create_collection(
        name="HW7NewsCollection",
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        df = pd.read_csv(base / "news.csv")
        df = df.drop_duplicates(subset=["company_name", "URL"]).reset_index(drop=True)
        for start in range(0, len(df), 100):
            batch = df.iloc[start:start + 100]
            texts = (batch["company_name"] + ": " + batch["Document"]).tolist()
            collection.add(
                ids=[f"news_{i}" for i in batch.index],
                documents=batch["Document"].tolist(),
                embeddings=embed(texts),
                metadatas=[
                    {"company": r.company_name, "date": r.Date,
                     "days_since_2000": int(r.days_since_2000), "url": r.URL}
                    for r in batch.itertuples()
                ],
            )
    return collection

collection = get_news_collection()

# --- Testing block (leave visible until we move on to the chatbot) ---
test_query = st.sidebar.text_input("Test the vector DB:")
if test_query:
    res = collection.query(query_embeddings=embed([test_query]), n_results=3)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        st.sidebar.write(f"**{meta['company']}** ({meta['date'][:10]}), similarity {1 - dist:.3f}")
        st.sidebar.write(doc[:150])
        st.sidebar.write(meta["url"])