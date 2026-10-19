import streamlit as st
import pandas as pd
import json
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

#testing
#test_query = st.sidebar.text_input("Test the vector DB:")
#if test_query:
    #res = collection.query(query_embeddings=embed([test_query]), n_results=3)
    #for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        #st.sidebar.write(f"**{meta['company']}** ({meta['date'][:10]}), similarity {1 - dist:.3f}")
        #st.sidebar.write(doc[:150])
        #st.sidebar.write(meta["url"])


def search_news(query):
    res = collection.query(query_embeddings=embed([query]), n_results=N_RESULTS)
    articles = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        articles.append(
            f"Company: {meta['company']} | Date: {meta['date'][:10]} | "
            f"Relevance: {1 - dist:.2f}\n"
            f"Article: {doc}\n"
            f"URL: {meta['url']}"
        )
    return "\n\n".join(articles) if articles else "No matching articles found."
 
 
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search the news article database and return the 3 most relevant "
                "articles. Use this for any question about news, companies, or "
                "events. Write the query as a descriptive search phrase, and "
                "include the company name when the user mentions one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search phrase to embed and match against the articles.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]
 
SYSTEM_PROMPT = """You are a news assistant for a large global law firm that monitors news about its clients.
 
- Report ONLY on articles returned by the search_news tool. Never add outside knowledge about events. If nothing relevant is found, say so.
- You MUST cite your sources and provide the link to click for more information.
- When asked for the most interesting news, return a ranked list and explain in one sentence why each story matters to a law firm monitoring its clients."""
 
use_advanced = st.sidebar.checkbox("Use advanced model")
model = MODELS["advanced" if use_advanced else "basic"]
st.sidebar.caption(f"Current model: {model}")
 

if "messages" not in st.session_state:
    st.session_state.messages = []
 
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 

if prompt := st.chat_input("Ask about the news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
 
    buffer = st.session_state.messages[-11:]
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + buffer
 
    with st.chat_message("assistant"):
        first = client.chat.completions.create(
            model=model,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = first.choices[0].message
 
        if msg.tool_calls:
            api_messages.append(msg.to_dict())
            for call in msg.tool_calls:
                args = json.loads(call.function.arguments)
                result = search_news(args["query"])
                api_messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
            )
            answer = st.write_stream(stream)
        else:
            answer = msg.content
            st.markdown(answer)
 
    st.session_state.messages.append({"role": "assistant", "content": answer})