from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

if 'open_ai_client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.open_ai_client = OpenAI(api_key=api_key)

def add_to_collection(collection, text, file_name):
    client = st.session_state.open_ai_client
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )

    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )

def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_html = f.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    text = soup.get_text()
    return text

def load_html_to_collection(folder_path, collection):
    html_folder = Path(folder_path)
    for html_file in html_folder.glob("*.html"):
        text = extract_text_from_file(html_file)
        chunk1, chunk2 = chunk_text(text)
        add_to_collection(collection, chunk1, f"{html_file.name}_1")
        add_to_collection(collection, chunk2, f"{html_file.name}_2")

def chunk_text(text): #I chose this 'fixed size chunking' method because while there is the risk of losing out on proper context on the chunk-break, it is most efficient and for over 500 html files, I figured this would be the best method.Given the nature of the documents, I feel that chunk1 will be much more valuable than chunk2. 
    mid = len(text) // 2
    chunk1 = text[:mid]
    chunk2 = text[mid:]
    return chunk1, chunk2

@st.cache_resource #this debug was from claude and the following function
def get_chroma_collection():
    db_path = str(Path(__file__).parent / 'ChromaDB_for_HW')
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name='HW4Collection')
    if collection.count() == 0:
        load_html_to_collection('./su-orgs/', collection)
    return collection

collection = get_chroma_collection()


system_prompt = {'role': 'system', 'content': "Input a user's question and answer it. Then ask if they want to know more information. IF YES, give more information and AGAIN ask if they want more information. IF NO, ask what else you can help with. ALL OUTPUTS should be understandable by a 10 year old."}
st.title("HW4: Chatbot using RAG")

#topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

#if topic:
 #   client = st.session_state.open_ai_client
  #  response = client.embeddings.create(
   #     input=topic,
    #    model='text-embedding-3-small'
    #)

    #query_embedding = response.data[0].embedding

    #results = collection.query(
     #   query_embeddings = [query_embedding],
      #  n_results = 10
    #)

    #st.subheader(f'Results for: {topic}')

    #for i in range(len(results['documents'][0])):
     #   doc = results['documents'][0][i]
      #  doc_id = results['ids'][0][i]

       # st.write(f'**{i+1}. {doc_id}**')
#else:
    #st.info('Enter a topic in the sidebar to seach the collection')

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I help you?'}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message('user'):
            st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    client = st.session_state.open_ai_client
    response = client.embeddings.create(
       input=prompt,
       model='text-embedding-3-small'
    )

    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = 3
    )


    relevant_doc = ''
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        doc_id = results['ids'][0][i]
        relevant_doc += f"{doc_id}: {doc} \n"

    dynamic_system_prompt = {'role': 'system', 'content': system_prompt['content'] + "\n The following text is your RAG context. You MUST end every single response with a new line reading exactly: 'Source(s): ' followed by a comma-separated list of the exact document filenames you used from the provided context. If you did not use any retrieved documents to answer, write 'Source(s): none'. \n" + relevant_doc}


    buffer_messages = st.session_state.messages[-10:]
    passed_messages = []
    passed_messages.append(dynamic_system_prompt)
    passed_messages.extend(buffer_messages)

    stream = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=passed_messages,
        stream=True)

    with st.chat_message('assistant'):
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})

