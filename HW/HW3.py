import streamlit as st
from openai import OpenAI
import tiktoken
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

st.session_state.setdefault('url1_data', {'url': None, 'text': None}) #claude helped with .setdefault
st.session_state.setdefault('url2_data', {'url': None, 'text': None}) 

openai_api_key = st.secrets.OPENAI_API_KEY
anthropic_api_key = st.secrets.ANTHROPIC_API_KEY

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
token_based_buffer = 2000

st.title("MY Lab3 question answering chatbot")
st.markdown(f"Token buffer: {token_based_buffer}")
st.write('This chatbot will answer your questions based on the URL(s) you provide. It will remember your chats up to 2000 tokens.')

url1_input = st.text_input("Enter URL1")
url2_input = st.text_input("Enter URL2")

if not url1_input:
    st.session_state.url1_data = {'url': None, 'text': None}
elif url1_input == st.session_state.url1_data['url']:
    pass
else:
    fetched_text = read_url_content(url1_input)
    if fetched_text is None:
        st.error(f"Couldn't read content from URL 1: {url1_input}")
        st.session_state.url1_data = {'url': None, 'text': None}
    else:
        st.session_state.url1_data['url'] = url1_input
        st.session_state.url1_data['text'] = fetched_text

if not url2_input:
    st.session_state.url2_data = {'url': None, 'text': None}
elif url2_input == st.session_state.url2_data['url']:
    pass 
else:
    fetched_text = read_url_content(url2_input)
    if fetched_text is None:
        st.error(f"Couldn't read content from URL 2: {url2_input}")
        st.session_state.url2_data = {'url': None, 'text': None}
    else:
        st.session_state.url2_data['url'] = url2_input
        st.session_state.url2_data['text'] = fetched_text

provider = st.sidebar.selectbox('Select provider:', [
    'OpenAI',
    'Anthropic'
])

if 'provider' not in st.session_state or provider != st.session_state.provider: #claude suggested I use the or statement and it seemed more efficient than my previous approach
    st.session_state.provider = provider
    try:
        if provider == "OpenAI":
            client = OpenAI(api_key=openai_api_key)
        elif provider == "Anthropic":
            client = Anthropic(api_key=anthropic_api_key)
        client.models.list()
        st.session_state.client = client
        st.session_state.key_valid = True
    except Exception as e:
        st.error(f"Invalid API key: {e}")
        st.session_state.key_valid = False

if provider == "OpenAI":
    model = "gpt-6-astra"
elif provider == "Anthropic":
    model = "claude-opus-5" 


system_prompt_text = "Input a user's question and answer it. Then ask if they want to know more information. IF YES, give more information and AGAIN ask if they want more information. IF NO, ask what else you can help with. ALL OUTPUTS should be understandable by a 10 year old."

if st.session_state.url1_data['text']:
    system_prompt_text += f"\n\nContent from URL 1 ({st.session_state.url1_data['url']}):\n{st.session_state.url1_data['text']}"

if st.session_state.url2_data['text']:
    system_prompt_text += f"\n\nContent from URL 2 ({st.session_state.url2_data['url']}):\n{st.session_state.url2_data['text']}"

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'How can I help you?'}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

client = st.session_state.get('client') #claude suggested .get()

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    #buffer_messages = st.session_state.messages[-4:]
    #^ used for message count conversation buffer

    buffer_messages = []
    total_tokens = 0

    for msg in reversed(st.session_state.messages):
        token_count = len(encoding.encode(msg['content']))
        if total_tokens + token_count < token_based_buffer:
            buffer_messages.append(msg)
            total_tokens = total_tokens + token_count
        else:
            break


    buffer_messages.reverse()
    if provider == 'OpenAI':
        passed_messages = [{'role': 'system', 'content': system_prompt_text}] + buffer_messages
        stream = client.chat.completions.create(
            model=model,
            messages=passed_messages,
            stream=True)
        with st.chat_message('assistant'):
            response = st.write_stream(stream)
    elif provider == 'Anthropic':
        with client.messages.stream(
            model=model,
            max_tokens=1000,
            system=system_prompt_text,
            messages=buffer_messages
        ) as stream:
            with st.chat_message('assistant'):
                response = st.write_stream(stream.text_stream)

    st.session_state.messages.append({"role": "assistant", "content": response})


