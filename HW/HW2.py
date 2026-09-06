import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


# Show title and description.
st.title("MY url question answering")
st.write(
    "Enter a URL below and determine how you want it summarized – GPT will answer! "
)

# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets.OPENAI_API_KEY
anthropic_api_key = st.secrets.ANTHROPIC_API_KEY


language = st.sidebar.selectbox('Select language:', [
    'English',
    'Spanish',
    'French'
])

choice = st.sidebar.selectbox('Select one:', [
    'Summarize the document in 100 words',
    'Summarize the document in 2 paragraphs',
    'Summarize the document in 5 bullet points'
])

provider = st.sidebar.selectbox('Select provider:', [
    'OpenAI',
    'Anthropic'
])
client = None
key_valid = False

try: #I had claude help me to figure out how to nest the if statements inside the try statement - lines 49-53
    if provider == "OpenAI":
        client = OpenAI(api_key=openai_api_key)
    elif provider == "Anthropic":
        client = Anthropic(api_key=anthropic_api_key)

    client.models.list()
    key_valid = True
except Exception as e: #claude helped with the exception statement
    st.error(f"Invalid API key: {e}")


check = st.sidebar.checkbox('Use advanced model')
if provider == "OpenAI": #had 4 elif statements for 2x2 grid of possibilies between provider and check, claude suggested I use this shape instead
    model = "gpt-5-mini" if check else "gpt-5-nano"
elif provider == "Anthropic":
    model = "claude-sonnet-5" if check else "claude-haiku-4-5"

if key_valid:
    url = st.text_input("Enter a URL to summarize")

    if url:
        document = read_url_content(url)

        if document:
            if provider == 'OpenAI':
                messages = [
                    {
                        "role": "system",
                        "content": f"{choice}, in {language}"
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this document: {document}"
                    }
                ]

                # Generate an answer using the OpenAI API.
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                # Stream the response to the app using `st.write_stream`.
                st.write_stream(stream)
            elif provider == 'Anthropic':
                system = f"{choice}, in {language}"
                messages = [{'role': 'user', 'content': f"Summarize this document: {document}"}]
                with client.messages.stream(model=model, system=system, messages=messages) as stream:
                    st.write_stream(stream.text_stream)