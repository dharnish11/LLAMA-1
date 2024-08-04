import os
from dotenv import load_dotenv  # Import for loading environment variables
import streamlit as st
from pinecone import Pinecone

# Updated pinecone import
from llama_index.core.settings import Settings
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

load_dotenv()  # Load environment variables

api_key = "1d40e2a132c9493a9c5eadaa37c89d60"
azure_endpoint = "https://dr-ai-dev-1001.openai.azure.com/"
api_version = "2023-07-01-preview"

llm = AzureOpenAI(
  
)

embed_model = AzureOpenAIEmbedding(

)

# Create an instance of Pinecone client (using Pinecone class)
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
)

from llama_index.core import VectorStoreIndex

from llama_index.vector_stores.pinecone import PineconeVectorStore

print("**Streamlit LlamaIndex Documentation Helper**")


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pinecone_index = pc.Index(
        name="llamaindex-documentation-helper"
    )  # Use name argument
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, add_sparse_vector=True
    )

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    # service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
    # llm = OpenAI(model="gpt-4", temperature=0)
    Settings.llm = llm
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        vector_store_query_mode="hybrid",
        llm=llm,
    )
st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex docs")

# Initialize messages in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex docs?",
        }
    ]

if prompt := st.chat_input("Your question"):
    # Hard-code the additional prompt to include document link
    augmented_prompt = f"{prompt}  give short summary of documents in this format, Document Summary and Reference to the document:"
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )  # Display user's prompt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the augmented prompt instead of the raw user input
            response = st.session_state.chat_engine.chat(message=augmented_prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)

import os
import logging
import openai
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from transformers import ViTImageProcessor

# Load environment variables from .env file
load_dotenv()

# Access the API keys using the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Set the OpenAI API key

# Initialize the image processor (for future image search capability)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Configure service context
node_parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
Settings.llm = OpenAI()
Settings.embed_model = OpenAIEmbedding()

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def load_data_from_excel_files(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory_path, filename)
            df = pd.read_excel(file_path)
            for _, row in df.iterrows():
                content = row.to_dict()
                document_text = "\n".join(
                    [f"{key}: {value}" for key, value in content.items()]
                )
                documents.append(Document(text=document_text))
    return documents


# Function to load text data from local directory
def load_data_from_directory(directory_path):
    documents = load_data_from_excel_files(directory_path)
    reader = SimpleDirectoryReader(directory_path)
    documents += reader.load_data()
    return documents


# Load documents from the directory
documents = load_data_from_directory("./datasource")

# Initialize Pinecone
index_name = "llamaindex-documentation-helper"
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index(name=index_name)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create an index from the documents
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    show_progress=True,
)

print("Finished ingesting...")

# You can now use the 'index' object to perform queries and other operations

import os
from dotenv import load_dotenv  # Import for loading environment variables
import streamlit as st
from pinecone import Pinecone

# Updated pinecone import
from llama_index.core.settings import Settings
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import ViTImageProcessor

load_dotenv()  # Load environment variables

# Access the API keys using the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Set the OpenAI API key

# Initialize the image processor (for future image search capability)
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Configure service context
node_parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=20)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
embed_model = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
Settings.llm = OpenAI()
Settings.embed_model = OpenAIEmbedding()

# Create an instance of Pinecone client (using Pinecone class)
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
)

from llama_index.core import VectorStoreIndex

from llama_index.vector_stores.pinecone import PineconeVectorStore

print("**Streamlit LlamaIndex Documentation Helper**")


@st.cache_resource(show_spinner=False)
def get_index() -> VectorStoreIndex:
    pinecone_index = pc.Index(
        name="llamaindex-documentation-helper"
    )  # Use name argument
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, add_sparse_vector=True
    )

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager(handlers=[llama_debug])
    # service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
    # llm = OpenAI(model="gpt-4", temperature=0)
    Settings.llm = llm
    # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = get_index()
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        vector_store_query_mode="hybrid",
        llm=llm,
    )
st.set_page_config(
    page_title="Chat with LlamaIndex docs, powered by LlamaIndex",
    page_icon="",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

st.title("Chat with LlamaIndex docs")

# Initialize messages in session state
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex docs?",
        }
    ]

if prompt := st.chat_input("Your question"):
    # Hard-code the additional prompt to include document link
    augmented_prompt = f"{prompt}  give short summary of documents in this format, Document Summary and Reference to the document:"
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )  # Display user's prompt

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use the augmented prompt instead of the raw user input
            response = st.session_state.chat_engine.chat(message=augmented_prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
