import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import openai
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "docx", "txt"],
    )
    st.sidebar.markdown(
        """
### Useful Links:
- [GitHub Repository](https://github.com/leftnadal/Fullstack-GPT)
"""
    )
    api_key = st.text_input("Please write your OpenAI API Key...")


class ChatCallBackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    streaming=True,
    callbacks=[
        ChatCallBackHandler(),
    ],
    openai_api_key=api_key,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


@st.cache_resource(show_spinner="Embedding file....")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(
            file_content,
        )
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n", chunk_size=600, chunk_overlap=100
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"messages": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def pain_history():
    for message in st.session_state["messages"]:
        send_message(message["messages"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


memory = ConversationBufferMemory(return_messages=True)


def load_memory(_):
    x = memory.load_memory_variables({})
    return {"hitory": x["history"]}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
        Context: {context}
    """,
        ),
        ("human", "{question}"),
    ]
)


st.title("Document GPT2")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar
"""
)


if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save=False)
    pain_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(chat_history=load_memory),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

    else:
        st.session_state["messages"] = []
