from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st


# 대화 프롬프트 정의
chat_prompt = ChatPromptTemplate.from_template(
    """
    The following is a conversation between a user and an AI assistant. 
    The assistant provides helpful and accurate information based ONLY on the given website context.

    Conversation history:
    {history}
    
    Context:
    {context}

    User: {question}
    AI:
    """
)

# 대화 메모리 초기화
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True)

memory = st.session_state["memory"]


def parse_page(soup):
    """페이지 내용을 파싱하여 텍스트만 추출."""
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    """Sitemap에서 URL을 로드하고 텍스트를 처리."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/vectorixe\/).*",
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 1
    docs = loader.load_and_split(text_splitter=splitter)
    vector_stores = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_stores.as_retriever()


def chatbot_response(question, retriever):
    """질문에 대한 챗봇 응답 생성."""
    # 문서에서 컨텍스트 검색
    context_docs = retriever.get_relevant_documents(question)
    context = "\n".join(doc.page_content for doc in context_docs)

    # 대화 기록 가져오기
    history = "\n".join(
        f"User: {msg.content}" if msg.type == "human" else f"AI: {msg.content}"
        for msg in memory.chat_memory.messages
    )

    # 대화 프롬프트를 생성하고 응답 생성
    chat_chain = chat_prompt | llm
    response = chat_chain.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )

    # 대화 기록에 추가
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response.content)

    return response.content


# Streamlit UI 설정
st.set_page_config(
    page_title="SiteGPT Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("SiteGPT Chatbot")

# 사이드바 URL 입력
with st.sidebar:
    url = st.text_input(
        "Enter the sitemap URL:",
        placeholder="https://example.com/sitemap.xml",
    )
    st.sidebar.markdown(
        """
    ### Useful Links:
    - [GitHub Repository](https://github.com/leftnadal/Fullstack-GPT)
    """
    )
    api_key = st.text_input("Please write your OpenAI API Key...")

if url:
    if ".xml" not in url:
        st.sidebar.error("Please enter a valid Sitemap URL.")
    else:
        # 사이트 데이터 로드
        retriever = load_website(url)
        st.write("Website data loaded successfully! Start chatting below.")

        # 대화 기록 출력
        st.markdown("### Conversation")
        for msg in memory.chat_memory.messages:
            if msg.type == "human":
                st.chat_message("user").write(msg.content)
            else:
                st.chat_message("assistant").write(msg.content)

        # 사용자 입력창
        st.markdown("---")  # 구분선 추가
        user_query = st.text_input(
            "Your question:",
            placeholder="Ask me anything about the website...",
            key="chat_input",  # 세션 상태를 유지하기 위한 키
        )

        if user_query:
            # 챗봇 응답 생성
            bot_response = chatbot_response(user_query, retriever)
            st.write(bot_response)

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=api_key,
)
