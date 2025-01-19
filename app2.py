import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import json

# 난이도 선택을 반영한 프롬프트
question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role-playing as a teacher.

    Based ONLY on the following context, make {difficulty} 3 questions to test the user's knowledge about the text.

    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    Question examples:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue

    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi|Manila|Beirut

    Question: When was Avatar released?
    Answers: 2007|2001|2009|1998

    Question: Who was Julius Caesar?
    Answers: A Roman Emperor|Painter|Actor|Model

    Your turn!

    Context: {context}
""",
        )
    ]
)

function_calling = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"},
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


@st.cache_data(show_spinner="File is uploaded...")
def handle_file(file):
    upload_content = file.read()
    file_path = f"./.cache/quiz_functioncalling_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(upload_content)
    loader = UnstructuredFileLoader(file_path)
    load_docs = loader.load()
    return load_docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def search_wiki(topic):
    retriever = WikipediaRetriever(top_k_results=5)
    wikipedia_docs = retriever.get_relevant_documents(topic)
    st.session_state["click"] = False
    return "\n\n".join(page.page_content for page in wikipedia_docs)


@st.cache_data(show_spinner="Generating Quiz...")
def generate_quiz(docs, difficulty):
    chain = question_prompt | llm
    response = chain.invoke({"context": docs, "difficulty": difficulty})
    response_json = json.loads(response.additional_kwargs["function_call"]["arguments"])
    return response_json


if "click" not in st.session_state:
    st.session_state["click"] = False

with st.sidebar:
    choice = st.selectbox("Choose what you want to use", ["Your own file", "Wikipedia"])
    difficulty = st.selectbox(
        "Select difficulty level", ["easy", "medium", "hard"]
    )  # 추가
    docs = None
    if choice == "Your own file":
        file = st.file_uploader(
            "Upload a .docx, .txt or .pdf file", type=["pdf", "txt", "docx"]
        )
        if file:
            load_docs = handle_file(file)
            if st.session_state.get("file") != file.name:
                st.session_state["file"] = file.name
                st.session_state["click"] = False
            docs = load_docs[0].page_content
    else:
        topic = st.text_input("Name of the Article")
        if topic:
            docs = search_wiki(topic)
            if st.session_state.get("topic") != topic:
                st.session_state["topic"] = topic
                st.session_state["click"] = False
    st.sidebar.markdown(
        """
### Useful Links:
- [GitHub Repository](https://github.com/leftnadal/Fullstack-GPT)
"""
    )
    api_key = st.text_input("Please write your OpenAI API Key...")


llm = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
).bind(
    function_call={"name": "create_quiz"},
    functions=[function_calling],
    openai_api_key=api_key,
)

st.set_page_config(page_title="QuizFunctionCallingGPT", page_icon="❓")
st.title("QuizFunctionCallingGPT")

st.markdown(
    """
    Welcome to QuizFunctionCallingGPT.

    This page will use a 'function calling' in OpenAI.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
)


if docs:
    st.divider()
    placeholder = st.empty()
    if not st.session_state["click"]:
        button = placeholder.button("Start Quiz")
        if button:
            st.session_state["click"] = True
            placeholder.empty()

    if st.session_state["click"]:
        quiz_data = generate_quiz(docs, difficulty)

        with st.form("quiz"):
            user_score = 0
            total_questions = len(quiz_data["questions"])
            for question in quiz_data["questions"]:
                value = st.radio(
                    label=question["question"],
                    options=[answer["answer"] for answer in question["answers"]],
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    user_score += 1
                elif value is not None:
                    correct_answer = next(
                        answer["answer"]
                        for answer in question["answers"]
                        if answer["correct"]
                    )
                    st.error(f"Wrong! The correct answer is '{correct_answer}'")

            submitted = st.form_submit_button("Submit")
            if submitted:
                if user_score == total_questions:
                    st.balloons()
                    st.success(
                        f"🎉 Congratulations! You got a perfect score: {user_score}/{total_questions} 🎉"
                    )
                    st.markdown(
                        """
                        <style>
                        .stSuccess {
                            font-size: 20px;
                            font-weight: bold;
                            color: #2E8B57;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        f"Your score is {user_score}/{total_questions}. Try again!"
                    )
                    st.session_state["click"] = False
