import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Project",
    layout="wide"
)

st.markdown("""
<style>

#MainMenu{
    visibility:hidden;
}

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

[data-testid="stFileUploaderDropzoneInstructions"] small{
    display:none;
}

.stChatMessage{
    border-radius:15px;
    padding:10px;
}

.citation-box{
    background:#f3f4f6;
    color:black;
    padding:12px;
    border-radius:10px;
    margin-bottom:10px;
    border-left:4px solid #22c55e;
}

.current-doc{
    background:#dbeafe;
    color:#1e3a8a;
    padding:12px;
    border-radius:10px;
    margin-bottom:20px;
    font-weight:600;
}

.loaded-doc{
    background:#dcfce7;
    color:#166534;
    padding:10px;
    border-radius:10px;
    margin-top:10px;
    font-weight:600;
}

.answer-box{
    background:#87CEEB;
    color:black;
    padding:18px;
    border-radius:12px;
    margin-bottom:15px;
    line-height:1.8;
    font-size:16px;
}

.model-badge{
    background:#1e293b;
    color:white;
    padding:4px 10px;
    border-radius:20px;
    display:inline-block;
    font-size:12px;
    margin-bottom:10px;
}

[data-testid="stMetric"]{
    border:1px solid #e5e7eb;
    border-radius:10px;
    padding:10px;
}

/* Floating model selector */
.model-selector{
    position:fixed;
    bottom:85px;
    right:30px;
    width:220px;
    z-index:999;
    background:white;
    padding:8px;
    border-radius:12px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.15);
}

</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama8b"

with st.sidebar:

    st.title("RAG Project")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file:

        if st.button(
            "Upload Document",
            use_container_width=True
        ):

            with st.spinner("Uploading..."):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "application/pdf"
                    )
                }

                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files
                )

                if response.status_code == 200:
                    st.success("Upload Successful")
                    st.rerun()

    try:

        status = requests.get(
            f"{BACKEND_URL}/status"
        ).json()

        if status.get("uploaded"):

            st.markdown(
                f"""
                <div class="loaded-doc">
                    {status['filename']}
                </div>
                """,
                unsafe_allow_html=True
            )

    except:
        st.error("Backend Offline")

st.title("Find answers for your questions")

try:

    status = requests.get(
        f"{BACKEND_URL}/status"
    ).json()

    if status.get("uploaded"):

        st.markdown(
            f"""
            <div class="current-doc">
                Current Document: {status['filename']}
            </div>
            """,
            unsafe_allow_html=True
        )

except:
    pass

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "user":

            st.markdown(message["content"])

        else:

            st.markdown(
                f"""
                <div class="model-badge">
                {message.get("model","Unknown")}
                </div>
                """,
                unsafe_allow_html=True
            )

            for answer in message.get("answers", []):

                st.markdown(
                    f"""
                    <div class="answer-box">
                    <b>Answer {answer['answer_rank']}</b><br><br>
                    {answer['answer']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                citations = answer.get("citations", [])

                if citations:

                    with st.expander(
                        f"View Citations - Answer {answer['answer_rank']}"
                    ):

                        for citation in citations:

                            st.markdown(
                                f"""
                                <div class="citation-box">
                                <b>Page:</b> {citation['page_range']}
                                <br><br>
                                {citation['matched_text']}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

# -----------------------------
# FLOATING MODEL SELECTOR
# -----------------------------
with st.container():

    st.markdown(
        '<div class="model-selector">',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Model",
        [
            "llama8b",
            "llama70b",
            "gptoss20b",
            "gptoss120b"
        ],
        key="selected_model"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

question = st.chat_input(
    f"Ask anything... ({st.session_state.selected_model})"
)

if question:

    current_model = st.session_state.selected_model

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        st.markdown(
            f"""
            <div class="model-badge">
            {current_model}
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.spinner("Searching document..."):

            payload = {
                "question": question,
                "model_name": current_model
            }

            response = requests.post(
                f"{BACKEND_URL}/query",
                json=payload
            )

            result = response.json()

            answers = result.get(
                "top_k_answers",
                []
            )

            if not answers:

                st.error("No answer found")

            else:

                for answer in answers:

                    st.markdown(
                        f"""
                        <div class="answer-box">
                        <b>Answer {answer['answer_rank']}</b><br><br>
                        {answer['answer']}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    citations = answer.get(
                        "citations",
                        []
                    )

                    if citations:

                        with st.expander(
                            f"View Citations - Answer {answer['answer_rank']}"
                        ):

                            for citation in citations:

                                st.markdown(
                                    f"""
                                    <div class="citation-box">
                                    <b>Page:</b> {citation['page_range']}
                                    <br><br>
                                    {citation['matched_text']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "answers": answers,
                        "model": current_model
                    }
                )