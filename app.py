import os
import re
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
import pymupdf
pymupdf.set_messages(path=os.devnull)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="AIVLE School 학습 도우미",
    page_icon="📘",
    layout="wide"
)

BASE_DIR = Path.cwd()
CACHE_DIR = BASE_DIR / "aivle_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# API KEY 로드
# =========================================================
def load_api_key():
    # 1) 이미 환경변수에 있으면 사용
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")

    # 2) 현재 폴더의 api_key.txt
    candidate_files = [
        BASE_DIR / "api_key.txt",
        Path("/content/drive/MyDrive/project04/api_key.txt")
    ]

    for filepath in candidate_files:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            return os.getenv("OPENAI_API_KEY")

    return None

OPENAI_API_KEY = load_api_key()

# =========================================================
# CSS
# =========================================================
def inject_css():
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: "Pretendard", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        background: linear-gradient(180deg, #f8fbfb 0%, #f5fbfa 100%);
        color: #1f2937;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    /* 공통 카드 */
    .soft-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #d9efec;
        border-radius: 24px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }

    .left-panel {
        background: linear-gradient(180deg, #f3fbfa 0%, #eef9f7 100%);
        border: 1px solid #d8efec;
        border-radius: 28px;
        padding: 24px 18px;
        min-height: 930px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.04);
    }

    .app-logo {
        font-size: 40px;
        margin-bottom: 8px;
    }

    .app-title {
        font-size: 20px;
        font-weight: 800;
        line-height: 1.3;
        color: #111827;
        margin-bottom: 18px;
    }

    .nav-item {
        background: #ffffff;
        border: 1px solid #dcefee;
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 700;
        color: #1f2937;
    }

    .nav-item.active {
        background: #dff6f2;
        border: 1px solid #cfece7;
        color: #0f766e;
    }

    .mini-card-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 10px;
    }

    .progress-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-top: 8px;
    }

    .progress-circle {
        width: 62px;
        height: 62px;
        border-radius: 50%;
        border: 6px solid #dff4ef;
        border-top-color: #3ac7b6;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        color: #0f766e;
        background: white;
        font-size: 14px;
    }

    .robot-box {
        margin-top: 28px;
        text-align: center;
        padding: 28px 12px 20px 12px;
        border-top: 1px solid #dbeceb;
    }

    .robot-emoji {
        font-size: 72px;
        line-height: 1;
    }

    .robot-caption {
        margin-top: 10px;
        font-size: 14px;
        color: #4b5563;
        line-height: 1.6;
    }

    .main-title {
        font-size: 34px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 4px;
    }

    .main-subtitle {
        font-size: 16px;
        color: #475569;
        margin-bottom: 16px;
    }

    .chip-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 18px;
    }

    .hero-card {
        background: linear-gradient(180deg, #f6fffd 0%, #eefaf7 100%);
        border: 1px solid #d7efea;
        border-radius: 26px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }

    .hero-title {
        font-size: 24px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 8px;
    }

    .hero-text {
        font-size: 16px;
        color: #475569;
        line-height: 1.6;
    }

    .user-msg {
        width: fit-content;
        max-width: 76%;
        margin-left: auto;
        background: #dff6f2;
        border-radius: 20px;
        padding: 14px 18px;
        font-size: 16px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 14px;
        border: 1px solid #caece6;
    }

    .assistant-card {
        background: rgba(255,255,255,0.95);
        border: 1px solid #dcefee;
        border-radius: 24px;
        padding: 22px 22px 14px 22px;
        box-shadow: 0 10px 26px rgba(0,0,0,0.04);
    }

    .answer-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 17px;
        color: #111827;
    }

    .answer-foot {
        margin-top: 14px;
        padding-top: 12px;
        border-top: 1px solid #edf2f7;
        color: #475569;
        font-size: 14px;
    }

    .section-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: #f2fcfa;
        border: 1px solid #d8efea;
        color: #0f766e;
        font-weight: 700;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    .right-panel {
        background: rgba(255,255,255,0.92);
        border: 1px solid #d9efec;
        border-radius: 28px;
        padding: 22px;
        min-height: 930px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.04);
    }

    .right-title {
        font-size: 22px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 4px;
    }

    .right-sub {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 18px;
    }

    .doc-card {
        background: #fbfefe;
        border: 1px solid #d7efea;
        border-radius: 20px;
        padding: 16px;
        margin-bottom: 14px;
    }

    .doc-title {
        font-size: 15px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 6px;
    }

    .doc-snippet {
        font-size: 13px;
        color: #475569;
        line-height: 1.55;
    }

    .doc-page {
        float: right;
        font-size: 13px;
        font-weight: 700;
        color: #64748b;
    }

    .bottom-input-wrap {
        background: rgba(255,255,255,0.95);
        border: 1px solid #d9efec;
        border-radius: 22px;
        padding: 12px 16px;
        margin-top: 16px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.04);
    }

    /* 버튼 */
    .stButton > button {
        border-radius: 999px;
        border: 1px solid #d7ece8;
        background: white;
        color: #1f2937;
        font-weight: 700;
        padding: 0.6rem 1rem;
    }

    .stButton > button:hover {
        border-color: #35c7b2;
        color: #0f766e;
    }

    /* 파일 업로더 */
    .stFileUploader {
        background: white;
        border-radius: 18px;
        padding: 10px;
        border: 1px dashed #bcded9;
    }

    /* 데이터프레임/테이블 */
    table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 14px !important;
    }

    th {
        background-color: #eefaf7 !important;
        color: #0f766e !important;
        font-weight: 800 !important;
        text-align: center !important;
    }

    th, td {
        border: 1px solid #d9ece8 !important;
        padding: 10px !important;
    }

    /* expander */
    .streamlit-expanderHeader {
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_docs" not in st.session_state:
    st.session_state.latest_docs = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "doc_name" not in st.session_state:
    st.session_state.doc_name = "AIVLE School 통합 백서.pdf"

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

# =========================================================
# RAG 함수
# =========================================================
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def get_file_hash(file_bytes: bytes):
    return hashlib.md5(file_bytes).hexdigest()

def extract_section_title(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "근거 문서"
    first = lines[0]
    first = re.sub(r"^#+\s*", "", first)
    first = re.sub(r"^\*\*|\*\*$", "", first)
    return first[:40]

@st.cache_resource(show_spinner=False)
def build_rag(pdf_path: str, file_hash: str):
    # 문서 로드
    loader = PyMuPDF4LLMLoader(pdf_path)
    docs = loader.load()

    # 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # 임베딩
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # 벡터 저장소
    persist_dir = str(CACHE_DIR / "chroma_db")
    vectorstore = Chroma(
        collection_name=f"aivle_{file_hash[:12]}",
        embedding_function=embedding,
        persist_directory=persist_dir
    )

    # 이미 추가된 문서가 없을 때만 추가
    existing = vectorstore.get()
    existing_ids = set(existing.get("ids", [])) if existing else set()

    ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    new_docs = []
    new_ids = []

    for doc, doc_id in zip(chunks, ids):
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        vectorstore.add_documents(new_docs, ids=new_ids)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )

    return retriever, llm

def ensure_rag_ready(uploaded_file):
    if uploaded_file is None:
        return

    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    file_hash = get_file_hash(file_bytes)

    if st.session_state.uploaded_hash == file_hash and st.session_state.vector_ready:
        return

    temp_dir = CACHE_DIR / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf_path = temp_dir / uploaded_file.name
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

    with st.spinner("백서를 분석하고 RAG 파이프라인을 준비하고 있습니다..."):
        retriever, llm = build_rag(str(temp_pdf_path), file_hash)

    st.session_state.retriever = retriever
    st.session_state.llm = llm
    st.session_state.vector_ready = True
    st.session_state.doc_name = uploaded_file.name
    st.session_state.uploaded_hash = file_hash

def make_prompt(context, history, question):
    system_prompt = f"""
당신은 'AIVLE School 학습 도우미'입니다.
반드시 제공된 백서 근거를 기반으로 한국어로 답변하세요.

[답변 원칙]
1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답하세요.
3. 비교 질문이면 가능한 경우 마크다운 표로 정리하세요.
4. 결석 처리 기준, 수료 기준, 제적 기준은 구분해서 설명하세요.
5. 숫자, 기간, 조건은 문서 그대로 유지하세요.
6. 답변은 이해하기 쉽게 구조화하세요.
7. 마지막에는 핵심 한 줄 요약을 한 문장으로 적어주세요.

[대화 이력]
{history}

[문서 근거]
{context}
"""
    human_prompt = question
    return [
        ("system", system_prompt),
        ("human", human_prompt)
    ]

def ask_rag(question: str):
    if not st.session_state.vector_ready:
        return "먼저 PDF 백서를 업로드해 주세요.", []

    docs = st.session_state.retriever.invoke(question)
    context = format_docs(docs)

    history_text = ""
    for m in st.session_state.messages[-4:]:
        history_text += f"{m['role']}: {m['content']}\n"

    messages = make_prompt(context, history_text, question)
    answer = st.session_state.llm.invoke(messages).content

    return answer, docs

def run_question(question: str):
    if not question:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    answer, docs = ask_rag(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "docs": docs
    })

    st.session_state.latest_docs = docs

# =========================================================
# 왼쪽 패널
# =========================================================
left_col, center_col, right_col = st.columns([1.1, 4.2, 1.9], gap="large")

with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.markdown('<div class="app-logo">📘</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">AIVLE School<br>학습 도우미</div>', unsafe_allow_html=True)

    st.markdown('<div class="nav-item active">🏠 홈</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">💬 대화</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📄 문서 관리</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🔖 추천 질문</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="mini-card-title">백서 학습 현황</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="progress-wrap">
        <div class="progress-circle">72%</div>
        <div>
            <div style="font-weight:800; color:#111827;">핵심 주제 학습 중</div>
            <div style="color:#475569; font-size:14px;">(8 / 11)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="robot-box">
        <div class="robot-emoji">🤖</div>
        <div class="robot-caption">
            백서를 바탕으로 궁금한 내용을 빠르게 정리하고,<br>
            근거와 함께 이해할 수 있도록 도와드려요.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 가운데 메인
# =========================================================
with center_col:
    st.markdown('<div class="main-title">AIVLE School 학습 도우미 ✦</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">AIVLE School 통합 백서를 기반으로 교육과정, 프로젝트, 취업 등 궁금한 내용을 빠르고 정확하게 알려드려요.</div>',
        unsafe_allow_html=True
    )

    # 빠른 메뉴 버튼
    chip_cols = st.columns(5)
    with chip_cols[0]:
        show_upload = st.button("⬆ 백서 업로드", use_container_width=True)
    with chip_cols[1]:
        if st.button("📚 AI-DX 트랙 비교", use_container_width=True):
            st.session_state.pending_question = "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이를 비교해줘."
    with chip_cols[2]:
        if st.button("🗓 출결·수료 기준", use_container_width=True):
            st.session_state.pending_question = "AIVLE School의 출결 기준, 수료 기준, 제적 기준을 구분해서 설명해줘."
    with chip_cols[3]:
        if st.button("📁 프로젝트 안내", use_container_width=True):
            st.session_state.pending_question = "미니프로젝트와 빅프로젝트의 역할과 특징을 설명해줘."
    with chip_cols[4]:
        if st.button("💼 취업 준비", use_container_width=True):
            st.session_state.pending_question = "AIVLE School에서 제공하는 취업 지원 프로그램을 정리해줘."

    # 업로드 영역
    if show_upload or not st.session_state.vector_ready:
        st.markdown('<div class="hero-card">', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">AIVLE School 학습 도우미에 오신 것을 환영합니다! 👋</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="hero-text">AIVLE School 통합 백서를 기반으로 궁금한 내용을 질문해보세요.<br>문서의 근거를 확인하며 더 깊이 있게 학습할 수 있어요.</div>',
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("PDF 백서 업로드", type=["pdf"], label_visibility="collapsed")
        if uploaded_file is not None:
            ensure_rag_ready(uploaded_file)
            st.success(f"'{uploaded_file.name}' 업로드 및 분석이 완료되었습니다.")
        st.markdown('</div>', unsafe_allow_html=True)

    # pending question 실행
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        run_question(q)

    # 기본 예시 표시 or 실제 대화 표시
    if not st.session_state.messages:
        st.markdown('<div class="user-msg">AI 개발자 트랙과 DX 컨설턴트 트랙의 차이를 비교해줘.</div>', unsafe_allow_html=True)

        st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
        st.markdown('<div class="answer-head">🟢 AIVLE 학습 도우미</div>', unsafe_allow_html=True)

        demo_md = """
AI 개발자 트랙과 DX 컨설턴트 트랙은 **학습 목적, 교육과정, 핵심 역량, 진로 방향**이 다릅니다.

| 구분 | AI 개발자 트랙 | DX 컨설턴트 트랙 |
|---|---|---|
| 교육 목표 | AI 기술을 활용한 서비스 개발 및 구현 역량 강화 | 데이터·AI 기반 DX 문제 해결 및 사업 제안 역량 강화 |
| 교육과정 | STEP 1 AI 모델링 / STEP 2 서비스 개발 / STEP 3 통합 프로젝트 | STEP 1 AI·데이터 기술 활용 / STEP 2 IT·Cloud 및 제안 전략 / STEP 3 실무 종합 프로젝트 |
| 핵심 역량 | 데이터 처리, AI 모델 활용, AI 기반 웹 서비스 구현 | 데이터 해석, AI 활용 가능성 분석, 고객 요구사항 분석 및 사업 제안 |
| 주요 진로 | 소프트웨어 개발자, Cloud 엔지니어, 데이터사이언티스트, AI 개발자 | DX 컨설턴트, 데이터 분석·AI 전략 기획, B2B 사업 제안 관련 직무 |

### 빅프로젝트의 역할
- **AI 개발자 트랙**: AI 서비스를 설계·구현·검증하는 실무 경험 강화
- **DX 컨설턴트 트랙**: 실제 수요 기반 과제 수행과 제안서 구성·발표 경험 강화

**핵심 한 줄 요약:** AI 개발자 트랙은 구현 중심, DX 컨설턴트 트랙은 문제 해결과 제안 중심입니다.
        """
        st.markdown(demo_md)

        st.markdown("""
        <div class="answer-foot">
            근거: 통합 백서 3.2절, 3.3절, 3.4절
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # 최근 대화만 크게 표시
        user_msg = None
        assistant_msg = None

        for m in reversed(st.session_state.messages):
            if assistant_msg is None and m["role"] == "assistant":
                assistant_msg = m
            elif assistant_msg is not None and user_msg is None and m["role"] == "user":
                user_msg = m
                break

        if user_msg:
            st.markdown(f'<div class="user-msg">{user_msg["content"]}</div>', unsafe_allow_html=True)

        if assistant_msg:
            st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
            st.markdown('<div class="answer-head">🟢 AIVLE 학습 도우미</div>', unsafe_allow_html=True)
            st.markdown(assistant_msg["content"])

            if "docs" in assistant_msg and assistant_msg["docs"]:
                st.markdown('<div class="answer-foot">근거 문서: ', unsafe_allow_html=True)
                badges = ""
                for d in assistant_msg["docs"][:4]:
                    title = extract_section_title(d.page_content)
                    badges += f'<span class="section-badge">{title}</span>'
                st.markdown(badges, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # 후속 액션 버튼
            action_cols = st.columns(3)
            with action_cols[0]:
                if st.button("✨ 더 쉽게 설명해줘", use_container_width=True):
                    st.session_state.pending_question = "방금 답변을 더 쉽게 설명해줘."
                    st.rerun()
            with action_cols[1]:
                if st.button("📄 교육과정만 정리해줘", use_container_width=True):
                    st.session_state.pending_question = "방금 내용에서 교육과정만 따로 정리해줘."
                    st.rerun()
            with action_cols[2]:
                if st.button("🧭 관련 진로를 추천해줘", use_container_width=True):
                    st.session_state.pending_question = "방금 내용과 관련해서 진로를 추천해줘."
                    st.rerun()

    # 입력창
    st.markdown('<div class="bottom-input-wrap">', unsafe_allow_html=True)
    with st.form("question_form", clear_on_submit=True):
        q_col1, q_col2 = st.columns([8.5, 1.5])
        with q_col1:
            user_input = st.text_input(
                "질문 입력",
                placeholder="추가로 궁금한 내용을 입력하세요.",
                label_visibility="collapsed"
            )
        with q_col2:
            submitted = st.form_submit_button("➤ 보내기", use_container_width=True)

        if submitted:
            if not st.session_state.vector_ready:
                st.warning("먼저 PDF 백서를 업로드해 주세요.")
            elif user_input.strip():
                run_question(user_input.strip())
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 오른쪽 근거 문서 패널
# =========================================================
with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="right-title">근거 문서 보기</div>', unsafe_allow_html=True)
    st.markdown('<div class="right-sub">답변에 활용된 백서의 내용을 확인할 수 있어요.</div>', unsafe_allow_html=True)

    docs_to_show = st.session_state.latest_docs

    if docs_to_show:
        for doc in docs_to_show[:3]:
            title = extract_section_title(doc.page_content)
            snippet = doc.page_content.replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet)
            snippet = snippet[:120] + "..." if len(snippet) > 120 else snippet

            page = doc.metadata.get("page", None)
            if page is not None:
                page_text = f"p. {page + 1}"
            else:
                page_text = "p. ?"

            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-title">{title}<span class="doc-page">{page_text}</span></div>
                <div class="doc-snippet">{snippet}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="soft-card" style="padding:16px;">
            <div style="font-weight:800; margin-bottom:8px;">현재 문서</div>
            <div style="font-size:14px; color:#475569;">{st.session_state.doc_name}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="doc-card">
            <div class="doc-title">3.2 DX 컨설턴트 트랙 <span class="doc-page">p. 18</span></div>
            <div class="doc-snippet">DX 컨설턴트 트랙은 데이터와 AI를 활용한 문제 해결과 사업 제안을 목표로 하며...</div>
        </div>
        <div class="doc-card">
            <div class="doc-title">3.3 AI 개발자 트랙 <span class="doc-page">p. 21</span></div>
            <div class="doc-snippet">AI 개발자 트랙은 AI 모델링부터 서비스 개발, 통합 프로젝트까지 단계적으로 학습하며...</div>
        </div>
        <div class="doc-card">
            <div class="doc-title">3.4 프로젝트 운영 <span class="doc-page">p. 24</span></div>
            <div class="doc-snippet">미니프로젝트와 빅프로젝트를 통해 실제 문제 해결 경험을 제공하며...</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
