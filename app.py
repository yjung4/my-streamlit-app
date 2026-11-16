import os
import re
import hashlib
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AIVLE | 백서 학습 도우미", page_icon="📘", layout="wide", initial_sidebar_state="expanded")

# -------------------- 디자인 --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'Noto Sans KR',sans-serif;}
[data-testid="stAppViewContainer"] {background:#FCFDFD;}
[data-testid="stHeader"] {background:rgba(252,253,253,.88);}
[data-testid="stSidebar"] {background:#F2FAF8;border-right:1px solid #E3EFEC;}
[data-testid="stSidebar"] h2 {color:#0E635A;}
.block-container {max-width:1080px;padding-top:2.4rem;padding-bottom:5rem;}
h1,h2,h3 {letter-spacing:-.045em;}
.hero {border:1px solid #DAECE8;background:linear-gradient(115deg,#F0FAF8 0%,#FFFFFF 75%);padding:27px 30px;border-radius:22px;margin-bottom:21px;}
.eyebrow {font-size:12px;color:#108979;font-weight:700;letter-spacing:1.5px;margin-bottom:10px;}
.hero h1 {font-size:29px!important;line-height:1.38;margin:0 0 10px;color:#153B38;}
.hero p {font-size:14px;color:#61736F;margin:0;line-height:1.8;}
.sectionline {font-size:14px;font-weight:700;color:#32635C;margin:19px 0 10px;}
.stButton > button {border:1px solid #DCEAE6!important;border-radius:12px!important;background:white!important;color:#24544E!important;min-height:42px;box-shadow:none!important;}
.stButton > button:hover {border-color:#79C7B8!important;background:#F1FAF7!important;}
[data-testid="stChatMessage"] {border:1px solid #EBF0EF;border-radius:18px;background:white;padding:12px 16px;margin:12px 0;}
[data-testid="stChatInput"] {border-color:#BEDDD6;}
[data-testid="stFileUploader"] {border-radius:12px;}
[data-testid="stExpander"] {border:1px solid #E4EFEC;border-radius:12px;}
.smallnote {color:#7A8C88;font-size:12px;line-height:1.7;}
.dot {display:inline-block;width:8px;height:8px;border-radius:50%;background:#2AB59F;margin-right:8px;}
</style>
""", unsafe_allow_html=True)


# -------------------- 환경 설정 --------------------
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".aivle_cache"
CACHE_DIR.mkdir(exist_ok=True)


def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key)
    except Exception:
        pass
    local_key = BASE_DIR / "api_key.txt"
    if local_key.exists():
        for line in local_key.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def format_docs(docs):
    texts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page")
        page_label = f" / 페이지 {page + 1}" if isinstance(page, int) else ""
        texts.append(f"[근거 {i}{page_label}]\n{doc.page_content}")
    return "\n\n".join(texts)


# -------------------- RAG --------------------
@st.cache_resource(show_spinner=False)
def rag_pipeline(filepath, file_hash, api_key):
    """PDF 파일을 최초 1회 임베딩하고, 같은 파일은 Chroma 인덱스를 재사용."""
    os.environ["OPENAI_API_KEY"] = api_key

    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()
    if not docs or not any(doc.page_content.strip() for doc in docs):
        raise ValueError("PDF에서 텍스트를 읽지 못했어요. 스캔 PDF라면 OCR이 필요합니다.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
    index_dir = CACHE_DIR / f"index_{file_hash[:16]}_v1"
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=f"aivle_{file_hash[:16]}",
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )
    # 중간에 임베딩이 중단되어 불완전한 DB가 남은 경우에도 복구
    stored_ids = set(vectorstore.get(include=[])["ids"])
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    if stored_ids != set(ids):
        vectorstore.reset_collection()
        vectorstore.add_documents(chunks, ids=ids)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": min(10, len(chunks))},
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 KT AIVLE School 통합 백서를 바탕으로 답변하는 친절하고 정확한 학습 도우미입니다.

[규칙]
1. 반드시 아래 검색된 문서 내용에 근거해서 답하세요.
2. 문서에 근거가 없다면 '업로드된 문서에서 확인되지 않습니다.'라고 답하세요.
3. 숫자·일정·기수·지원조건 등은 문서 그대로 정확하게 구분하세요.
4. 서로 다른 조건(결석 처리, 수료, 제적, 장려금 지급 등)을 혼동하지 마세요.
5. 비교 요청에는 표를 활용할 수 있습니다.
6. 여러 관련 규정이 있다면 빠뜨리지 말고 정리하되, 근거가 부족한 내용은 보충해서 지어내지 마세요.
7. 답변은 한국어로 작성하고 중요한 결론을 먼저 제시하세요.
8. 출처를 언급할 때에는 [근거 1]처럼 제공된 문서 번호만 사용하세요.
9. 문서 내용이 충돌하면 그 차이를 명확히 설명하세요.

[검색된 문서]
{context}"""),
        ("human", "{input}"),
    ])
    qa_chain = prompt | llm | StrOutputParser()
    return {"retriever": retriever, "qa_chain": qa_chain, "chunk_count": len(chunks)}


def answer_question(question, rag):
    sources = rag["retriever"].invoke(question)
    answer = rag["qa_chain"].invoke({
        "context": format_docs(sources),
        "input": question,
    })
    # 검색된 출처를 그대로 보존 (답변 출처를 임의로 만들지 않음)
    return answer, sources


# -------------------- 상태 --------------------
for key, default in {
    "messages": [],
    "active_hash": None,
    "active_name": None,
    "queued_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

suggestions = [
    "AI 개발자 트랙과 DX 컨설턴트 트랙의 차이를 비교해 줘.",
    "수료 조건과 제적 조건을 구분해서 설명해 줘.",
    "비전공자가 DX 트랙을 준비하는 학습 로드맵을 알려줘.",
]

# -------------------- 사이드바 --------------------
with st.sidebar:
    st.markdown("## ◈ AIVLE Study")
    st.caption("14조 | 백서 기반 AI 학습 도우미")
    st.divider()
    st.markdown("**01　나의 학습 문서**")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"], label_visibility="collapsed")
    st.caption("백서 PDF를 넣으면 문서를 분석하고 질문에 답해요.")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        pdf_path = CACHE_DIR / f"upload_{digest[:16]}.pdf"
        if not pdf_path.exists():
            pdf_path.write_bytes(pdf_bytes)
        if st.session_state.active_hash != digest:
            st.session_state.active_hash = digest
            st.session_state.active_name = uploaded_file.name
            st.session_state.messages = []
    else:
        # 동일 폴더에 PDF가 있으면 첫 실행에서 자동 선택
        default_pdf = BASE_DIR / "AIVLE School 백서.pdf"
        if default_pdf.exists():
            pdf_bytes = default_pdf.read_bytes()
            digest = hashlib.sha256(pdf_bytes).hexdigest()
            pdf_path = default_pdf
            if st.session_state.active_hash != digest:
                st.session_state.active_hash = digest
                st.session_state.active_name = default_pdf.name
                st.session_state.messages = []
        else:
            pdf_path = None
            if st.session_state.active_hash is not None:
                st.session_state.active_hash = None
                st.session_state.active_name = None
                st.session_state.messages = []

    if st.session_state.active_name:
        st.success(f"연결된 문서: {st.session_state.active_name}")
    else:
        st.info("PDF를 업로드하면 대화를 시작할 수 있어요.")

    st.divider()
    st.markdown("**02　추천 질문**")
    for i, question in enumerate(suggestions):
        if st.button(question, key=f"side_q_{i}", use_container_width=True):
            st.session_state.queued_question = question

    st.divider()
    if st.button("↻  새 대화 시작", use_container_width=True):
        st.session_state.messages = []
        st.session_state.queued_question = None
        st.rerun()
    st.caption("업로드한 문서만을 근거로 답변하며, 답변 아래에서 검색 근거를 확인할 수 있어요.")

# -------------------- 메인 --------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AIVLE SCHOOL · DOCUMENT LEARNING</div>
  <h1>질문에서 이해까지,<br>백서와 함께 깊이 있게.</h1>
  <p>궁금한 내용을 물어보세요. 백서 속 근거를 찾아 답하고,<br>어려운 개념은 더 쉽게 풀어드리는 나만의 학습 파트너입니다.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sectionline"><span class="dot"></span>빠른 학습 시작</div>', unsafe_allow_html=True)
cols = st.columns(3, gap="small")
for i, (label, question) in enumerate([
    ("🧭 트랙 비교", suggestions[0]),
    ("📋 출결·수료", suggestions[1]),
    ("🌱 비전공자 로드맵", suggestions[2]),
]):
    with cols[i]:
        if st.button(label, key=f"quick_{i}", use_container_width=True):
            st.session_state.queued_question = question

st.markdown('<div class="sectionline"><span class="dot"></span>백서와 대화하기</div>', unsafe_allow_html=True)

api_key = get_api_key()
rag = None
if not api_key:
    st.warning("OpenAI API 키를 찾지 못했어요. 같은 폴더의 api_key.txt에 OPENAI_API_KEY=... 를 넣거나 Streamlit Secrets에 등록해 주세요.")
elif pdf_path is not None:
    try:
        with st.spinner("백서 검색 인덱스를 준비하고 있어요. 최초 한 번만 임베딩합니다..."):
            rag = rag_pipeline(str(pdf_path), st.session_state.active_hash, api_key)
        st.caption(f"📗 {st.session_state.active_name} · 검색 준비 완료 · {rag['chunk_count']}개 청크")
    except Exception as exc:
        st.error(f"백서 처리 중 오류가 발생했어요: {exc}")
else:
    st.info("왼쪽에서 통합 백서 PDF를 업로드해 주세요. 앱 파일과 같은 폴더에 'AIVLE School 백서.pdf'를 넣어도 자동으로 읽어요.")

if not st.session_state.messages:
    st.markdown("""<div class="smallnote">✨ 질문을 입력하면 핵심 답변과 검색 근거를 함께 보여드려요. 답변 아래의 '더 쉽게 설명해줘'로 이어서 학습할 수도 있어요.</div>""", unsafe_allow_html=True)

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "📘"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 이 답변에 사용된 검색 근거 보기"):
                for j, source in enumerate(msg["sources"], 1):
                    page = source.get("page")
                    title = f"근거 {j}" + (f" · PDF {page + 1}쪽" if isinstance(page, int) else "")
                    st.markdown(f"**{title}**")
                    st.text(source["text"][:1800])
                    if j < len(msg["sources"]):
                        st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✨ 더 쉽게 설명해줘", key=f"easy_{i}"):
                    st.session_state.queued_question = "방금 설명한 내용을 초보자도 이해하기 쉽게 예시와 함께 다시 설명해 줘. 이전 질문: " + st.session_state.messages[i - 1]["content"] if i > 0 else "방금 답변을 쉽게 설명해 줘."
            with c2:
                if st.button("📝 핵심만 요약해줘", key=f"short_{i}"):
                    st.session_state.queued_question = "다음 질문에 대한 백서의 핵심만 3줄로 요약해 줘: " + st.session_state.messages[i - 1]["content"] if i > 0 else "방금 답변을 3줄로 요약해 줘."

user_input = st.chat_input("AIVLE School 백서에 대해 질문해 보세요...", disabled=rag is None)
question = st.session_state.queued_question or user_input
st.session_state.queued_question = None

if question:
    if rag is None:
        st.warning("PDF와 API 키를 준비한 다음 다시 질문해 주세요.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="📘"):
            try:
                with st.spinner("백서에서 근거를 찾고 있어요..."):
                    response, source_docs = answer_question(question, rag)
                st.markdown(response)
                sources = [{"text": d.page_content, "page": d.metadata.get("page")} for d in source_docs]
                st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
            except Exception as exc:
                st.error(f"답변 생성 중 오류: {exc}")
        st.rerun()
