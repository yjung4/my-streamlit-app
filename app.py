
import os, base64, json, requests
import streamlit as st
from openai import OpenAI
from datetime import date, datetime
from streamlit_calendar import calendar 

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="AIVLE | AI 개인비서", page_icon="✦", layout="wide")

with open("Aivle_banner.png", "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>
.aivle-banner {{
    width: 100%;
    max-height: 360px;
    object-fit: contain;
    border-radius: 20px;
    margin-bottom: 24px;
    background: linear-gradient(135deg, #10283A, #164A57);
    padding: 12px;
}}
</style>

<img class="aivle-banner"
     src="data:image/png;base64,{encoded_image}">
""", unsafe_allow_html=True)

# 디자인 테마: 기존 기능은 유지하고 화면 스타일만 변경
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
:root { --navy:#132E40; --mint:#16A991; --pale:#F3F7FA; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family:'DM Sans','Noto Sans KR',sans-serif;
}
.stApp { background: #F3F7FA; color:#203A4B; }
.block-container { padding-top:1.8rem; padding-bottom:3rem; max-width:1500px; }
[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { color:#132E40 !important; letter-spacing:-0.035em; }
h1 { font-size:1.7rem !important; }
h2,h3 { font-size:1.2rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius:18px !important;
}
[data-testid="stForm"], [data-testid="stFileUploader"],
[data-testid="stChatMessage"] {
    border-radius:16px !important;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button {
    background:#16A991; color:white; border:0; border-radius:12px;
    font-weight:700; padding:0.6rem 1.2rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    border-color:#138C79; color:#138C79;
}
[data-testid="stProgressBar"] > div > div > div { background:#16A991; }
[data-testid="stChatInput"] { border-radius:14px; }
.aivle-top { display:flex; align-items:center; justify-content:space-between;
    padding:8px 0 20px; }
.aivle-brand { font-weight:800; font-size:25px; letter-spacing:-1px; color:#132E40; }
.aivle-brand span { color:#16A991; }
.aivle-tag { font-size:11px; color:#718997; font-weight:700; letter-spacing:2px; }
.aivle-pill { border-radius:99px; background:#DFF5EE; color:#118C76;
    padding:8px 14px; font-size:12px; font-weight:700; }
.aivle-hero { background:linear-gradient(115deg,#132E40,#1D4B56);
    border-radius:24px; padding:32px 38px; margin-bottom:26px;
    box-shadow:0 14px 30px rgba(16,43,58,.12); }
.aivle-eyebrow { color:#79E3C9; font-size:11px; font-weight:800;
    letter-spacing:2px; margin-bottom:10px; }
.aivle-hero h1 { color:white !important; font-size:34px !important;
    font-weight:800; margin:0 0 9px; }
.aivle-hero p { color:#D1E1E7; font-size:14px; margin:0; }
.aivle-section { color:#132E40; font-size:12px; font-weight:800;
    letter-spacing:1px; margin:14px 0 8px; }
@media(max-width:760px) {
    .aivle-hero { padding:24px; }
    .aivle-hero h1 { font-size:26px !important; }
    .aivle-pill { display:none; }
}
</style>
<div class="aivle-top">
  <div><div class="aivle-brand">✦ AIV<span>LE</span></div>
       <div class="aivle-tag">YOUR AI WORKSPACE</div></div>
  <div class="aivle-pill">● AI ASSISTANT</div>
</div>
<div class="aivle-hero">
  <div class="aivle-eyebrow">SMART WORKSPACE</div>
  <h1>Your work, simplified.</h1>
  <p>회의 요약부터 업무 관리까지, 에이블이 당신의 하루를 더 스마트하게 만듭니다.</p>
</div>
""", unsafe_allow_html=True)

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.
"""

# 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

left_col, right_col = st.columns([1.8, 1.2], gap="large")

with left_col:
    st.markdown('<div class="aivle-section">WORK MANAGEMENT</div>', unsafe_allow_html=True)
    # 지시 영역
    with st.form("task_form"):
        uploaded_file = st.file_uploader("텍스트 파일 업로드", type=["txt"])
        audio_value = st.audio_input("무엇을 도와드릴까요?")
        submit = st.form_submit_button("실행")

    # 버튼을 눌렀을 때만 실행
    if submit:
        # 업로드 된 파일이 있으면 디코딩
        text = ""
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8")

        # 녹음이 없으면 녹음 요청
        if audio_value is None:
            st.warning("음성을 녹음해주세요.")
            st.stop()

        # 시스템 메세지
        system_content = sys_role
        if text:
            system_content += f"\n\n다음 내용을 참고하세요:\n{text}"
        input_messages = [{"role": "system", "content": system_content}]

        # STT: 음성 → 텍스트
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", audio_value, "audio/wav"))

        # 사용자 메시지
        input_messages.append({"role": "user", "content": transcript.text})

        # LLM 호출 및 응답 듣기
        with st.spinner("답변을 준비하고 있습니다..."):
            # LLM 호출
            response = client.responses.create(
                model="gpt-4o-mini",
                input=input_messages
            )
            answer = response.output_text

            # 응답 출력
            st.subheader("AI 답변")
            st.write(answer)

    # 체크리스트 구현

    st.title("📋 김 대리의 업무 체크리스트")
    st.write("회의록을 요약하고 김 대리의 업무를 관리합니다.")

    # 역할 부여
    sys_role = """
    당신은 김 대리의 개인 비서 '에이블'입니다.
    김 대리가 요청한 업무를 정확하고 성실하게 처리합니다.
    회의록에 명시된 정보만 사용합니다.
    담당자나 기한을 임의로 만들어내지 않습니다.
    김 대리에게 배정된 업무와 다른 참석자의 업무를 구분합니다.
    """

    # 업무 목록 초기화
    if "tasks" not in st.session_state:
      st.session_state.tasks=[]

    if "summary" not in st.session_state:
      st.session_state.summary=""

    # 캘린더 목록 초기
    if "calendar_tasks" not in st.session_state:
        st.session_state.calendar_tasks = []

    meeting_date=st.date_input(
        "회의 날짜를 선택하세요",
        value=date.today()
    )


    # 회의록 업로드
    uploaded_file = st.file_uploader(
        "회의록 파일을 업로드하세요",
        type=["txt"]
    )

    # 회의 내용 요약 및 업무 추출
    if uploaded_file is not None:
      meeting_text= uploaded_file.getvalue().decode("utf-8")

      if st.button("회의록 분석하기"):
        with st.spinner("에이블이 회의록을 분석하고 있습니다..."):
          prompt=f"""

  
      다음 회의록을 읽고, 회의에 참석하지 않은 사람도 회의 결과와 다음 단계를 빠르게 이해할 수 있도록
      한국어로 간결하게 요약해 주세요.
  
      [회의록]
      {meeting_text}

      [작성 원칙]
      - 회의록에 명시된 사실만 사용하고, 추측하거나 세부사항을 만들어내지 마세요.
      - 제안이나 의견을 확정된 결정으로 표현하지 마세요.
      - 업무의 담당자나 기한이 회의록에 없으면 "명시되지 않음"이라고 적으세요.
      - "빨리", "조속히"처럼 구체적인 날짜가 아닌 표현은 그대로 표시하고,
      구체적인 기한은 미정이라고 덧붙이세요.
      - 김 대리에게 직접 배정된 업무를 식별하고, 다른 사람의 업무와 혼동하지 마세요.
      - 같은 내용을 반복하지 말고, 전체 요약은 한 페이지 이내로 작성하세요.

      다음 형식으로만 답변해 주세요. 제목 이름과 순서를 바꾸지 마세요.

      첫 줄은 회의록에 나온 회의 주제를 한 줄로 쓰고, 이어서 "미팅 내용을 요약해 전달드립니다."라고 작성합니다.
      주제 예시는 형식만 보여 주는 것이므로 실제 회의 주제로 바꾸세요.
      예: OO 미팅 내용을 요약해 전달드립니다.

  
      [출력 형식]

      [미팅 요약]
      - 회의록에 실제로 나온 주제만 골라, 주제별로 한 줄씩 씁니다.
      - 확정된 내용과 의견·검토 사항을 구분합니다.
      - 회의록에 없는 숫자, 담당자, 기한은 만들지 않습니다.

      [김대리님께 배정된 업무]
      1. 업무 제목
        - 해야 할 일을 회의록 표현 그대로 적습니다.
        - 그 업무와 관련하여 제시된 의견을 정리합니다.
        - 기한이 합의되었으면 그 표현을 적고, 없으면 "별도의 완료 기한은 회의에서 정해지지 않았습니다."라고 적습니다.

      김 대리에게 배정된 업무가 없으면 [김대리님께 배정된 업무] 아래에 "배정된 업무 없음"이라고 적습니다.
          다음 회의록을 읽고 한국어로 간결하게 요약하세요.
          """

          response=client.responses.create(
              model="gpt-4o-mini",
              input=[
                  {"role":"system", "content":sys_role},
                  {"role":"user", "content": prompt}
              ]
          )

          #회의 요약 저장
          st.session_state.summary=response.output_text

          #요약에서 김대리 업무 추출
          task_response=client.responses.create(
              model="gpt-4o-mini",
              input=f"""
              다음 회의 요약에서 김 대리에게 배정된 업무와 마감일을 추출하세요.

              [회의 요약]
              {response.output_text}

              [회의 날짜]
              {meeting_date.isoformat()}

              [회의록]
              {meeting_text}

              [규칙]
              - 김 대리의 업무만 추출하세요.
              - 다른 사람의 업무는 제외하세요.
              - 제안만 된 사항은 확정 업무로 만들지 마세요.
              - 업무가 없다면 빈 리스트를 반환하세요.
              - 마감일은 회의록에 명시된 경우에만 지정하세요.
              - 마감일이 명시되지 않은 업무는 반드시 deadline을 null로 반환하세요.
              - 마감일이 없다고 해서 회의일로부터 7일 뒤, 다음 주 또는 임의의 날짜를 지정하지 마세요.
              - '다음 주까지'는 회의 날짜를 기준으로 다음 주 금요일로 지정하세요.
              - '이번 주까지'는 회의 날짜를 기준으로 이번 주 금요일로 지정하세요.
              - '다음 주 금요일까지'는 다음 주 금요일로 지정하세요.
              - '11월 12일까지'처럼 특정 날짜가 명시되었다면 해당 날짜를 우선 적용하세요.
              - '3일 뒤까지', '일주일 뒤까지'처럼 명확한 상대 날짜가 명시된 경우에만 회의 날짜를 기준으로 계산하세요.
              - 여러 날짜가 언급되면 해당 업무의 실제 마감일을 선택하세요.
              - 날짜가 모호하거나 마감일인지 확실하지 않으면 deadline을 null로 반환하세요.
              - 날짜는 반드시 YYYY-MM-DD 형식으로 작성하세요.
              - 반드시 JSON 형식으로 답변하세요.
              - 업무와 마감일을 판단할 때는 회의 요약보다 원본 회의록을 우선하세요.
              - 요약에 날짜가 있더라도 원본 회의록에 근거가 없으면 마감일을 지정하지 마세요.

             {{
              "tasks": [
                {{
                  "title": "업무1",
                  "deadline": "YYYY-MM-DD"
                }},
                {{
                  "title": "업무2",
                  "deadline": null
                }}
              ]
            }}
            """,
              text={"format": {"type": "json_object"}}
            )
      
          # JSON을 파이썬 리스트로 변환
          extracted_tasks = json.loads(
              task_response.output_text
          )["tasks"]

          # 기존 업무와 중복되지 않는 업무만 추가
          for task in extracted_tasks:
              task_name=task["title"]
              task_deadline=task.get("deadline")

              if task_name not in st.session_state.tasks:
                st.session_state.tasks.append(task_name)
              if task_deadline:
                try:
                  valid_date=date.fromisoformat(task_deadline)

                  existing=[
                      event
                      for event in st.session_state.calendar_tasks
                      if event["title"] == task_name
                  ]

                  if not existing:
                    st.session_state.calendar_tasks.append({
                        "title":task_name,
                        "start": valid_date.isoformat(),
                        "allDay":True
                    }
                    )
                except (ValueError,TypeError):
                      pass

        st.success("회의록 분석이 완료되었습니다!")

    # 회의 요약 표시
    if st.session_state.summary:
      st.divider()
      st.subheader("회의 요약")

      st.write(st.session_state.summary)

    # 체크리스트

    st.divider()
    st.subheader("✅ 김 대리의 업무 체크리스트")

    completed=0

    for i, task in enumerate(st.session_state.tasks):
      if st.checkbox(task, key=f"task_{i}"):
        completed += 1

    # 진행률 표시
    total=len(st.session_state.tasks)
    st.progress(completed/total if total else 0)
    st.write(f"완료한 업무:{completed}/{total}개")

    #새로운 업무 추가
    st.divider()
    st.subheader("업무 추가")

    new_task=st.text_input("새로운 업무 입력")

    if st.button("업무 추가"):
      if new_task.strip():
        if new_task.strip() not in st.session_state.tasks:
          st.session_state.tasks.append(new_task.strip())
          st.rerun()
        else:
          st.warning("이미 등록된 업무입니다.")
      else:
        st.warning("업무를 입력해주세요.")

    # streamlit calendar 구현

    st.divider()
    st.title("📅 김 대리의 업무 캘린더")

    with st.form("calendar_form"):
      task_name=st.selectbox(
          "캘린더에 등록할 업무",
          options=st.session_state.tasks
      ) if st.session_state.tasks else None
      deadline=st.date_input("마감일", value=date.today())
      submitted=st.form_submit_button("캘린더에 등록/날짜 수정")

      if submitted:
        if task_name:
          existing=[
              event
              for event in st.session_state.calendar_tasks
              if event["title"] == task_name
          ]

          if existing:
            existing[0]["start"] = deadline.isoformat()
            st.success("캘린더 날짜가 수정되었습니다.")
          else:
            st.session_state.calendar_tasks.append({
                "title":task_name,
                "start":deadline.isoformat(),
                "allDay":True
            })
            st.success("캘린더에 등록되었습니다.")
        else:
            st.warning("업무명을 입력해주세요.")

    # 캘린더 표시
    calendar(
        events=st.session_state.calendar_tasks,
        options={
            "initialView":"dayGridMonth",
            "locale":"ko",
            "height":550
        },
        key="work_calendar"
    )

with right_col:
    st.markdown('<div class="aivle-section">AI ASSISTANT</div>', unsafe_allow_html=True)
    st.subheader("💬 AIVLE AI 챗봇")
    st.caption("에이블에게 무엇이든 물어보세요!")

    chatbot_role = """
    당신은 AI 개인비서 에이블(AIVLE)입니다.
    사용자의 질문에 친절하고 정확하게 답변하세요.
    업무 관리, 일정 계획, 문서 작성 등을 도와주세요.
    """

    # 대화 기록 출력 (오른쪽 영역 안에서 스크롤)
    with st.container(height=450):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 사용자 입력
    with st.container(border=True):
        user_input = st.chat_input(
            "에이블에게 메시지를 입력하세요...",
            key="aivle_chat"
        )

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("에이블이 생각하고 있습니다..."):
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": chatbot_role},
                    *st.session_state.messages
                ]
            )
            answer = response.output_text

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.rerun()
