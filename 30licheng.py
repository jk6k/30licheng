# streamlit_app.py
# 职责: 一个完整的、单文件的Streamlit应用，整合了原FastAPI+React项目的所有核心功能。
# 版本: 4.2 (优化模式四的报告显示体验)

import streamlit as st
import os
import re
import json
import asyncio
import uuid
from datetime import datetime, timezone

# ==============================================================================
# SECTION 0: LANGSMITH TRACING CONFIGURATION
# ==============================================================================
# 显式禁用LangSmith追踪，以防止在受限网络环境中出现连接错误。
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

# --- 依赖项 ---
import pandas as pd
from sqlalchemy import create_engine, Column, String, Text, ForeignKey, JSON
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from serpapi import GoogleSearch

    SEARCH_TOOL_ENABLED = True
except ImportError:
    SEARCH_TOOL_ENABLED = False

# ==============================================================================
# SECTION 1: LLM 和外部服务设置
# ==============================================================================

# --- 加载环境变量/Secrets ---
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY") or st.secrets.get("LLM_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or st.secrets.get("OPENAI_API_BASE")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY")

# --- 初始化LLM和搜索工具 ---
llm = None
search_tool = None


def check_services():
    if not LLM_API_KEY or not OPENAI_API_BASE:
        st.error("错误：请在Streamlit Cloud的Secrets中设置 `LLM_API_KEY` 和 `OPENAI_API_BASE`。")
        return False

    global llm
    if llm is None:
        try:
            llm = ChatOpenAI(
                model="claude-sonnet-4-20250514",
                temperature=0.7,
                api_key=LLM_API_KEY,
                base_url=OPENAI_API_BASE,
                max_retries=3,
                timeout=30,
            )
        except Exception as e:
            st.error(f"LLM 初始化失败: {e}")
            return False

    global search_tool
    if not SERPAPI_API_KEY:
        st.warning("警告：未找到 `SERPAPI_API_KEY`。需要用到搜索功能的服务（如研究报告）将不可用。")
    elif SEARCH_TOOL_ENABLED:
        search_tool = GoogleSearch
    else:
        st.warning("警告：`serpapi` 未安装。搜索功能已禁用。")
    return True


# --- 提示词 (Personas) ---
PERSONA_MODE_1 = """
你是一个职业探索与研究助理，属于‘30历程’的【目标研究】模块。你的风格需模仿一位经验丰富、直言不讳的职业导师。
你的任务是：在对用户画像进行总结分析后，提供职业建议。
你的输出必须严格遵循以下格式，其中JSON部分必须被包裹在一个完整的 ```json ... ``` 代码块中:

这里是你的导师总结文本...

```json
{{
  "summary": "这里是你的总结文本，内容应与上面的导师总结文本一致",
  "suggestions": [
    {{
      "title": "职业建议1",
      "reason": "推荐这个职业的具体理由1"
    }},
    {{
      "title": "职业建议2",
      "reason": "推荐这个职业的具体理由2"
    }}
  ]
}}
```
在所有互动中，你必须始终应用【个人特质背景成长适切】和【发展趋势研究】两大原则进行提问和总结。你的所有回答都必须使用简体中文。
"""
PERSONA_MODE_2 = """
你是一个清醒而犀利的职业决策教练，属于‘30历程’的【决策与评估】模块。你的任务是帮助用户戳破幻想，进行严格的现实检验。你的核心任务是：
1.  **设计高效的检验计划**：引导用户设计包含【职业访谈】和【现场观察】的计划。在设计访谈时，你必须明确建议用户去接触两类最关键的人：一是具体负责招聘的业务主管，二是在该领域深耕15年以上（35-50岁）的资深人士。提醒用户，刚毕业的师兄师姐可能“连门都没摸到呢”。
2.  **引导批判性思考**：在用户反馈访谈见闻后，你的首要追问必须是：“在这些对话中，哪些是毋庸置疑的‘事实’，哪些只是对方的‘观点’？”
3.  **探寻深层价值**：引导用户反思这份工作是否符合其“个人特质背景成长适切”。你要帮助他们超越简单的性格标签（如I人E人），去寻找真正的“价值锚点”——即那种能带来核心成就感、让自己安身立命的东西。
你的所有回答都必须使用简体中文。
"""
PERSONA_MODE_3 = """
你是一个精密的职业行动规划师，属于‘30历程’的【计划与行动】模块。你的任务不仅是制定计划，更是要向用户阐明每个行动背后的深刻逻辑。
你的核心任务是：将一个已确立的职业目标，转化为一份结构化的行动蓝图。
你的输出必须严格遵循以下格式，其中JSON部分必须被包裹在一个完整的 ```json ... ``` 代码块中:

```json
{{
    "plan_details": "对整个行动蓝图的总体描述，可以为空字符串",
    "academic": "关于学业准备的详细清单和说明",
    "practice": "关于科研、竞赛、实习的详细清单和说明",
    "skills": "关于学生干部、社团、志愿活动和社会资源利用的详细清单和说明"
}}
```
确保每个清单都具有高度的可操作性，并与用户的大学专业背景紧密结合。你的所有回答都必须使用简体中文。
"""
PERSONA_MODE_4 = """
你是一个长期的职业生涯导航员，属于‘30历程’的【未来发展因应】模块。你的任务是帮助用户理解规划的动态性，并始终保持前瞻性。你的核心任务是：
1.  **引导动态调整循环**：帮助用户追踪其行动计划的进度。当计划顺利时，引导其进行“达成与优化”的复盘；当遇到内外部变化时，引导其进行“变化与调整”的思考。你要让用户明白，规划是动态的，但绝不等于随波逐流。
2.  **提供三维度的未来洞察**：当用户请求时，你需要主动监测并向他们报告与其职业领域相关的未来发展趋势。你的报告必须围绕以下三个核心维度展开：
    * **技术趋势与学习**：有哪些新技术正在改变这个领域的工作方式？
    * **环境变化**：国际局势、社会结构或自然环境的哪些变化可能带来机会或挑战？
    * **观念发展**：社会大众或行业内部对这个领域的价值观念正在发生什么变化？
确保你的分析能帮助用户“比时代快一点点”，做出更明智的调整。你的所有回答都必须使用简体中文。
"""


# --- 服务函数 (LLM调用逻辑) ---
async def generate_suggestions_service(profile_data: dict) -> str:
    if not llm: return "LLM服务不可用。"
    prompt = ChatPromptTemplate.from_messages([
        ("system", PERSONA_MODE_1),
        ("human", "我的个人画像如下:\n{profile_json}")
    ])
    chain = prompt | llm
    profile_json_str = json.dumps(profile_data, ensure_ascii=False, indent=2)
    response = await chain.ainvoke({"profile_json": profile_json_str})
    return response.content


async def research_job_service(target_job: str, profile_data: dict) -> str:
    if not llm: return "LLM服务不可用。"
    if not search_tool: return "搜索服务不可用，无法进行深入研究。"
    queries = [f'"{target_job}" 发展趋势 报告', f'"{target_job}" 核心能力要求 技能', f'"{target_job}" 薪酬范围']
    search_context = ""
    for query in queries:
        try:
            params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY}
            results = search_tool(params).get_dict().get("organic_results", [])
            for result in results[:2]:
                if snippet := result.get('snippet'): search_context += snippet + "\n\n"
        except Exception as e:
            st.error(f"搜索查询 '{query}' 失败: {e}")

    prompt_template_str = PERSONA_MODE_1 + """
        我选择了【{target_job}】。这是搜索到的信息摘要：
        ---
        {search_context}
        ---
        请严格按照以下结构，生成一份关于【{target_job}】的研究报告：
        1. **趋势与政策分析**
        2. **岗位胜任力模型**
        3. **初步适切性评估**（结合用户画像）

        在报告的末尾，请附带一个用于数据可视化的JSON代码块，不要包含任何解释性文字，格式如下：
        ```json
        {{
            "salary_range": [
                {{"level": "初级", "low": 8000, "high": 15000}},
                {{"level": "中级", "low": 15000, "high": 25000}},
                {{"level": "高级", "low": 25000, "high": 40000}}
            ],
            "skill_importance": [
                {{"skill": "数据分析", "importance": 90}},
                {{"skill": "项目管理", "importance": 75}},
                {{"skill": "沟通协作", "importance": 85}},
                {{"skill": "技术栈X", "importance": 95}}
            ]
        }}
        ```
        用户画像参考: {profile_json}"""

    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    chain = prompt | llm
    profile_json_str = json.dumps(profile_data, ensure_ascii=False)
    response = await chain.ainvoke({
        "target_job": target_job,
        "search_context": search_context or '无特定信息',
        "profile_json": profile_json_str
    })
    return response.content


async def generate_validation_plan_service(target_name: str) -> str:
    if not llm: return "LLM服务不可用。"
    prompt = ChatPromptTemplate.from_template(
        PERSONA_MODE_2 + "\n我的目标职业是“{target_name}”。请帮我设计一份高效的现实检验计划，包括建议的访谈问题、可行的观察渠道，并特别强调我应该访谈哪些类型的关键人物。")
    chain = prompt | llm
    response = await chain.ainvoke({"target_name": target_name})
    return response.content


async def analyze_feedback_service(target_name: str, feedback: str) -> str:
    if not llm: return "LLM服务不可用。"
    prompt = ChatPromptTemplate.from_template(
        PERSONA_MODE_2 + "\n教练您好，我完成了对“{target_name}”的现实检验，以下是我的反馈：\n\n{feedback}\n\n请您基于我的反馈，通过苏格拉底式的提问，引导我分辨其中的‘事实’与‘观点’，并帮助我探寻这份工作是否触动了我的‘价值锚点’。")
    chain = prompt | llm
    response = await chain.ainvoke({"target_name": target_name, "feedback": feedback})
    return response.content


async def generate_action_plan_service(target_name: str, profile_data: dict, research_summary: str) -> str:
    if not llm: return "LLM服务不可用。"
    prompt_template = (
            PERSONA_MODE_3 +
            "\n我的目标职业是: {target_name}\n我的个人画像是: {profile_json}\n我的纸面研究报告摘要是: {research_summary}")
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    profile_json = json.dumps(profile_data, ensure_ascii=False)
    response = await chain.ainvoke({
        "target_name": target_name,
        "profile_json": profile_json,
        "research_summary": research_summary or "无",
    })
    return response.content


async def generate_trends_report_service(target_name: str) -> str:
    if not llm: return "LLM服务不可用。"
    if not search_tool: return "搜索服务不可用，无法进行深入研究。"
    queries = [f'"{target_name}" 技术趋势 2025', f'"{target_name}" 行业社会环境变化', f'"{target_name}" 职业观念发展']
    search_context = ""
    for query in queries:
        try:
            params = {"engine": "google", "q": query, "api_key": SERPAPI_API_KEY}
            results = search_tool(params).get_dict().get("organic_results", [])
            for result in results[:2]:
                if snippet := result.get('snippet'):
                    search_context += f"来源: {result.get('link')}\n摘要: {snippet}\n\n"
        except Exception as e:
            st.error(f"搜索查询 '{query}' 失败: {e}")
    prompt = ChatPromptTemplate.from_template(
        PERSONA_MODE_4 + "\n我的目标是“{target_name}”。这是刚搜索到的相关信息：\n---\n{search_context}\n---\n请基于此，严格按照【技术趋势与学习】、【环境变化】和【观念发展】三个维度，为我生成一份深刻的未来趋势洞察报告。")
    chain = prompt | llm
    response = await chain.ainvoke({"target_name": target_name, "search_context": search_context or '无特定信息'})
    return response.content


# ==============================================================================
# SECTION 2: 数据库设置
# ==============================================================================

DATABASE_URL = "sqlite:///./30licheng_st.db"
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default="main_user")
    profile_data = Column(JSON, default=lambda: {})
    chat_history = Column(JSON, default=lambda: {})
    career_targets = relationship("CareerTarget", back_populates="user", cascade="all, delete-orphan")
    progress_logs = relationship("ProgressLog", back_populates="user", cascade="all, delete-orphan")


class CareerTarget(Base):
    __tablename__ = "career_targets"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, index=True, nullable=False)
    status = Column(String, default="researching")
    research_report = Column(Text)
    research_chart_data = Column(JSON)
    validation_plan = Column(Text)
    action_plan = Column(JSON)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, default="main_user")
    user = relationship("User", back_populates="career_targets")


class ProgressLog(Base):
    __tablename__ = "progress_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    date = Column(String, nullable=False)
    log = Column(Text, nullable=False)
    target_name = Column(String)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, default="main_user")
    user = relationship("User", back_populates="progress_logs")


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@st.cache_resource
def get_db_engine():
    Base.metadata.create_all(bind=engine)
    return engine


# ==============================================================================
# SECTION 3: 应用辅助函数
# ==============================================================================

def get_db_session():
    engine = get_db_engine()
    db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    return db


def get_user_session(db):
    user = db.query(User).filter_by(id="main_user").first()
    if not user:
        user = User(id="main_user")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def extract_json_from_llm(text_content: str) -> dict | None:
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text_content)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


def update_chat_history(db, user, mode_key, human_msg, ai_msg):
    history = user.chat_history or {}
    mode_history = history.get(mode_key, [])
    mode_history.append({"role": "user", "content": human_msg})
    mode_history.append({"role": "assistant", "content": ai_msg})
    history[mode_key] = mode_history

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(user, "chat_history")

    db.commit()


# ==============================================================================
# SECTION 4: Streamlit UI 渲染函数
# ==============================================================================

def render_dashboard(db):
    st.header("🏠 导航看板")
    st.write("欢迎来到“30历程”，请选择您要进行的规划阶段。建议从“模式一”开始，循序渐进。")

    user = get_user_session(db)
    targets = user.career_targets

    is_mode2_enabled = any(t.status in ['researching', 'active', 'paused', 'planning_done'] for t in targets)
    is_mode3_enabled = any(t.status in ['active', 'planning_done'] for t in targets)
    is_mode4_enabled = is_mode3_enabled

    mode_info = [
        {"icon": "🎯", "title": "模式一：目标研究", "desc": "探索内在特质与外部机会，确立并研究初步的职业目标。",
         "enabled": True, "unlock_req": ""},
        {"icon": "🤔", "title": "模式二：决策与评估", "desc": "通过现实检验方法，评估目标的真实性与个人匹配度。",
         "enabled": is_mode2_enabled, "unlock_req": "完成模式一的目标研究后解锁"},
        {"icon": "🚀", "title": "模式三：计划与行动", "desc": "将已验证的目标，分解为学业、实践等具体行动计划。",
         "enabled": is_mode3_enabled, "unlock_req": "在模式二中激活一个目标后解锁"},
        {"icon": "🔭", "title": "模式四：未来发展因应", "desc": "长期追踪进度，应对未来变化，动态调整您的职业路径。",
         "enabled": is_mode4_enabled, "unlock_req": "在模式三中完成计划后解锁"},
    ]

    for info in mode_info:
        with st.container(border=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"<p style='font-size: 48px; text-align: center;'>{info['icon']}</p>",
                            unsafe_allow_html=True)
            with col2:
                st.subheader(info['title'])
                st.write(info['desc'])
                if st.button(f"进入 {info['title']}", key=f"dash_{info['title']}", disabled=not info['enabled'],
                             use_container_width=True):
                    st.session_state.current_view = info['title']
                    st.rerun()
                if not info['enabled']:
                    st.caption(f"🔒 {info['unlock_req']}")


def render_mode1(db):
    st.header("🎯 模式一：目标研究")
    st.write("在本模式中，我们将通过结构化分析，从自我探索开始，确立并研究您的潜在职业方向。")

    user = get_user_session(db)

    with st.expander("第一步：完善您的个人画像", expanded=True):
        with st.form("profile_form"):
            profile_data = user.profile_data or {}
            uniqueness = st.text_area("天赋、兴趣 (请用逗号分隔)",
                                      value=", ".join(profile_data.get("personal_uniqueness", [])))
            platform = st.text_input("大学平台、专业空间", value=profile_data.get("university_platform", ""))
            others = st.text_input("重要他人：家人、师友的期望或建议",
                                   value=profile_data.get("significant_others_input", ""))
            serendipity = st.text_area("机缘：对您产生特别影响的偶然经历", value=profile_data.get("serendipity", ""))

            submitted = st.form_submit_button("保存画像并生成初步职业建议")
            if submitted:
                updated_profile = {
                    "personal_uniqueness": [s.strip() for s in uniqueness.split(',') if s.strip()],
                    "university_platform": platform,
                    "significant_others_input": others,
                    "serendipity": serendipity,
                }
                user.profile_data = updated_profile
                db.commit()

                with st.spinner("AI导师正在为您分析..."):
                    raw_content = asyncio.run(generate_suggestions_service(updated_profile))
                    st.session_state.m1_raw_response = raw_content
                    parsed_json = extract_json_from_llm(raw_content)
                    st.session_state.m1_suggestions = parsed_json.get("suggestions", []) if parsed_json else []
                    human_msg = "这是我的个人画像，请分析并生成职业建议。"
                    update_chat_history(db, user, "mode1", human_msg, raw_content)
                st.success("分析完成！")

    if st.session_state.get('m1_raw_response'):
        st.markdown("---")
        st.subheader("AI导师的分析与建议")
        raw_content = st.session_state.m1_raw_response
        text_part = raw_content.split("```json")[0].strip()
        st.info(text_part)

    st.markdown("---")
    st.subheader("第二步：执行纸面研究")

    suggestions = st.session_state.get("m1_suggestions", [])

    col1, col2 = st.columns([3, 1])
    with col1:
        target_job_input = st.text_input("手动输入职业名称进行研究", key="m1_manual_job")
    with col2:
        st.write("")
        st.write("")
        if st.button("研究手动输入的目标", use_container_width=True, disabled=not target_job_input):
            st.session_state.m1_job_to_research = target_job_input

    if suggestions:
        st.write("或从AI建议中选择：")
        for s in suggestions:
            if st.button(f"研究 '{s['title']}'", key=f"research_{s['title']}", use_container_width=True):
                st.session_state.m1_job_to_research = s['title']

    if 'm1_job_to_research' in st.session_state and st.session_state.m1_job_to_research:
        final_target_job = st.session_state.m1_job_to_research
        with st.spinner(f"AI助理正在深入研究 '{final_target_job}'..."):
            try:
                raw_report = asyncio.run(research_job_service(final_target_job, user.profile_data))
                text_content = raw_report.split("```json")[0].strip()
                chart_data = extract_json_from_llm(raw_report)

                target = db.query(CareerTarget).filter_by(name=final_target_job, user_id=user.id).first()
                if not target:
                    target = CareerTarget(name=final_target_job, user_id=user.id)
                    db.add(target)

                target.research_report = text_content
                target.research_chart_data = chart_data
                target.status = "researching"
                db.commit()

                human_msg = f"请为我研究 '{final_target_job}' 这个职业。"
                update_chat_history(db, user, "mode1", human_msg, text_content)
                st.success(f"'{final_target_job}' 的研究报告已生成并保存！")
            except Exception as e:
                st.error(f"研究失败: {e}")
        del st.session_state.m1_job_to_research

    targets = db.query(CareerTarget).all()
    if targets:
        st.markdown("---")
        st.subheader("已研究的目标库")
        for t in targets:
            with st.expander(f"**{t.name}** (状态: {t.status})"):
                st.markdown(t.research_report or "暂无报告文本。")
                if t.research_chart_data:
                    try:
                        if 'skill_importance' in t.research_chart_data and t.research_chart_data['skill_importance']:
                            st.write("核心技能重要性:")
                            chart_df = pd.DataFrame(t.research_chart_data['skill_importance'])
                            st.bar_chart(chart_df.set_index('skill'))
                        if 'salary_range' in t.research_chart_data and t.research_chart_data['salary_range']:
                            st.write("薪酬范围参考 (元/月):")
                            salary_df = pd.DataFrame(t.research_chart_data['salary_range'])
                            st.bar_chart(salary_df.set_index('level'))
                    except Exception as e:
                        st.warning(f"无法渲染图表: {e}")


def render_mode2(db):
    st.header("🤔 模式二：决策与评估")
    st.write("在本模式中，我们将通过现实检验来戳破幻想，并深入内心找到自己的“价值锚点”。")

    user = get_user_session(db)

    targets_for_eval = [t for t in user.career_targets if t.status in ['researching', 'active', 'paused']]
    if not targets_for_eval:
        st.info("请先在“模式一”中研究至少一个目标，才能开始决策与评估。")
        return

    target_options = {t.name: t for t in targets_for_eval}
    selected_target_name = st.selectbox("选择一个目标进行评估", options=target_options.keys())

    if selected_target_name:
        target = target_options[selected_target_name]

        st.markdown("---")
        st.subheader(f"评估目标: **{target.name}**")

        if not target.validation_plan:
            if st.button("1. 为我生成检验计划"):
                with st.spinner("AI教练正在为您设计检验计划..."):
                    plan = asyncio.run(generate_validation_plan_service(target.name))
                    target.validation_plan = plan
                    db.commit()
                    st.success("检验计划已生成！")
                    st.rerun()
        else:
            with st.expander("1. 查看检验计划", expanded=True):
                st.markdown(target.validation_plan)

        if target.validation_plan:
            st.markdown("---")
            with st.form("feedback_form"):
                st.subheader("2. 记录检验反馈")
                feedback_text = st.text_area("请在此详细记录您在访谈或观察中的见闻和感受...", height=200)
                submitted_feedback = st.form_submit_button("提交反馈并获取AI分析")

                if submitted_feedback and feedback_text:
                    with st.spinner("AI教练正在分析您的反馈..."):
                        analysis = asyncio.run(analyze_feedback_service(target.name, feedback_text))

                        log_entry = ProgressLog(
                            date=datetime.now(timezone.utc).isoformat(),
                            log=f"【检验反馈】:\n{feedback_text}",
                            target_name=target.name,
                            user_id=user.id
                        )
                        db.add(log_entry)

                        human_msg = f"这是我关于“{target.name}”的检验反馈：\n{feedback_text}"
                        update_chat_history(db, user, "mode2", human_msg, analysis)

                        st.session_state.latest_feedback_analysis = analysis

                        st.success("反馈分析完成！")

            if 'latest_feedback_analysis' in st.session_state:
                st.markdown("---")
                st.subheader("AI教练的分析与洞察")
                st.info(st.session_state.latest_feedback_analysis)

        st.markdown("---")
        st.subheader("3. 做出最终决策")
        st.write("基于您的现实检验和AI分析，现在是时候对这个目标做出决策了。")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("✅ 激活目标", use_container_width=True, help="将此目标设为积极追求状态，以进行下一步规划。"):
                target.status = "active"
                db.commit()
                st.success(f"目标 '{target.name}' 已激活！现在可以去模式三制定行动计划了。")
                st.rerun()
        with col2:
            if st.button("⏸️ 暂时搁置", use_container_width=True, help="暂时搁置此目标，以后可以重新评估。"):
                target.status = "paused"
                db.commit()
                st.success(f"目标 '{target.name}' 已搁置。")
                st.rerun()
        with col3:
            if st.button("❌ 彻底放弃", use_container_width=True, help="将此目标从您的列表中移除。"):
                db.delete(target)
                db.commit()
                st.success(f"目标 '{target.name}' 已放弃并移除。")
                st.rerun()


def render_mode3(db):
    st.header("🚀 模式三：计划与行动")
    st.write("在本模式中，我们将把已激活的职业目标，分解为具体、可执行的行动蓝图。")

    user = get_user_session(db)

    active_targets = [t for t in user.career_targets if t.status in ['active', 'planning_done']]
    if not active_targets:
        st.info("请先在“模式二”中激活至少一个目标，才能开始制定行动计划。")
        return

    target_options = {t.name: t for t in active_targets}
    selected_target_name = st.selectbox("选择一个已激活的目标进行规划", options=target_options.keys())

    if selected_target_name:
        target = target_options[selected_target_name]

        if not target.action_plan or isinstance(target.action_plan, str):
            if st.button(f"为“{target.name}”生成三合一行动蓝图"):
                with st.spinner("AI规划师正在为您量身定制行动蓝图..."):
                    raw_plan = asyncio.run(
                        generate_action_plan_service(target.name, user.profile_data, target.research_report))
                    plan_json = extract_json_from_llm(raw_plan)

                    target.action_plan = plan_json
                    target.status = "planning_done"
                    db.commit()

                    human_msg = f"请为我的目标“{target.name}”生成行动蓝图。"
                    update_chat_history(db, user, "mode3", human_msg, raw_plan)
                    st.success("行动蓝图已生成！")
                    st.rerun()

        if target.action_plan and isinstance(target.action_plan, dict):
            st.markdown("---")
            st.subheader(f"“{target.name}”的行动蓝图")
            plan = target.action_plan

            st.markdown("#### 📚 学业清单")
            st.markdown(plan.get("academic", "暂无内容"))

            st.markdown("#### 🏅 科研竞赛实习清单")
            st.markdown(plan.get("practice", "暂无内容"))

            st.markdown("#### 🧩 学干社团与社会资源清单")
            st.markdown(plan.get("skills", "暂无内容"))


def render_mode4(db):
    st.header("🔭 模式四：未来发展因应")
    st.write("在本模式中，我们将长期追踪您的进展，并动态洞察未来趋势，确保您的规划永不过时。")

    user = get_user_session(db)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("我的进展日志")
        logs = sorted(user.progress_logs, key=lambda x: x.date, reverse=True)
        if not logs:
            st.write("暂无日志。在模式二中提交检验反馈后，将自动生成日志。")
        else:
            for log in logs:
                with st.container(border=True):
                    st.caption(f"{log.date} | 目标: {log.target_name}")
                    st.markdown(log.log)

    with col2:
        st.subheader("获取未来趋势洞察报告")
        trend_targets = [t for t in user.career_targets if t.status in ['active', 'planning_done']]
        if not trend_targets:
            st.info("请先激活一个目标，才能获取趋势报告。")
        else:
            target_options = {t.name: t for t in trend_targets}
            selected_target_name = st.selectbox("选择一个目标获取趋势报告", options=target_options.keys())
            if st.button("生成趋势洞察报告"):
                with st.spinner(f"AI导航员正在分析“{selected_target_name}”的未来趋势..."):
                    report = asyncio.run(generate_trends_report_service(selected_target_name))
                    human_msg = f"请为我的目标“{selected_target_name}”生成一份未来趋势洞察报告。"
                    update_chat_history(db, user, "mode4", human_msg, report)

                    # [已修复] 将报告结果存入session_state以便立即显示
                    st.session_state.latest_trends_report = report

                    st.success("趋势报告已生成！")

        # [已修复] 在按钮外部显示最新的报告结果
        if 'latest_trends_report' in st.session_state:
            st.markdown("---")
            st.subheader("AI导航员的未来趋势洞察")
            st.info(st.session_state.latest_trends_report)


# ==============================================================================
# SECTION 5: 主应用逻辑
# ==============================================================================

def main():
    st.set_page_config(page_title="30历程 - 职业规划Agent", layout="wide")
    st.title("🌱 30历程")
    st.caption("一个基于大语言模型，帮助您进行职业探索、决策与规划的智能助理。")

    if not check_services():
        st.stop()

    get_db_engine()  # 初始化数据库引擎和表
    db = get_db_session()

    # 初始化视图状态
    if "current_view" not in st.session_state:
        st.session_state.current_view = "导航看板"

    # 清除特定模式的临时状态函数
    def clear_temp_states():
        if 'latest_feedback_analysis' in st.session_state:
            del st.session_state.latest_feedback_analysis
        if 'latest_trends_report' in st.session_state:
            del st.session_state.latest_trends_report

    # 侧边栏导航
    with st.sidebar:
        st.title("导航")
        if st.button("🏠 导航看板", use_container_width=True):
            st.session_state.current_view = "导航看板"
            clear_temp_states()

        st.markdown("---")

        user_for_nav = get_user_session(db)
        targets_for_nav = user_for_nav.career_targets

        is_mode2_enabled = any(
            t.status in ['researching', 'active', 'paused', 'planning_done'] for t in targets_for_nav)
        is_mode3_enabled = any(t.status in ['active', 'planning_done'] for t in targets_for_nav)
        is_mode4_enabled = is_mode3_enabled

        nav_items = {
            "模式一：目标研究": True,
            "模式二：决策与评估": is_mode2_enabled,
            "模式三：计划与行动": is_mode3_enabled,
            "模式四：未来发展因应": is_mode4_enabled
        }

        for item, enabled in nav_items.items():
            if st.button(item, use_container_width=True, disabled=not enabled):
                st.session_state.current_view = item
                clear_temp_states()

        st.markdown("---")
        st.subheader("聊天历史")
        with st.expander("显示/隐藏当前模式聊天记录"):
            chat_history_data = user_for_nav.chat_history if isinstance(user_for_nav.chat_history, dict) else {}
            mode_key_map = {
                "导航看板": "mode1",  # 默认显示模式一历史
                "模式一：目标研究": "mode1",
                "模式二：决策与评估": "mode2",
                "模式三：计划与行动": "mode3",
                "模式四：未来发展因应": "mode4"
            }
            current_mode_key = mode_key_map.get(st.session_state.current_view, "mode1")
            history_for_mode = chat_history_data.get(current_mode_key, [])

            if not history_for_mode:
                st.write("暂无聊天记录。")
            else:
                for message in history_for_mode:
                    with st.chat_message(message["role"]):
                        display_content = message["content"].split("```json")[0].strip()
                        st.markdown(display_content)

    # 主内容区渲染
    mode_render_map = {
        "导航看板": render_dashboard,
        "模式一：目标研究": render_mode1,
        "模式二：决策与评估": render_mode2,
        "模式三：计划与行动": render_mode3,
        "模式四：未来发展因应": render_mode4,
    }

    render_function = mode_render_map.get(st.session_state.current_view)
    if render_function:
        render_function(db)

    db.close()


if __name__ == "__main__":
    main()
