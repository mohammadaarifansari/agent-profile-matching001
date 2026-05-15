import os, json, re
from typing import TypedDict
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from tools.file_tools  import read_resume, read_job_description, list_job_descriptions
from tools.rag_tools   import search_resumes, get_store
from tools.agent_tools import extract_requirements, compare_candidates, generate_interview_questions

print("[Agent] Warming up vector store...")
get_store()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"], temperature=0)

class AgentState(TypedDict):
    messages: list
    user_query: str
    jd_filename: str
    jd_text: str
    requirements: str
    candidate_ids: list[str]
    shortlist: list[dict]
    report: str
    interview_questions: dict
    feedback: str
    current_node: str

def _id_to_filename(cid: str) -> str:
    for f in Path("data/resumes").iterdir():
        if f.stem.upper().startswith(cid.upper()):
            return f.name
    return f"{cid}.txt"

def node_parse_jd(state: AgentState) -> AgentState:
    print("[Node] parse_jd")
    state["current_node"] = "parse_jd"
    jd_dir = Path("data/job_descriptions")
    available = [f.name for f in jd_dir.iterdir() if f.suffix in (".txt", ".pdf")]
    query = state.get("user_query", "").lower()
    matched = next((f for f in available if any(w in f.lower() for w in query.split())), available[0] if available else None)
    if not matched:
        state["jd_text"] = "No JD found."
        return state
    state["jd_filename"] = matched
    state["jd_text"] = read_job_description.invoke({"filename": matched})
    state["messages"].append(AIMessage(content=f"Loaded JD: {matched}"))
    return state

def node_extract_requirements(state: AgentState) -> AgentState:
    print("[Node] extract_requirements")
    state["current_node"] = "extract_requirements"
    jd = state.get("jd_text", "")
    if not jd or "No JD" in jd:
        return state
    reqs = extract_requirements.invoke({"jd": jd})
    state["requirements"] = reqs
    state["messages"].append(AIMessage(content=f"Requirements extracted:\n{reqs}"))
    return state

def node_search_resumes(state: AgentState) -> AgentState:
    print("[Node] search_resumes")
    state["current_node"] = "search_resumes"
    query = f"{state.get('user_query','')} {state.get('requirements','')[:300]}"
    results = search_resumes.invoke({"query": query, "top_k": 6})
    ids = list(dict.fromkeys(re.findall(r'\bC\d{3}\b', results)))
    state["candidate_ids"] = ids
    state["messages"].append(AIMessage(content=f"Found candidates: {', '.join(ids)}\n{results[:500]}"))
    return state

def node_rank_candidates(state: AgentState) -> AgentState:
    print("[Node] rank_candidates")
    state["current_node"] = "rank_candidates"
    shortlist = []
    for cid in state.get("candidate_ids", []):
        resume = read_resume.invoke({"filename": _id_to_filename(cid)})
        if "ERROR" in resume:
            continue
        prompt = f"""Score this candidate for Senior Frontend Developer.

REQUIREMENTS:
{state.get('requirements','')}

RESUME:
{resume}

Reply ONLY with valid JSON (no markdown):
{{"score": <0-100>, "match_level": "<Strong Match|Good Match|Partial Match|Poor Match>", "key_strengths": ["..."], "key_gaps": ["..."], "reasoning": "..."}}"""
        try:
            raw = llm.invoke(prompt).content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            data = json.loads(raw)
            data["candidate_id"] = cid
            shortlist.append(data)
        except Exception as e:
            print(f"  [WARN] Score parse failed for {cid}: {e}")
            shortlist.append({"candidate_id": cid, "score": 50, "match_level": "Unknown", "key_strengths": [], "key_gaps": [], "reasoning": "Manual review needed."})
    shortlist.sort(key=lambda x: x.get("score", 0), reverse=True)
    state["shortlist"] = shortlist
    summary = "\n".join(f"  {i+1}. {s['candidate_id']} — {s['score']}/100 ({s['match_level']})" for i, s in enumerate(shortlist))
    state["messages"].append(AIMessage(content=f"Rankings:\n{summary}"))
    return state

def node_generate_report(state: AgentState) -> AgentState:
    print("[Node] generate_report")
    state["current_node"] = "generate_report"
    shortlist = state.get("shortlist", [])
    if not shortlist:
        state["report"] = "No candidates found."
        return state
    questions = {}
    for c in shortlist[:2]:
        cid = c["candidate_id"]
        questions[cid] = generate_interview_questions.invoke({"candidate_id": cid})
    state["interview_questions"] = questions
    lines = [f"# Resume Matching Report\n**JD:** {state.get('jd_filename','N/A')}\n\n## Ranked Shortlist\n"]
    for i, s in enumerate(shortlist):
        cid = s["candidate_id"]
        lines.append(f"### {i+1}. {cid} — {s['score']}/100 ({s['match_level']})")
        lines.append(f"**Reasoning:** {s['reasoning']}")
        if s.get("key_strengths"): lines.append("**Strengths:** " + ", ".join(s["key_strengths"]))
        if s.get("key_gaps"):      lines.append("**Gaps:** " + ", ".join(s["key_gaps"]))
        if cid in questions:       lines.append(f"\n**Interview Questions:**\n{questions[cid][:400]}...")
        lines.append("")
    top_ids = [s["candidate_id"] for s in shortlist[:3]]
    if len(top_ids) >= 2:
        comparison = compare_candidates.invoke({"candidate_ids": top_ids})
        lines += ["\n## Head-to-Head Comparison\n", comparison]
    state["report"] = "\n".join(lines)
    state["messages"].append(AIMessage(content=state["report"]))
    return state

def node_human_feedback(state: AgentState) -> AgentState:
    print("[Node] human_feedback")
    state["current_node"] = "human_feedback"
    feedback = state.get("feedback", "").strip()
    if not feedback:
        state["messages"].append(AIMessage(content="Report complete! Ask follow-up questions anytime."))
        return state
    prompt = f"""User gave feedback on candidate ranking.

SHORTLIST: {json.dumps(state.get('shortlist',[]), indent=2)}
FEEDBACK: {feedback}

Explain how ranking should change and why. Be specific."""
    response = llm.invoke(prompt).content
    state["messages"].append(HumanMessage(content=f"Feedback: {feedback}"))
    state["messages"].append(AIMessage(content=f"Re-ranking based on feedback:\n{response}"))
    state["feedback"] = ""
    return state

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("parse_jd",             node_parse_jd)
    g.add_node("extract_requirements", node_extract_requirements)
    g.add_node("search_resumes",       node_search_resumes)
    g.add_node("rank_candidates",      node_rank_candidates)
    g.add_node("generate_report",      node_generate_report)
    g.add_node("human_feedback",       node_human_feedback)
    g.set_entry_point("parse_jd")
    g.add_edge("parse_jd",             "extract_requirements")
    g.add_edge("extract_requirements", "search_resumes")
    g.add_edge("search_resumes",       "rank_candidates")
    g.add_edge("rank_candidates",      "generate_report")
    g.add_edge("generate_report",      "human_feedback")
    g.add_edge("human_feedback",       END)
    return g.compile()

def run_matching(user_query: str, feedback: str = "") -> AgentState:
    agent = build_agent()
    state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "jd_filename": "", "jd_text": "", "requirements": "",
        "candidate_ids": [], "shortlist": [],
        "report": "", "interview_questions": {},
        "feedback": feedback, "current_node": "",
    }
    return agent.invoke(state)

if __name__ == "__main__":
    result = run_matching("Find best candidates for senior frontend developer")
    print(result["report"])
