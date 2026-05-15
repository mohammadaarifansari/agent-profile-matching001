import os
from pathlib import Path
from langchain.tools import tool
from langchain_groq import ChatGroq

BASE_DIR    = Path(__file__).parent.parent
RESUMES_DIR = BASE_DIR / "data" / "resumes"

_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"], temperature=0)

def _read_resume(candidate_id: str) -> str:
    for path in RESUMES_DIR.iterdir():
        if path.stem.upper().startswith(candidate_id.upper()):
            return path.read_text(encoding="utf-8")
    return f"Resume for {candidate_id} not found."

@tool
def extract_requirements(jd: str) -> str:
    """Parse a job description into must-have and nice-to-have requirements."""
    prompt = f"""Analyze this JD and extract requirements in this exact format:

MUST-HAVE:
- (list each mandatory requirement)

NICE-TO-HAVE:
- (list each preferred requirement)

KEY_RESPONSIBILITIES:
- (list main responsibilities)

EXPERIENCE_YEARS: (minimum years)
EDUCATION: (required level)

JD:
{jd}

Reply ONLY with the structured format above."""
    return _llm.invoke(prompt).content

@tool
def compare_candidates(candidate_ids: list[str]) -> str:
    """Compare multiple candidates head-to-head. Pass list of candidate IDs like ['C001','C002']."""
    resumes_text = "\n\n===\n\n".join(f"CANDIDATE {cid}:\n{_read_resume(cid)}" for cid in candidate_ids)
    prompt = f"""Compare these candidates for a Senior Frontend Developer role.

{resumes_text}

Provide:
1. Comparison table (React exp, TypeScript, Testing, State Mgmt, Education)
2. Strengths per candidate
3. Gaps per candidate
4. Final ranking with reasoning"""
    return _llm.invoke(prompt).content

@tool
def generate_interview_questions(candidate_id: str) -> str:
    """Generate tailored interview questions for a specific candidate."""
    resume = _read_resume(candidate_id)
    if "not found" in resume:
        return resume
    prompt = f"""Generate 8 targeted interview questions for this candidate.

RESUME:
{resume}

Categories:
- TECHNICAL DEPTH (3 questions): probe their stated skills deeply
- EXPERIENCE VALIDATION (2 questions): verify specific resume claims
- GAP PROBING (2 questions): address missing skills
- BEHAVIORAL (1 question): situation-based

For each question add: → What to listen for: ..."""
    return _llm.invoke(prompt).content
