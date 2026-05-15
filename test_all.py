import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def ok(m):   print(f"  ✅ {m}")
def fail(m): print(f"  ❌ {m}")
def section(t): print(f"\n{'='*50}\n  TEST: {t}\n{'='*50}")

section("File Tools")
try:
    from tools.file_tools import list_resumes, list_job_descriptions, read_resume
    r = list_resumes.invoke({})
    assert len(r) >= 1
    ok(f"Resumes: {r}")
    t = read_resume.invoke({"filename": r[0]})
    assert len(t) > 50
    ok(f"Read resume: {len(t)} chars")
except Exception as e: fail(e)

section("RAG / Vector Store")
try:
    from tools.rag_tools import build_vector_store, search_resumes
    store = build_vector_store(force_rebuild=True)
    ok(f"Store built: {store._collection.count()} chunks")
    results = search_resumes.invoke({"query": "React TypeScript", "top_k": 3})
    ok(f"Search returned {len(results)} chars")
    print(f"\n  Preview: {results[:200]}")
except Exception as e: fail(e)

section("Agent Tools (needs GROQ_API_KEY)")
try:
    from tools.agent_tools import extract_requirements, generate_interview_questions
    jd = Path("data/job_descriptions/senior_frontend_dev.txt").read_text()
    reqs = extract_requirements.invoke({"jd": jd})
    ok(f"extract_requirements: {len(reqs)} chars")
    qs = generate_interview_questions.invoke({"candidate_id": "C001"})
    ok(f"interview questions: {len(qs)} chars")
except Exception as e: fail(e)

section("Full Agent Pipeline")
try:
    from matching_agent import run_matching
    result = run_matching("Find best candidates for senior frontend developer")
    assert result["shortlist"]
    ok(f"Pipeline complete. {len(result['shortlist'])} candidates ranked.")
    ok(f"Top: {result['shortlist'][0]['candidate_id']} — {result['shortlist'][0].get('score')}/100")
    print(f"\n  Report preview:\n{result['report'][:400]}")
except Exception as e: fail(e)

print(f"\n{'='*50}\n  Done! Run: python app.py\n{'='*50}\n")
