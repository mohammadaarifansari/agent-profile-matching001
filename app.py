import os, json, re
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from matching_agent import run_matching
from tools.file_tools  import list_resumes, list_job_descriptions
from tools.agent_tools import compare_candidates, generate_interview_questions

console = Console()
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"], temperature=0)
session = {"shortlist": [], "report": "", "jd_text": "", "requirements": "", "user_query": ""}

def classify_intent(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["find","search","match","best candidate","screen"]): return "full_match"
    if any(w in q for w in ["compare","side by side","vs","versus"]):            return "compare"
    if any(w in q for w in ["why","explain","rank higher","rank lower","score"]): return "explain"
    if any(w in q for w in ["interview","question","ask"]):                       return "interview"
    if any(w in q for w in ["rerank","re-rank","prioritize","adjust","focus on"]): return "rerank"
    if any(w in q for w in ["list","available","files"]):                         return "list"
    return "chat"

def handle_full_match(query):
    console.print(Panel("Running full pipeline... (~30-60 sec)", style="yellow"))
    result = run_matching(query)
    session.update({"shortlist": result["shortlist"], "report": result["report"],
                    "jd_text": result["jd_text"], "requirements": result["requirements"], "user_query": query})
    console.print(Markdown(result["report"]))
    feedback = Prompt.ask("\n[bold cyan]Any feedback on ranking? (Enter to skip)[/]", default="")
    if feedback:
        result2 = run_matching(query, feedback=feedback)
        session["shortlist"] = result2["shortlist"]
        console.print(Panel(result2["messages"][-1].content, title="Updated Ranking", style="cyan"))

def handle_compare(query):
    ids = re.findall(r'\bC\d{3}\b', query.upper()) or [s["candidate_id"] for s in session["shortlist"][:3]]
    if len(ids) < 2:
        console.print("[red]Need 2+ candidates. Run a search first.[/]"); return
    console.print(Markdown(compare_candidates.invoke({"candidate_ids": ids})))

def handle_explain(query):
    if not session["shortlist"]:
        console.print("[red]No ranking yet. Run a search first.[/]"); return
    prompt = f"SHORTLIST: {json.dumps(session['shortlist'], indent=2)}\nQUESTION: {query}\nExplain with specific scores and skills."
    console.print(Panel(llm.invoke(prompt).content, title="Explanation", style="green"))

def handle_interview(query):
    ids = re.findall(r'\bC\d{3}\b', query.upper())
    cid = ids[0] if ids else (session["shortlist"][0]["candidate_id"] if session["shortlist"] else None)
    if not cid:
        console.print("[red]Specify a candidate or run a search first.[/]"); return
    console.print(Markdown(generate_interview_questions.invoke({"candidate_id": cid})))

def handle_list():
    t = Table(title="Available Files")
    t.add_column("Type", style="cyan"); t.add_column("File", style="white")
    for f in list_job_descriptions.invoke({}): t.add_row("Job Description", f)
    for f in list_resumes.invoke({}):          t.add_row("Resume", f)
    console.print(t)

def handle_chat(query):
    ctx = f"Shortlist: {json.dumps(session['shortlist'])}" if session["shortlist"] else "No search done yet."
    console.print(Panel(llm.invoke(f"Context: {ctx}\nUser: {query}\nAnswer helpfully.").content, style="blue"))

HELP = """
[bold cyan]Commands:[/]
  • Find candidates for senior frontend role  → full pipeline
  • Compare C001 C002 C003                   → side-by-side
  • Why did C001 rank higher than C002?      → explain ranking
  • Generate interview questions for C001    → screening Qs
  • Re-rank prioritizing GraphQL             → adjust criteria
  • List available files                     → show all files
  • help / quit
"""

def main():
    console.print(Panel.fit("[bold green]Resume Matching Agent[/]\nLangGraph + Groq + ChromaDB", border_style="green"))
    console.print(HELP)
    while True:
        try:
            query = Prompt.ask("\n[bold green]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query: continue
        if query.lower() in ("quit","exit","q"): break
        if query.lower() == "help": console.print(HELP); continue
        intent = classify_intent(query)
        console.print(f"[dim]Intent: {intent}[/]")
        try:
            if   intent == "full_match": handle_full_match(query)
            elif intent == "compare":    handle_compare(query)
            elif intent == "explain":    handle_explain(query)
            elif intent == "interview":  handle_interview(query)
            elif intent == "rerank":     handle_full_match(query)
            elif intent == "list":       handle_list()
            else:                        handle_chat(query)
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

if __name__ == "__main__":
    main()
