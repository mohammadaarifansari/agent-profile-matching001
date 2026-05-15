import fitz
from pathlib import Path
from langchain.tools import tool

RESUMES_DIR = Path(__file__).parent.parent / "data" / "resumes"
JD_DIR      = Path(__file__).parent.parent / "data" / "job_descriptions"

def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")

@tool
def list_resumes() -> list[str]:
    """List all resume filenames."""
    return sorted(f.name for f in RESUMES_DIR.iterdir() if f.suffix in (".txt", ".pdf"))

@tool
def read_resume(filename: str) -> str:
    """Read a resume by filename."""
    path = RESUMES_DIR / filename
    return _read_file(path) if path.exists() else f"ERROR: {filename} not found."

@tool
def list_job_descriptions() -> list[str]:
    """List all JD filenames."""
    return sorted(f.name for f in JD_DIR.iterdir() if f.suffix in (".txt", ".pdf"))

@tool
def read_job_description(filename: str) -> str:
    """Read a job description by filename."""
    path = JD_DIR / filename
    return _read_file(path) if path.exists() else f"ERROR: {filename} not found."
