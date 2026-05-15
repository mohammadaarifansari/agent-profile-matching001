import fitz
from pathlib import Path
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

BASE_DIR    = Path(__file__).parent.parent
RESUMES_DIR = BASE_DIR / "data" / "resumes"
VECTOR_DIR  = str(BASE_DIR / "vector_store")
EMBEDDINGS  = OllamaEmbeddings(model="nomic-embed-text")

def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")

def build_vector_store(force_rebuild: bool = False) -> Chroma:
    db = Chroma(collection_name="resumes", embedding_function=EMBEDDINGS, persist_directory=VECTOR_DIR)
    if not force_rebuild and db._collection.count() > 0:
        print(f"[RAG] Loaded existing store ({db._collection.count()} chunks)")
        return db
    print("[RAG] Building vector store...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    for p in RESUMES_DIR.iterdir():
        if p.suffix.lower() not in (".txt", ".pdf"):
            continue
        for i, chunk in enumerate(splitter.split_text(_read_file(p))):
            docs.append(Document(page_content=chunk, metadata={
                "source": p.name,
                "candidate_id": p.stem.split("_")[0],
                "chunk_index": i
            }))
    db.add_documents(docs)
    print(f"[RAG] Indexed {len(docs)} chunks")
    return db

_store = None

def get_store() -> Chroma:
    global _store
    if _store is None:
        _store = build_vector_store()
    return _store

@tool
def search_resumes(query: str, top_k: int = 5) -> str:
    """Semantic search across all resumes. Use for finding candidates by skills/experience."""
    results = get_store().similarity_search_with_score(query, k=top_k)
    if not results:
        return "No matching candidates found."
    seen, output = set(), []
    for doc, score in results:
        cid = doc.metadata.get("candidate_id", "?")
        if cid not in seen:
            seen.add(cid)
            output.append(f"[{cid}] {doc.metadata.get('source')} | Relevance: {round((1-score)*100,1)}%\n{doc.page_content[:300]}")
    return "\n\n---\n\n".join(output)
