import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateInput(BaseModel):
    prompt: str
    mode: Optional[str] = "suggest"  # suggest | outline | tasks | clarify | rewrite | expand | critique
    text: Optional[str] = None  # optional source text for transforms


class GenerateResponse(BaseModel):
    suggestions: List[str]
    provider: str
    used_backend: bool


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


# -------------------------
# AI generation proxy
# -------------------------

def _heuristic_generate(data: GenerateInput) -> List[str]:
    prompt = (data.prompt or "").strip()
    source = (data.text or "").strip()
    mode = (data.mode or "suggest").lower()

    def extract_keywords(text: str) -> List[str]:
        import re
        words = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
        stop = {"with","that","this","from","have","about","into","your","their","there","which","and","the","for"}
        uniq = []
        for w in words:
            if len(w) > 3 and w not in stop and w not in uniq:
                uniq.append(w)
        return uniq[:6]

    if mode == "outline":
        base = prompt.rstrip('.')
        return [
            f"Objective: {base}",
            "Audience: Who benefits and why",
            "Success: Measurable outcomes",
            "Flow: Discover → Co-create → Iterate → Share",
        ]
    if mode == "tasks":
        kw = extract_keywords(prompt)
        return [
            "Draft a compelling headline",
            "Create a two-sentence value proposition",
            f"List 3 user outcomes{' about ' + ', '.join(kw) if kw else ''}",
            "Design the primary CTA and success metric",
        ]
    if mode in {"clarify", "rewrite"}:
        text = source or prompt
        cleaned = (
            text.replace(" really", "").replace(" very", "").replace(" actually", "")
            .replace(" basically", "").replace(" simply", "")
        )
        lines = [s if len(s) <= 120 else s[:117] + "…" for s in cleaned.split('. ')]
        return [". ".join(lines).strip()]
    if mode == "expand":
        text = source or prompt
        parts = [p for p in text.split('. ') if p]
        extra = " Add a clear outcome, a user benefit, and a success metric."
        return [" ".join([(p.rstrip('.') + ':' + extra) for p in parts])]
    if mode == "critique":
        checks = [
            "Is the goal specific and testable?",
            "Who is the audience and what do they gain?",
            "Is there a metric to track success?",
            "Is the language inclusive and clear?",
        ]
        note = source or prompt
        return ["Critique Checklist\n- " + "\n- ".join(checks) + f"\nNotes: {note[:160] + ('…' if len(note)>160 else '')}"]

    # default suggest
    kw = extract_keywords(prompt + " " + source)
    headline = " + ".join([w.capitalize() for w in kw[:2]]) or "Human + AI"
    return [
        f"{headline}: co-create outcomes, not just outputs",
        "Add a short success metric under each idea",
        "Invite feedback loops: draft → test → refine",
    ]


def _call_openai(messages: List[dict]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "messages": messages,
                "temperature": 0.4,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


def _call_anthropic(messages: List[dict]) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    # Convert OpenAI-style messages to Anthropic format
    try:
        # messages: [{role, content}]
        system = "You are a concise product ideation assistant. Return short, bulleted suggestions."
        content = []
        for m in messages:
            if m["role"] == "user":
                content.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                content.append({"role": "assistant", "content": m["content"]})
            elif m["role"] == "system":
                system = m["content"]
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
                "max_tokens": 400,
                "system": system,
                "messages": content,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Anthropic returns a list of content blocks
        return "".join([blk.get("text", "") for blk in data.get("content", [])])
    except Exception:
        return None


def _call_grok(messages: List[dict]) -> Optional[str]:
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        return None
    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("GROK_MODEL", "grok-2-latest"),
                "messages": messages,
                "temperature": 0.4,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible schema
        return data.get("choices", [{}])[0].get("message", {}).get("content")
    except Exception:
        return None


def _parse_bullets(text: str) -> List[str]:
    if not text:
        return []
    # Split on lines/bullets
    lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
    # further split if it's a single paragraph with bullets separated by ';'
    if len(lines) <= 1 and ";" in text:
        lines = [s.strip() for s in text.split(";") if s.strip()]
    # cap and clean
    out = []
    for l in lines:
        if len(l) > 240:
            l = l[:237] + "…"
        out.append(l)
    return out[:8]


@app.post("/api/generate", response_model=GenerateResponse)
def generate(data: GenerateInput):
    """Smart generator that uses Grok/OpenAI/Claude if keys are present, otherwise falls back to heuristics."""
    if not data.prompt and not data.text:
        raise HTTPException(status_code=400, detail="prompt or text required")

    # Build instruction
    task = data.mode or "suggest"
    system = (
        "You are a product co-creation assistant. Return crisp, actionable bullets for the requested mode. "
        "Keep outputs short and scannable. Avoid markdown numbering."
    )
    user = f"Mode: {task}\nPrompt: {data.prompt}\nText: {data.text or ''}\nReturn 3-6 concise suggestions or the transformed text."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    provider = "heuristic"
    content = None

    # Try providers in order: Grok -> OpenAI -> Anthropic
    content = _call_grok(messages)
    if content:
        provider = "grok"
    else:
        content = _call_openai(messages)
        if content:
            provider = "openai"
        else:
            content = _call_anthropic(messages)
            if content:
                provider = "anthropic"

    if content:
        suggestions = _parse_bullets(content)
        if not suggestions:
            suggestions = [content.strip()][:1]
        return GenerateResponse(suggestions=suggestions, provider=provider, used_backend=True)

    # Heuristic fallback
    fallback = _heuristic_generate(data)
    return GenerateResponse(suggestions=fallback, provider=provider, used_backend=False)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
