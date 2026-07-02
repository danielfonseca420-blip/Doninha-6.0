"""
API REST para o Modelo Híbrido - Compatível com Open WebUI
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====================== CONFIGURAÇÃO ======================
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = FastAPI(title="Doninha Hybrid LLM API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080", "http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== MODELOS ======================
class ProcessRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None
    use_agent: Optional[bool] = False
    skip_l5: bool = False

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ProcessResponse(BaseModel):
    response: str
    truth_value: float = 0.0
    state: str = ""
    certainty: float = 0.0
    contradiction: float = 0.0
    confidence_label: str = ""
    session_id: Optional[str] = None


# ====================== CARREGAMENTO ======================
_config = None
_pipeline = None
_sessions: Dict[str, Any] = {}
_max_turns = 10

try:
    from config_loader import load_config
    from pipeline import HybridLLMPipeline
    from chat_session import ChatSession

    _config = load_config()
    _pipeline = HybridLLMPipeline(config=_config, verbose=False)
    _max_turns = _config.get("chat", {}).get("max_turns_in_context", 10)
except Exception as e:
    print(f"[AVISO] Erro ao carregar módulos: {e}")


# ====================== ROTAS ======================
@app.get("/health")
async def health():
    return {"status": "ok", "model": "hybrid_llm"}


@app.get("/ports")
async def get_ports():
    return {
        "status": "ok",
        "message": "Tool server running",
        "available_tools": ["process", "chat", "agent"]
    }


@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    if not _pipeline:
        raise HTTPException(500, "Pipeline não carregado")

    session_id = req.session_id or str(uuid4())
    # ... (resto da lógica da sessão)
    try:
        result = _pipeline.process(
            req.prompt,
            chat_session=None,   # ajuste conforme necessário
            use_agent=req.use_agent,
            skip_l5=req.skip_l5,
        )
        return ProcessResponse(
            response=result.response,
            truth_value=getattr(result, 'truth_value', 0.0),
            state=getattr(result, 'state', ''),
            certainty=getattr(result, 'certainty', 0.0),
            contradiction=getattr(result, 'contradiction', 0.0),
            confidence_label=getattr(result, 'confidence_label', ''),
            session_id=session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def openai_compatible(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        prompt = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), 
            "Olá"
        )

        if not _pipeline:
            content = "Pipeline não carregado. API está em modo básico."
        else:
            result = _pipeline.process(prompt)
            content = getattr(result, 'response', str(result))

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "doninha-hybrid",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 120,
                "total_tokens": len(prompt.split()) + 120
            }
        }
    except Exception as e:
        print(f"[ERRO /v1/chat/completions] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/v1/openapi.json")
async def openapi_json():
    """Endpoint necessário para o Open WebUI reconhecer a API como OpenAI-compatible"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Doninha Hybrid LLM API",
            "version": "1.0"
        },
        "servers": [{"url": "/v1"}],
        "paths": {
            "/chat/completions": {
                "post": {
                    "summary": "Chat Completion",
                    "operationId": "chatCompletion",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "messages": {"type": "array"},
                                        "model": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "doninha",
                "object": "model",
                "created": 1700000000,
                "owned_by": "doninha-ia",
                "permission": [],
                "root": "doninha",
                "parent": None
            }
        ]
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)