"""
Chain of Verification (CoVe) agent para a camada L4.
====================================================
Implementa o workflow Factor + Revise como etapa adicional de verificao
da sntese L4 antes da resposta final ser entregue.
"""

from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from llm_provider_client import generate_text as generate_provider_text

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore


def generate_with_ollama_l4(
    context: str,
    model: str = "doninha8:latest",
    ollama_host: str = "http://localhost:11434",
    temperature: float = 0.2,
) -> str:
    """Gera resposta usando Ollama especificamente para a camada L4."""
    try:
        import ollama
        
        if ollama_host:
            os.environ["OLLAMA_HOST"] = ollama_host
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": context}],
            stream=False,
            think="high",
            options={
                "temperature": temperature,
                "num_ctx": 8192,
            },
        )
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "").strip()
        return str(response).strip() if response else ""
    except Exception:
        return ""


def generate_with_custom_lm(context: str, model_path: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
    try:
        from custom_lm_model import EpistemicLanguageModel, LMConfig, generate_text, load_lm
        from custom_tokenizer import CustomSPTokenizer, SPConfig
        import torch
        tokenizer = CustomSPTokenizer(SPConfig())
        tokenizer.load()
        vocab_size = tokenizer.vocab_size()
        model = load_lm(model_path, vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = generate_text(model, tokenizer, context, max_new_tokens=max_new_tokens, temperature=temperature, device=device)
        return out or ""
    except Exception:
        return ""


class ChainOfVerificationAgent:
    """Agente de Chain of Verification para a camada L4."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.provider = self.config.get("provider", "template")
        self.ollama_model = self.config.get("ollama_model", "doninha8:latest")
        self.ollama_host = self.config.get("ollama_host", "http://localhost:11434")
        self.custom_lm_path = self.config.get("custom_lm_path", "")

    def verify(
        self,
        prompt: str,
        baseline_response: str,
        context_summary: str,
    ) -> Tuple[str, List[str]]:
        """Executa o workflow CoVe e retorna resposta revisada + log."""
        if self.provider == "ollama":
            output = self._verify_with_ollama(prompt, baseline_response, context_summary)
        elif self.provider == "custom_lm" and self.custom_lm_path:
            output = self._verify_with_custom_lm(prompt, baseline_response, context_summary)
        elif self.provider in {"openai", "anthropic", "gemini", "grok", "groq", "meta"}:
            output = self._verify_with_provider(prompt, baseline_response, context_summary)
        else:
            output = self._template_verify(prompt, baseline_response, context_summary)

        revised, log = self._parse_verification_output(output, baseline_response)
        return revised, log

    def _verify_with_ollama(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        verification_prompt = self._build_agent_prompt(prompt, baseline_response, context_summary)
        return generate_with_ollama_l4(
            verification_prompt,
            model=self.ollama_model,
            ollama_host=self.ollama_host,
        )

    def _verify_with_custom_lm(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        verification_prompt = self._build_agent_prompt(prompt, baseline_response, context_summary)
        return generate_with_custom_lm(verification_prompt, self.custom_lm_path)

    def _verify_with_provider(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        verification_prompt = self._build_agent_prompt(prompt, baseline_response, context_summary)
        return generate_provider_text(
            provider=self.provider,
            prompt=verification_prompt,
            model=self.config.get("model") or self.ollama_model,
            base_url=self.config.get("base_url") or None,
            api_key=self.config.get("api_key") or None,
        )

    def _template_verify(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        claims = self._extract_claims(baseline_response)
        questions = self._build_verification_questions(claims)
        verifications = [f"{idx+1}. {q}  Incerto; verificao externa necessria." for idx, q in enumerate(questions)]
        revised = baseline_response.strip()
        if verifications:
            revised += "\n\nNota: esta resposta foi revisada com base em verificao interna limitada; algumas afirmaes permanecem pendentes de confirmao externa."
        sections = [
            "Baseline Response:",
            baseline_response.strip(),
            "",
            "Verification Questions:",
            *questions,
            "",
            "Independent Verification Results:",
            *verifications,
            "",
            "Cross-Check & Revise:",
            "Nenhuma inconsistncia formal identificada no contedo disponvel localmente.",
            "",
            "Revised Response:",
            revised,
        ]
        return "\n".join(sections)

    def _build_agent_prompt(self, prompt: str, baseline_response: str, context_summary: str) -> str:
        if get_layer_prompt is not None:
            cot_header = get_layer_prompt(
                "l4",
                prompt,
                {
                    "l3_summary": context_summary,
                    "kb_summary": self.config.get("kb_summary", "KB local e base Russelliana disponiveis quando configuradas."),
                },
            )
        else:
            cot_header = ""
        lines = [
            cot_header,
            "",
            "Voc  um engenheiro de prompts especialista em tcnicas avanadas de confiabilidade.",
            "A partir de agora, use o mtodo Chain of Verification (CoVe) - variante Factor + Revise para analisar e revisar a resposta.",
            "Responda usando sempre o fluxo: 1. Baseline Response, 2. Factoring, 3. Independent Verification, 4. Cross-Check & Revise.",
            "Seja rigoroso, conservador, e declare limitaes quando necessrio.",
            "",
            f"Pergunta original: {prompt}",
            "",
            "Contexto resumido de L4: ",
            context_summary or "Sem contexto adicional disponvel.",
            "",
            "Resposta inicial (Baseline Response):",
            baseline_response.strip(),
            "",
            "Tarefa:",
            "1. Gere de 6 a 12 perguntas de verificao independentes a partir das principais afirmaes da resposta inicial.",
            "2. Responda cada pergunta de forma independente, marcando como Confirmado, Refutado, Parcialmente correto ou Incerto.",
            "3. Compare a resposta inicial com os resultados e reescreva a resposta final incorporando apenas o que foi verificado.",
            "4. Entregue a estrutura completa com as sees claramente demarcadas e finalize com a resposta revisada.",
            "",
            "Formato de sada exigido:",
            "Baseline Response:",
            "<texto>",
            "",
            "Verification Questions:",
            "1. <pergunta>",
            "...",
            "",
            "Independent Verification Results:",
            "1. <marcao>  <resposta>",
            "...",
            "",
            "Cross-Check & Revise:",
            "<anlise>",
            "",
            "Revised Response:",
            "<texto revisado>",
        ]
        return "\n".join(lines)

    def _parse_verification_output(self, output: str, baseline_response: str) -> Tuple[str, List[str]]:
        if not output:
            return baseline_response, ["Nenhuma sada de verificao gerada."]

        revised = baseline_response
        log: List[str] = []
        if "Revised Response:" in output:
            parts = output.split("Revised Response:")
            revised = parts[-1].strip()
            log = [line.strip() for line in output.splitlines() if line.strip()]
        else:
            log = [line.strip() for line in output.splitlines() if line.strip()]
        return revised, log

    def _extract_claims(self, text: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]
        claims = []
        for sentence in sentences:
            if len(claims) >= 12:
                break
            if len(sentence.split()) >= 5:
                claims.append(sentence)
        return claims[:12] if claims else sentences[:min(6, len(sentences))]

    def _build_verification_questions(self, claims: List[str]) -> List[str]:
        questions: List[str] = []
        for claim in claims[:12]:
            question = f"A afirmao a seguir est correta e fundamentada? {claim}"
            questions.append(question)
        if len(questions) < 6:
            questions.extend([
                "A estrutura lgica da resposta est consistente com a informao disponvel?",
                "H alguma suposio implcita que precisa ser explicitada ou verificada?",
            ])
        return questions[:12]
