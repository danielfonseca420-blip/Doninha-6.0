"""
CAMADA L6  Resposta Final em Texto Fluido
==========================================
Transforma o output estruturado das camadas L1L5 em um texto contnuo,
claro e preciso, com tom profissional e acessvel.

Fluxo de processamento:
  Motor de Raciocnio  Output Estruturado  Sntese  Resposta Final
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from llm_provider_client import generate_text as generate_provider_text
from l4_synthesis import SynthesisResult
from layer_titles import LAYER_TITLES

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore

try:
    from l5_generation import generate_with_custom_lm, generate_with_ollama
except Exception:
    generate_with_custom_lm = None  # type: ignore
    generate_with_ollama = None  # type: ignore


@dataclass
class EpistemicContext:
    """Contexto epistemolgico agregado para L6."""
    proposition_states: List[Dict[str, Any]] = field(default_factory=list)
    many_valued_routes: List[Dict[str, Any]] = field(default_factory=list)
    bert_classifications: List[Dict[str, Any]] = field(default_factory=list)
    application_context: str = ""


class FinalResponseEngine:
    """Gera a resposta final em texto fluido a partir da sntese das camadas anteriores."""

    def finalize_response(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        epistemic_context: Optional[EpistemicContext] = None,
        generated_text: str = "",
        concepts_summary: str = "",
        top_judgments: str = "",
        agent_context: str = "",
    ) -> str:
        """Produz a resposta final nica e contnua seguindo regras de redao clara."""
        main_text = self._normalize_text(generated_text or synthesis_result.response or "")
        if not main_text:
            return "No h informao suficiente para formular uma resposta final."

        intro = self._build_intro(synthesis_result)
        conclusion = self._build_conclusion(synthesis_result)
        context_note = self._build_context_note(concepts_summary, top_judgments, agent_context, epistemic_context)

        if intro:
            if main_text.lower().startswith(intro.lower()):
                final = main_text
            else:
                final = f"{intro} {main_text}"
        else:
            final = main_text

        if conclusion:
            final = f"{final} {conclusion}"

        if context_note:
            final = f"{final} {context_note}"

        return final.strip()

    def _build_context_note(
        self,
        concepts_summary: str,
        top_judgments: str,
        agent_context: str,
        epistemic_context: Optional[EpistemicContext] = None,
    ) -> str:
        note_parts = []
        if concepts_summary:
            note_parts.append("o raciocnio integrou conceitos extrados e evidncias relevantes")
        if top_judgments:
            note_parts.append("os juzos kantianos foram usados para priorizar hipteses")
        if agent_context:
            note_parts.append("informaes de busca externas tambm foram consideradas")
        if epistemic_context is not None:
            if epistemic_context.application_context:
                note_parts.append("o contexto aplicacional do LogicLMSolver tambm foi considerado")
            if epistemic_context.many_valued_routes:
                note_parts.append("as rotas paraconsistentes do ManyValuedRouter foram analisadas")
            if epistemic_context.bert_classifications:
                note_parts.append("classificaes BERT (T/I/F) das principais hipteses influenciaram a formulao")
        if not note_parts:
            return ""
        return "Essa resposta reflete o processamento integrado das camadas anteriores, com ateno a evidncias e juzos relevantes."  

    # ------------------------------------------------------------------ #
    # Componentes textuais adaptativos                                    #
    # ------------------------------------------------------------------ #

    def _build_intro(self, synthesis_result: SynthesisResult) -> str:
        if synthesis_result.truth_value >= 0.85:
            return "Com base no motor de raciocnio L1L5, a melhor concluso indica"  
        if synthesis_result.truth_value >= 0.65:
            return "A partir da sntese das camadas L1L5, o cenrio mais slido sugere"  
        if synthesis_result.truth_value >= 0.45:
            return "Com certa cautela, a anlise das camadas L1L5 aponta"  
        return "A anlise das camadas L1L5 indica"  

    def _build_conclusion(self, synthesis_result: SynthesisResult) -> str:
        if synthesis_result.state in {"Indeterminado", "N"}:
            return (
                "Esta questo tem uma dimenso genuinamente indeterminada  no por falta de rigor, "
                "mas porque a evidncia emprica ainda no existe. Isso  diferente de 'no sabemos' "
                "  'no h dados para saber'."
            )
        if synthesis_result.state == "Inconsistente_local" or synthesis_result.contradiction > 0.25:
            return "Essa concluso  apresentada como a melhor interpretao disponvel, embora exista uma contradio local que recomenda prudncia."  
        if synthesis_result.truth_value < 0.65:
            return "Dado o grau de incerteza, vale considerar essa resposta como provisria at que evidncias adicionais sejam avaliadas."  
        return ""

    def _normalize_text(self, text: str) -> str:
        normalized_lines = []
        for line in text.splitlines():
            line = " ".join(line.split()).strip()
            if line:
                normalized_lines.append(line)
        return "\n\n".join(normalized_lines)

    def _ensure_single_paragraph(self, text: str) -> str:
        normalized_lines = []
        for line in text.splitlines():
            line = " ".join(line.split()).strip()
            if line:
                normalized_lines.append(line)
        return "\n\n".join(normalized_lines)

    def rewrite_response(
        self,
        prompt: str,
        synthesis_result: SynthesisResult,
        epistemic_context: Optional[EpistemicContext] = None,
        generated_text: str = "",
        concepts_summary: str = "",
        top_judgments: str = "",
        agent_context: str = "",
        provider: str = "template",
        custom_lm_path: str = "",
        ollama_model: str = "doninha8:latest",
        ollama_host: str = "http://localhost:11434",
        base_url: str = "",
        api_key: str = "",
    ) -> str:
        """Refina a resposta final com um segundo prompt no estilo de um agente escritor."""
        draft = self.finalize_response(
            prompt=prompt,
            synthesis_result=synthesis_result,
            epistemic_context=epistemic_context,
            generated_text=generated_text,
            concepts_summary=concepts_summary,
            top_judgments=top_judgments,
            agent_context=agent_context,
        )
        writer_prompt = self._build_writer_prompt(
            prompt,
            draft,
            synthesis_result,
            epistemic_context,
            concepts_summary,
            top_judgments,
            agent_context,
        )

        if provider == "ollama" and generate_with_ollama:
            text = generate_with_ollama(writer_prompt, model=ollama_model, ollama_host=ollama_host, temperature=0.3)
            if text:
                return text.strip()

        if provider == "custom_lm" and generate_with_custom_lm and custom_lm_path:
            text = generate_with_custom_lm(writer_prompt, custom_lm_path)
            if text:
                return text.strip()

        if provider in {"openai", "anthropic", "gemini", "grok", "groq", "meta"}:
            text = generate_provider_text(
                provider=provider,
                prompt=writer_prompt,
                model=ollama_model or None,
                base_url=base_url or None,
                api_key=api_key or None,
                temperature=0.3,
            )
            if text:
                return text.strip()

        return self._polish_writer_text(draft)

    def _build_writer_prompt(
        self,
        prompt: str,
        draft: str,
        synthesis_result: SynthesisResult,
        epistemic_context: Optional[EpistemicContext],
        concepts_summary: str,
        top_judgments: str,
        agent_context: str,
    ) -> str:
        lines = [
            "Voc  um agente escritor tcnico e comunicador.",
            "Transforme o raciocnio completo gerado pelas camadas L1 a L6 em um texto fluido, natural, coeso e fcil de ler.",
            "Respeite as proposies e concluses encontradas entre L1 e L6 e mantenha o rigor lgico e tcnico.",
            "Comece direto pela resposta principal, depois explique o caminho se necessrio.",
            "Use linguagem clara, conversacional e precisa e mencione incertezas ou trade-offs de forma elegante quando existirem.",
            "No separe o texto em passos numerados ou listas.",
            "Ao se referir s etapas, use os ttulos de seo designados abaixo:",
            f"L1: {LAYER_TITLES['l1']}",
            f"L2: {LAYER_TITLES['l2']}",
            f"L3: {LAYER_TITLES['l3']}",
            f"L4: {LAYER_TITLES['l4']}",
            f"L5: {LAYER_TITLES['l5']}",
            f"L6: {LAYER_TITLES['l6']}",
            "",
            f"Pergunta do usurio: {prompt}",
            "",
            "Texto preliminar:",
            draft,
            "",
            "Contexto de sntese:",
            f"Resposta de sntese L4: {synthesis_result.response}",
            f"Valor de verdade: {synthesis_result.truth_value:.2f}",
            f"Estado: {synthesis_result.state}",
            f"Certeza: {synthesis_result.certainty:+.2f}",
            f"Contradio: {synthesis_result.contradiction:+.2f}",
        ]
        if concepts_summary:
            lines.extend(["", f"Conceitos L1: {concepts_summary}"])
        if top_judgments:
            lines.extend(["", f"Juzos L2: {top_judgments}"])
        if agent_context:
            lines.extend(["", "Contexto de busca externo:", agent_context])
        if epistemic_context is not None:
            lines.extend(["", "Detalhes epistemolgicos:", self._summarize_epistemic_context(epistemic_context)])
        lines.extend([
            "",
            "Raciocnio completo:",
            "O texto deve sintetizar a extrao de conceitos, os juzos kantianos, a avaliao paraconsistente, a sntese russelliana e a formulao final.",
        ])
        base_prompt = "\n".join(lines)
        if get_layer_prompt is None:
            return base_prompt
        return get_layer_prompt(
            "l6",
            prompt,
            {
                "l5_text": draft,
                "full_context": base_prompt,
            },
        )

    def _summarize_epistemic_context(self, epistemic_context: EpistemicContext) -> str:
        parts: List[str] = []
        if epistemic_context.application_context:
            parts.append(f"Contexto de aplicao: {epistemic_context.application_context}")
        if epistemic_context.proposition_states:
            top_props = epistemic_context.proposition_states[:3]
            summary = ", ".join(
                f"{item.get('state', 'Desconhecido')} ({item.get('truth_value', 'n/a')})" for item in top_props
            )
            parts.append(f"Proposies avaliadas: {summary}")
        if epistemic_context.many_valued_routes:
            parts.append("Rotas paraconsistentes avaliadas.")
        if epistemic_context.bert_classifications:
            parts.append("Classificaes BERT (T/I/F) foram usadas para ajustar prioridades epistemolgicas.")
        return " ".join(parts)

    def _polish_writer_text(self, draft: str) -> str:
        return draft.strip()
