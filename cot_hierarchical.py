"""
Rastro hierarquico de raciocinio auditavel para Doninha IA (L1 a L7).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time


@dataclass
class CoTStep:
    layer: str
    title: str
    thought_process: str
    key_decisions: List[str]
    output_summary: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class HierarchicalCoTTrace:
    prompt: str
    steps: List[CoTStep] = field(default_factory=list)
    final_synthesis: str = ""
    overall_confidence: float = 0.0
    total_layers: int = 7
    started_at: float = field(default_factory=time.time)

    def add_step(
        self,
        layer: str,
        title: str,
        thought: str,
        decisions: List[str],
        summary: str,
        duration_ms: float = 0.0,
    ) -> None:
        self.steps.append(
            CoTStep(
                layer=layer,
                title=title,
                thought_process=thought,
                key_decisions=decisions,
                output_summary=summary,
                duration_ms=round(duration_ms, 3),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "steps": [vars(step) for step in self.steps],
            "final_synthesis": self.final_synthesis,
            "overall_confidence": self.overall_confidence,
            "total_layers": self.total_layers,
            "total_duration_ms": round((time.time() - self.started_at) * 1000, 3),
        }

    def to_markdown(self) -> str:
        md = ["# Chain of Thought Hierarquico - Doninha IA", ""]
        md.append(f"**Prompt:** {self.prompt}")
        md.append("")

        for step in self.steps:
            md.append(f"## {step.layer}: {step.title}")
            md.append(f"**Resumo do raciocinio:** {step.thought_process}")
            if step.key_decisions:
                md.append("**Decisoes chave:**")
                for decision in step.key_decisions:
                    md.append(f"- {decision}")
            md.append(f"**Resultado:** {step.output_summary}")
            if step.duration_ms:
                md.append(f"**Duracao:** {step.duration_ms:.1f} ms")
            md.append("")

        md.append("## Sintese Final")
        md.append(self.final_synthesis)
        return "\n".join(md)


class HierarchicalCoTOrchestrator:
    """Compatibilidade com o fluxo proposto no CoT.txt."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.trace: HierarchicalCoTTrace | None = None

    def process_with_cot(
        self,
        prompt: str,
        use_agent: bool = True,
        return_trace: bool = True,
    ) -> Dict[str, Any]:
        result = self.pipeline.process(prompt, use_agent=use_agent, return_cot=return_trace)
        trace = getattr(result, "cot_trace", None)
        self.trace = trace
        return {
            "response": result.response,
            "truth_value": result.truth_value,
            "state": result.state,
            "certainty": result.certainty,
            "cot_trace": trace.to_dict() if trace and return_trace else None,
            "cot_markdown": trace.to_markdown() if trace and return_trace else None,
        }
