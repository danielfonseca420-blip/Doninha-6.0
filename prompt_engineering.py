"""
Prompt engineering com raciocinio auditavel por camada.

As instrucoes abaixo pedem resumos estruturados e decisoes verificaveis,
sem depender de expor raciocinio interno detalhado do modelo.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from layer_titles import LAYER_TITLES


COT_INSTRUCTION = """
Construa um resumo de raciocinio por etapas, com foco em evidencias,
incertezas, decisoes tomadas e limites da conclusao.
"""


def get_layer_prompt(layer: str, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Retorna um prompt otimizado para a camada informada."""
    context = context or {}
    layer_key = layer.lower()

    prompts = {
        "l1": f"""
Voce esta na Camada L1: {LAYER_TITLES.get('l1', 'Demarcacao de Conceitos Fundamentais')}

{COT_INSTRUCTION}

Tarefa:
1. Extraia os conceitos principais do prompt do usuario.
2. Identifique relacoes semanticas relevantes: sinonimos, antonimos, hiperonimos e hiponimos.
3. Determine o dominio principal.
4. Liste ambiguidades ou conceitos que pedem desambiguacao.

Prompt do usuario: {prompt}

Responda no formato:
CONCEITOS: [lista]
DOMINIO: [dominio]
RELACOES: [relacoes importantes]
AMBIGUIDADES: [itens incertos]
""",
        "l2": f"""
Voce esta na Camada L2: {LAYER_TITLES.get('l2', 'Premissas e proposicoes centrais')}

{COT_INSTRUCTION}

Tarefa:
1. Transforme o prompt em proposicoes epistemologicas claras.
2. Classifique cada proposicao quanto a quantidade, qualidade, relacao e modalidade.
3. Atribua prioridade epistemologica de 0.0 a 1.0.
4. Identifique pressupostos implicitos.

Prompt: {prompt}
Conceitos L1: {context.get('concepts_summary', 'N/A')}

Responda com lista de juizos priorizados.
""",
        "l3": f"""
Voce esta na Camada L3: {LAYER_TITLES.get('l3', 'Analise da Estrutura Logico-filosofica')}

{COT_INSTRUCTION}

Analise as proposicoes usando logica paraconsistente:

Proposicoes: {context.get('propositions', prompt)}

Para cada uma:
- estime grau de crenca (mu) e descrenca (lambda);
- determine o estado logico;
- indique contradicoes locais;
- informe pesos do ensemble quando existirem.
""",
        "l4": f"""
Voce esta na Camada L4: {LAYER_TITLES.get('l4', 'Comparacao da equivalencia entre Estrutura formal e Mundo Empirico')}

{COT_INSTRUCTION}

Tarefa Russell + CoVe:
1. Liste as hipoteses principais das camadas anteriores.
2. Avalie correspondencia com fatos conhecidos da KB e base russelliana.
3. Gere perguntas independentes de verificacao.
4. Revise a sintese eliminando afirmacoes fracas.

Prompt original: {prompt}
Hipoteses L3: {context.get('l3_summary', 'N/A')}
KB relevante: {context.get('kb_summary', 'N/A')}

Termine com [SINTESE L4].
""",
        "l5": f"""
Voce esta na Camada L5: {LAYER_TITLES.get('l5', 'Sintese Intermediaria derivada das etapas anteriores')}

{COT_INSTRUCTION}

Com base no raciocinio anterior, gere uma resposta clara e natural.

Contexto completo L1-L4:
{context.get('full_context', '')}

Responda de forma fluida, preservando rigor epistemologico e incertezas relevantes.
""",
        "l6": f"""
Voce esta na Camada L6: {LAYER_TITLES.get('l6', 'Conclusao do raciocinio')}

{COT_INSTRUCTION}

Refine a resposta anterior para maxima clareza, coerencia e precisao.

Resposta preliminar: {context.get('l5_text', prompt)}
Contexto acumulado: {context.get('full_context', '')}

Melhore fluxo, remova redundancias e fortalece justificativas verificaveis.
""",
        "l7": f"""
Voce esta na Camada L7: {LAYER_TITLES.get('l7', 'Sintese Final e Redacao')}

{COT_INSTRUCTION}

Escreva o TEXTO FINAL DEFINITIVO.

Instrucoes:
- comece direto com a resposta principal;
- use linguagem clara e profissional;
- inclua nivel de confianca quando relevante;
- seja transparente sobre incertezas;
- mantenha tom consistente com o perfil da audiencia: {context.get('audience', 'tecnico')}.

Raciocinio auditavel anterior:
{context.get('full_cot', '')}

Escreva apenas o texto final, sem marcacoes.
""",
    }

    return prompts.get(layer_key, prompt)
