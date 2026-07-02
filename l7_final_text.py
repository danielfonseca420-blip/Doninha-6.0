"""
CAMADA L7  Texto Final Definitivo (Automtico e Robusto)
==========================================================
Gera o texto final de alta qualidade a partir do raciocnio acumulado
nas camadas L1 a L6.

A camada L7 funciona como um prompt adicional de escrita: ela recebe os
sumrios das camadas anteriores e transforma o contedo em um nico
bloco contnuo, fluido e persuasivo.

Suporta mltiplos providers:
- ollama: para modelos rodando localmente
- custom_lm: para modelos customizados
- template: fallback que retorna texto sem LLM
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import logging
from llm_provider_client import generate_text as generate_provider_text
from l4_synthesis import SynthesisResult
from layer_titles import LAYER_TITLES

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore

logger = logging.getLogger(__name__)

try:
    from l5_generation import generate_with_custom_lm
except Exception:
    generate_with_custom_lm = None  # type: ignore

try:
    import ollama
except Exception:
    ollama = None  # type: ignore


class FinalTextEngine:
    """Gera o texto final definitivo a partir do raciocnio L1L6."""

    # Classificao de audincia
    AUDIENCE_PROFILES = {
        "leigo": {
            "description": "Pblico geral sem conhecimento tcnico especializado",
            "style": "Linguagem simples e acessvel, analogias concretas do dia a dia, evitar notao formal, foco em aplicaes prticas e concluses teis",
            "examples": ["o que ", "como funciona", "explicar", "simples"]
        },
        "tcnico": {
            "description": "Profissional da rea com conhecimento tcnico intermedirio",
            "style": "Usar terminologia especfica da rea, incluir referncias conceituais, evitar tabelas de estados complexas, manter rigor tcnico sem excesso de formalismo",
            "examples": ["anlise", "implementao", "mtodo", "tcnica", "profissional"]
        },
        "acadmico": {
            "description": "Pesquisador ou acadmico com formao avanada",
            "style": "Notao completa e formal, referncias bibliogrficas detalhadas, incluir modo debug/disponibilidade de estados internos, rigor acadmico completo",
            "examples": ["teoria", "formal", "demonstrao", "referncia", "acadmico", "pesquisa"]
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa a FinalTextEngine com configuraes opcionais.
        
        Args:
            config: Dicionrio com configuraes de L7, incluindo:
                - provider: 'ollama', 'custom_lm', ou 'template'
                - model: Nome do modelo (para ollama)
                - temperature: Temperatura de gerao (padro: 0.7)
                - max_tokens: Nmero mximo de tokens (padro: 4096)
                - custom_lm_path: Caminho do modelo customizado (para custom_lm)
        """
        self.config = config or {}
        self.l7_config = self.config.get("l7", {})
        
    def _build_l7_prompt(self, 
                        prompt: str,
                        l1_summary: str,
                        l2_summary: str,
                        l3_summary: str,
                        l4_response: str,
                        l5_text: str,
                        l6_text: str,
                        audience_profile: str = "tcnico",
                        full_synthesis: Optional[str] = None,
                        synthesis_result: Optional[SynthesisResult] = None,
                        cot_context: str = "") -> str:
        """
        Constri automaticamente o prompt L7 para gerao de texto final.
        
        Este mtodo agrega todo o raciocnio das camadas L1-L6 e produz
        um prompt bem estruturado e adaptado ao perfil de audincia.
        
        Args:
            prompt: Pergunta/prompt original do usurio
            l1_summary: Resumo de conceitos extrados (L1)
            l2_summary: Resumo de juzos kantianos (L2)
            l3_summary: Resumo de anlise paraconsistente (L3)
            l4_response: Resposta da sntese russelliana (L4)
            l5_text: Texto gerado em L5 (se disponvel)
            l6_text: Texto refinado de L6
            audience_profile: Perfil de audincia ('leigo', 'tcnico', 'acadmico')
            full_synthesis: Sntese completa opcional
            
        Returns:
            String com prompt bem estruturado para gerao automtica
        """
        lines = []
        
        # SEO 1: Instruo Base
        lines.append("Voc  um excelente escritor tcnico e comunicador, especializado em sintetizar raciocnios complexos em textos claros, profundos e agradveis de ler.")
        lines.append("")
        lines.append("Sua funo  gerar o TEXTO FINAL DEFINITIVO a partir de todo o raciocnio desenvolvido nas camadas L1 a L6.")
        lines.append("")
        
        # SEO 2: Contexto do Prompt Original
        lines.append("" * 70)
        lines.append("PROMPT ORIGINAL DO USURIO:")
        lines.append("" * 70)
        lines.append(prompt)
        lines.append("")
        
        # SEO 3: Resumo das Camadas
        lines.append("" * 70)
        lines.append("RACIOCNIO ACUMULADO (CAMADAS L1L6):")
        lines.append("" * 70)
        lines.append(f"L1 - Conceitos Extrados: {l1_summary or 'No disponvel'}")
        lines.append(f"L2 - Juzos Kantianos: {l2_summary or 'No disponvel'}")
        lines.append(f"L3 - Anlise Paraconsistente: {l3_summary or 'No disponvel'}")
        lines.append(f"L4 - Sntese Russelliana: {l4_response or 'No disponvel'}")
        lines.append(f"L5 - Gerao de Resposta: {l5_text or 'No disponvel'}")
        lines.append(f"L6 - Refinamento Final: {l6_text or 'No disponvel'}")
        lines.append("")
        
        # SEO 4: Perfil de Audincia
        profile_data = self.AUDIENCE_PROFILES.get(audience_profile, self.AUDIENCE_PROFILES["tcnico"])
        lines.append("" * 70)
        lines.append(f"PERFIL DE AUDINCIA: {audience_profile.upper()}")
        lines.append("" * 70)
        lines.append(f"Descrio: {profile_data['description']}")
        lines.append(f"Estilo recomendado: {profile_data['style']}")
        lines.append("")
        
        # SEO 5: Indicadores epistmicos de tom
        should_be_cautious = False
        if synthesis_result is not None:
            truth_value = float(getattr(synthesis_result, "truth_value", 0.0) or 0.0)
            state = getattr(synthesis_result, "state", "")
            confidence_label = getattr(synthesis_result, "confidence_label", "")
            should_be_cautious = (
                state in {"Indeterminado", "N", "Baixa Confiana"}
                or confidence_label in {"Baixa Confiana", "Indeterminado"}
                or truth_value < 0.45
                or float(getattr(synthesis_result, "contradiction", 0.0) or 0.0) > 0.25
            )

        if should_be_cautious:
            tone_instruction = (
                "Tom: mais cauteloso e proporcional ao grau de incerteza; evite afirmaes categricas, destaque limites, lacunas e hipteses provisrias, e use linguagem explicitamente prudente quando a evidncia for fraca."
            )
        else:
            tone_instruction = (
                "Tom: profissional, claro e persuasivo, sem exagerar a certeza, e mantendo a abertura para nuances quando elas forem relevantes."
            )

        # SEO 6: Diretivas de Formatao (OBRIGATRIAS)
        lines.append("" * 70)
        lines.append("DIRETIVAS DE FORMATAO (OBRIGATRIAS):")
        lines.append("" * 70)
        lines.append(" Formato: Texto fluido, com pargrafos quando necessrio, sem ttulos, subttulos, bullets ou numerao.")
        lines.append(" Abertura: Comece diretamente com a tese ou resposta principal (1-2 frases fortes e claras).")
        lines.append(" Estrutura: Desenvolvimento gradual das premissas, nuances e evoluo do pensamento.")
        lines.append(" Integrao: Harmonize todas as camadas de forma natural, mostrando a evoluo do raciocnio.")
        lines.append(f" {tone_instruction}")
        lines.append(" Variao: Use frases de tamanhos variados com transies naturais e sofisticadas.")
        lines.append(" nfase: Destaque ideias importantes via posicionamento e repetio sutil (no bvia).")
        lines.append(" Rigor: Inclua tenses, trade-offs e incertezas com elegncia e maturidade intelectual.")
        lines.append("")
        
        # SEO 7: Tarefa Final
        lines.append("" * 70)
        lines.append("TAREFA:")
        lines.append("" * 70)
        lines.append("Transforme todo esse raciocnio em uma DISSERTAO EXPOSITIVA FLUIDA, COESA E NATURAL.")
        lines.append("Escreva o texto final agora:")
        lines.append("" * 70)
        lines.append("")
        base_prompt = "\n".join(lines)
        if get_layer_prompt is None:
            return base_prompt
        full_cot = cot_context or base_prompt
        return get_layer_prompt("l7", prompt, {"full_cot": full_cot, "audience": audience_profile})

    @classmethod
    def _classify_audience(cls, prompt: str, l1_summary: str, l2_summary: str, l3_summary: str) -> str:
        """
        Classifica o perfil da audincia baseado no prompt e contexto das camadas.
        Retorna: 'leigo', 'tcnico', ou 'acadmico'
        """
        # Combinar todo o contexto para anlise
        full_context = f"{prompt} {l1_summary} {l2_summary} {l3_summary}".lower()

        # Contar termos tcnicos e indicadores de nvel
        technical_indicators = {
            "leigo": 0,
            "tcnico": 0,
            "acadmico": 0
        }

        # Anlise de vocabulrio e termos
        for profile, data in cls.AUDIENCE_PROFILES.items():
            for keyword in data["examples"]:
                if keyword.lower() in full_context:
                    technical_indicators[profile] += 1

        # Anlise de extenso e complexidade
        prompt_length = len(prompt.split())
        has_formal_terms = any(term in full_context for term in [
            "formal", "demonstrao", "teorema", "axioma", "paradigma",
            "epistemologia", "ontologia", "metafsica", "transcendental"
        ])
        has_technical_jargon = any(term in full_context for term in [
            "lgica paraconsistente", "juzo kantiano", "sntese russelliana",
            "valor de verdade", "contradio", "cognio"
        ])

        # Regras de classificao
        if has_formal_terms or prompt_length > 50 or "referncia" in full_context:
            return "acadmico"
        elif has_technical_jargon or technical_indicators["tcnico"] > technical_indicators["leigo"]:
            return "tcnico"
        elif technical_indicators["leigo"] > 0 or prompt_length < 20:
            return "leigo"
        else:
            # Padro: analisar padro da pergunta
            question_patterns = {
                "acadmico": ["por que", "como se explica", "qual a teoria", "demonstre"],
                "tcnico": ["como implementar", "qual mtodo", "anlise de", "tcnica para"],
                "leigo": ["o que ", "para que serve", "como funciona", "exemplo"]
            }

            for profile, patterns in question_patterns.items():
                if any(pattern in prompt.lower() for pattern in patterns):
                    return profile

            return "tcnico"  # padro seguro

    def _enhance_with_writer_prompt(
        self,
        base_text: str,
        prompt: str,
        audience_profile: str,
        synthesis_result: Optional[SynthesisResult] = None,
        canonical_alerts: Optional[list] = None
    ) -> str:
        """
        Aprimora o texto base usando o prompt de redao (fallback sem LLM).
        Usada quando nenhum provider est disponvel.
        """
        # Aqui poderamos aplicar algumas transformaes/enhancements
        # que no requerem LLM, como formatting, reorganizao, etc.
        return self._build_writer_prompt(
            prompt=prompt,
            l1_summary="",
            l2_summary="",
            l3_summary="",
            l4_response="",
            l5_text="",
            l6_text=base_text,
            synthesis_result=synthesis_result,
            canonical_alerts=canonical_alerts,
            audience_profile=audience_profile
        )


    def finalize_text(
        self,
        prompt: str,
        l1_summary: str = "",
        l2_summary: str = "",
        l3_summary: str = "",
        l4_response: str = "",
        l5_text: str = "",
        l6_text: str = "",
        synthesis_result: Optional[SynthesisResult] = None,
        provider: str = "ollama",
        model: str = "doninha8:latest",
        custom_lm_path: str = "",
        canonical_alerts: Optional[list] = None,
        audience_profile: Optional[str] = None,
        cot_context: str = "",
        **kwargs) -> str:
        """
        Gera o texto final definitivo de forma automtica e robusta.
        
        Suporta mltiplos providers:
        - ollama: Executa modelos locais via Ollama
        - custom_lm: Usa modelo LM customizado
        - template: Retorna o melhor resultado de L6 sem LLM (fallback)
        
        Args:
            prompt: Pergunta/prompt original do usurio
            l1_summary: Resumo de conceitos (L1)
            l2_summary: Resumo de juzos kantianos (L2)
            l3_summary: Resumo de anlise paraconsistente (L3)
            l4_response: Resposta da sntese (L4)
            l5_text: Texto gerado (L5)
            l6_text: Texto refinado (L6)
            synthesis_result: Resultado da sntese L4
            provider: 'ollama', 'custom_lm', 'template' ou qualquer provider cloud suportado
            model: Nome do modelo (para ollama ou providers remotos)
            custom_lm_path: Caminho do modelo customizado
            canonical_alerts: Alertas de incompatibilidade semntica
            audience_profile: 'leigo', 'tcnico', ou 'acadmico'
            **kwargs: Argumentos adicionais (temperature, max_tokens, etc.)
            
        Returns:
            String com o texto final gerado
        """
        
        # 1. Classificar audincia se no foi fornecida
        if audience_profile is None:
            audience_profile = self._classify_audience(prompt, l1_summary, l2_summary, l3_summary)
        
        # 2. Construir prompt L7 automtico
        l7_prompt = self._build_l7_prompt(
            prompt=prompt,
            l1_summary=l1_summary,
            l2_summary=l2_summary,
            l3_summary=l3_summary,
            l4_response=l4_response,
            l5_text=l5_text,
            l6_text=l6_text,
            audience_profile=audience_profile,
            full_synthesis=synthesis_result.response if synthesis_result else None,
            synthesis_result=synthesis_result,
            cot_context=cot_context,
        )
        
        # 3. Gerar texto usando o provider selecionado
        generated_text = None
        
        if provider == "ollama" and ollama:
            generated_text = self._generate_with_ollama(
                prompt=l7_prompt,
                model=self.l7_config.get("model", model) or self.l7_config.get("ollama_model", model),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096)
            )
        elif provider == "custom_lm" and generate_with_custom_lm:
            custom_path = custom_lm_path or self.l7_config.get("custom_lm_path", "")
            if custom_path:
                generated_text = self._generate_with_custom_lm(
                    prompt=l7_prompt,
                    model_path=custom_path
                )
        elif provider in {"openai", "anthropic", "gemini", "grok", "groq", "meta"}:
            generated_text = self._generate_with_provider(
                provider=provider,
                prompt=l7_prompt,
                model=self.l7_config.get("model", model),
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            )

        if provider == "template":
            return (l6_text or l5_text or l4_response or "").strip()

        if generated_text:
            return generated_text.strip()

        return (l6_text or l5_text or l4_response or "").strip()

    def _generate_with_ollama(self, prompt: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> Optional[str]:
        """
        Gera texto usando Ollama (modelos locais).
        
        Args:
            prompt: Prompt para gerao
            model: Nome do modelo (e.g., 'llama2', 'neural-chat', 'mistral')
            temperature: Controla criatividade (0.0-1.0)
            max_tokens: Limite de tokens de sada
            
        Returns:
            Texto gerado ou None se falhar
        """
        try:
            if not ollama:
                logger.error("Ollama no est instalado")
                return None
            
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                think="high",
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 8192
                }
            )
            
            generated = response.get("message", {}).get("content", "").strip()
            if generated:
                logger.info(f"L7 (ollama/{model}): Texto gerado com sucesso ({len(generated)} chars)")
                return generated
            else:
                logger.warning("Ollama retornou resposta vazia")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao usar Ollama: {e}")
            return None

    def _generate_with_custom_lm(self, prompt: str, model_path: str) -> Optional[str]:
        """
        Gera texto usando modelo LM customizado.
        
        Args:
            prompt: Prompt para gerao
            model_path: Caminho do modelo customizado
            
        Returns:
            Texto gerado ou None se falhar
        """
        try:
            if not generate_with_custom_lm:
                logger.error("generate_with_custom_lm no est disponvel")
                return None
            
            generated = generate_with_custom_lm(prompt, model_path)
            if generated:
                logger.info(f"L7 (custom_lm): Texto gerado com sucesso ({len(generated)} chars)")
                return generated
            else:
                logger.warning("Custom LM retornou resposta vazia")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao usar Custom LM: {e}")
            return None

    def _generate_with_provider(
        self,
        provider: str,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Optional[str]:
        """Gera texto usando um provider cloud suportado."""
        try:
            generated = generate_provider_text(
                provider=provider,
                prompt=prompt,
                model=model or None,
                base_url=self.l7_config.get("base_url") or None,
                api_key=self.l7_config.get("api_key") or None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if generated:
                logger.info(f"L7 ({provider}/{model or 'default'}): Texto gerado com sucesso ({len(generated)} chars)")
                return generated
            logger.warning(f"Provider {provider} retornou resposta vazia")
            return None
        except Exception as e:
            logger.error(f"Erro ao usar provider {provider}: {e}")
            return None

    def _build_writer_prompt(
        self,
        prompt: str,
        l1_summary: str,
        l2_summary: str,
        l3_summary: str,
        l4_response: str,
        l5_text: str,
        l6_text: str,
        synthesis_result: Optional[SynthesisResult] = None,
        canonical_alerts: Optional[list] = None,
        audience_profile: str = "tcnico",
    ) -> str:
        lines = [
            "Voc  um excelente escritor tcnico e comunicador, com capacidade de sintetizar raciocnios complexos em textos claros e persuasivos.",
            "Sua funo  gerar o texto final de alta qualidade a partir do raciocnio desenvolvido nas camadas L1 a L6.",
            "Tarefa: transforme todo o raciocnio acumulado nas camadas L1 a L6 em uma dissertao expositiva fluida, coesa e natural, usando pargrafos claros quando for apropriado.",
            "Pblico-alvo: leitor inteligente de nvel intermedirio (no  especialista no tema).",
            "Formato: texto fluido, com pargrafos quando necessrio, sem ttulos, subttulos, bullets ou qualquer marcao.",
            "Estrutura recomendada: comece diretamente com a tese ou resposta principal em 1-2 frases fortes e claras. Em seguida, desenvolva as premissas, nuances e evolues do pensamento.",
            "Integre harmoniosamente o contedo das camadas anteriores, mostrando a evoluo natural do raciocnio e destacando tenses, trade-offs e incertezas com elegncia.",
            "Linguagem: clara, conversacional e precisa. Use termos tcnicos quando necessrios, explicando-os na sequncia.",
            "Estilo: profissional, acessvel, rigoroso e fcil de ler.",
            "",
            f"PERFIL DA AUDINCIA CLASSIFICADO: {audience_profile.upper()}",
        ]

        # Adicionar instrues especficas do perfil
        profile_data = self.AUDIENCE_PROFILES.get(audience_profile, self.AUDIENCE_PROFILES["tcnico"])
        lines.append(f"Descrio do perfil: {profile_data['description']}")
        lines.append(f"Instrues de estilo especficas: {profile_data['style']}")
        lines.append("")
        lines.append(f"Pergunta do usurio: {prompt}")
        lines.append("")
        lines.append("Raciocnio acumulado L1L6:")
        lines.append(f"L1 - {LAYER_TITLES['l1']}: {l1_summary or 'No disponvel.'}")
        lines.append(f"L2 - {LAYER_TITLES['l2']}: {l2_summary or 'No disponvel.'}")
        lines.append(f"L3 - {LAYER_TITLES['l3']}: {l3_summary or 'No disponvel.'}")
        lines.append(f"L4 - {LAYER_TITLES['l4']}: {l4_response or 'No disponvel.'}")
        lines.append(f"L5 - {LAYER_TITLES['l5']}: {l5_text or 'No disponvel.'}")
        lines.append(f"L6 - {LAYER_TITLES['l6']}: {l6_text or 'No disponvel.'}")
        lines.append(f"L7 - {LAYER_TITLES['l7']}: texto final de sntese e redao.")
        lines.append("")

        # Adicionar informaes da sntese L4 se disponvel
        if synthesis_result:
            lines.append(f"Estado da sntese L4: {synthesis_result.state}")
            lines.append(f"Valor de verdade L4: {synthesis_result.truth_value:.2f}")
            lines.append(f"Certeza L4: {synthesis_result.certainty:+.2f}")
            lines.append(f"Contradio L4: {synthesis_result.contradiction:+.2f}")
            lines.append("")
        else:
            lines.append("Estado da sntese L4: No disponvel.")
            lines.append("")

        # Adicionar alertas de incompatibilidade cannica
        if canonical_alerts:
            lines.append("Alertas de incompatibilidade cannica:")
            for alert in canonical_alerts:
                lines.append(f"- Conceito '{alert['concept']}': {alert['canonical_context']}")
                lines.append(f"  Uso incompatvel detectado: {alert['incompatible_usage']}")
            lines.append("")
            lines.append("IMPORTANTE: Inclua ressalvas no texto final sobre estes usos incompatveis dos conceitos.")
            lines.append("")

        lines.append("Escreva o texto final agora.")

        return "\n".join(lines)

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def _ensure_single_paragraph(self, text: str) -> str:
        return " ".join(text.replace("\n", " ").split()).strip()
