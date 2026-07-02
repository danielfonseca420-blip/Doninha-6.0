from l4_synthesis import SynthesisResult
from l7_final_text import FinalTextEngine


def test_build_l7_prompt_uses_cautious_tone_for_low_confidence():
    engine = FinalTextEngine()
    result = SynthesisResult(
        response="Resposta preliminar.",
        truth_value=0.18,
        certainty=-0.1,
        contradiction=0.82,
        state="Indeterminado",
    )

    prompt = engine._build_l7_prompt(
        prompt="Há evidência suficiente para afirmar isso?",
        l1_summary="conceitos básicos",
        l2_summary="julgamentos preliminares",
        l3_summary="avaliação incerta",
        l4_response=result.response,
        l5_text="texto provisório",
        l6_text="texto refinado",
        audience_profile="técnico",
        synthesis_result=result,
    )

    text = prompt.lower()
    assert "mais cauteloso" in text and "tom:" in text
    assert "confiante e acessível" not in text
