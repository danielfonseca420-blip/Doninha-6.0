#!/usr/bin/env python3
"""
Teste da classificao epistemolgica BERT em L2 com (T, I, F).

Demonstra como o juzo assertrico  classificado segundo:
  T + F > 1  paraconsistncia
  T + I + F < 1  incompletude
  I high  vagueza
  T high, I low, F low  assertiva confiante
"""

from l1_concept_table import ConceptTable
from l2_kantian_judgments import KantianJudgmentEngine, BERTAssertionClassifier


def main():
    print("=" * 70)
    print("TESTE: Classificao Epistemolgica em L2 (T, I, F)")
    print("=" * 70)

    # Inicializa as camadas L1 e L2
    concept_table = ConceptTable()
    kant_engine = KantianJudgmentEngine(concept_table)

    # Prompts de teste
    test_prompts = [
        "gua quente verdadeira",
        "pode ser falso e verdadeiro ao mesmo tempo",
        "indeterminado e indefinido",
        "sempre verdadeiro",
        "contraditrio e incompleteto",
    ]

    for prompt in test_prompts:
        print(f"\n Prompt: '{prompt}'")
        print("-" * 70)

        # L1: Extrao de conceitos
        concepts = concept_table.extract_concepts(prompt, llm_context=prompt)
        print(f"  L1 Conceitos extrados: {len(concepts)}")
        for concept in concepts[:3]:
            print(f"     {concept.term} [{concept.domain}]")
            if concept.application_context:
                print(f"      Contexto: {concept.application_context[:60]}...")

        # L2: Juzos kantianos com classificao epistemolgica
        judgments = kant_engine.refine(prompt, concepts)
        print(f"\n  L2 Juzos assertricos com (T, I, F):")

        # Filtra apenas juzos assertricos para mostrar classificao
        assertoric_judgments = [j for j in judgments if j.modalidade == "Assertrico"]
        for i, judgment in enumerate(assertoric_judgments[:5], 1):
            ec = judgment.epistemic_classification
            print(f"\n    {i}. [{judgment.quantidade}/{judgment.qualidade}]")
            print(f"       Proposio: {judgment.proposicao[:55]}...")
            print(f"       {ec}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
