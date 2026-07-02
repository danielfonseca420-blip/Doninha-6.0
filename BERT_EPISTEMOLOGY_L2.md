#!/usr/bin/env python3
"""
L2 BERT Epistemological Classification - Integration Documentation
===================================================================

## Implementação

A camada L2 (Juízos Kantianos) foi enriquecida com classificação epistemológica
baseada em BERT utilizando o esquema de três valores (T, I, F):

  T (Truth)       → Grau de verdade [0,1]
  I (Indeterminacy) → Grau de indeterminação [0,1]  
  F (Falsity)     → Grau de falsidade [0,1]

Nota: NÃO há restrição T+I+F=1, permitindo capturar:
  • Paraconsistência: T+F > 1 (contraditório)
  • Incompletude: T+I+F < 1 (informação incompleta)
  • Vagueza: I > 0.6 (conceito vago)

## Arquivos Modificados

1. **l2_kantian_judgments.py**
   - Adicionado: classe EpistemicClassification
   - Adicionado: classe BERTAssertionClassifier
   - Adicionado: campo epistemic_classification em KantianJudgment
   - Modificado: KantianJudgmentEngine.refine() para aplicar classificação
   
   Integração automática de classificação BERT em todos os juízos assertóricos
   (Singular/Afirmativo, Singular/Negativo, etc.)

2. **pipeline.py**
   - Atualizado: logging na ETAPA 6 menciona classificação epistemológica L2
   - Clarificado: flow L2→L3 mostra integração de classificações

## Uso do BERT

A classe BERTAssertionClassifier:

  • Tenta usar transformers.pipeline('zero-shot-classification')
  • Fallback heurístico se transformers não estiver disponível
  • Candidatos: ['verdadeiro', 'indeterminado', 'falso']
  • Scores automaticamente mapeados para (T, I, F)

## Classificações Automáticas

Com base em (T, I, F), o sistema gera 4 tipos:

  1. assertiva_confiante    → T > 0.7, I < 0.2, F < 0.2
  2. paraconsistência        → T + F > 1 (contradição local)
  3. incompletude            → T + I + F < 1 (falta informação)
  4. vagueza                 → I > 0.6 (conceito vago)
  5. indeterminado (default) → outros casos

## Exemplo de Saída

```
Proposição: "Este(a) água específico é quente"
T=0.80 I=0.10 F=0.10 [assertiva_confiante]

Proposição: "Este(a) contraditório não possui a propriedade oposta"
T=0.60 I=0.30 F=0.70 [paraconsistência]
```

## Validação

✓ Compilação: l2_kantian_judgments.py, pipeline.py
✓ Integração: Automática em todos os juízos assertóricos
✓ Teste: test_epistemic_classification.py executado com sucesso
✓ Fallback: Funciona com/sem transformers instalado

## Próximos Passos Opcionais

1. Usar classificação (T,I,F) para refinar contraposição em L3
2. Integrar scores BERT com graus μ/λ paraconsistentes
3. Treinar BERT customizado com dataset epistemológico português
4. Validação: coletar feedback de usuário sobre qualidade de classificação
"""

if __name__ == "__main__":
    print(__doc__)
