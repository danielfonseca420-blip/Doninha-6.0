import unittest

from l3_paraconsistent import ParaconsistentValue
from l4_synthesis import RussellianSynthesisEngine


class CitationBehaviorTests(unittest.TestCase):
    def test_generate_response_does_not_claim_unverified_bibliography(self):
        engine = RussellianSynthesisEngine(
            knowledge_base={"teste": 0.9},
            russell_concept_base=object(),
        )
        pv = ParaconsistentValue(
            proposition="A proposição principal é suportada por evidência local.",
            mu=0.8,
            lam=0.1,
        )

        response = engine._generate_response(pv, "Pergunta de teste", kb={"teste": 0.9})

        self.assertNotIn("Bertrand Russell", response)
        self.assertNotIn("The Problems of Philosophy", response)
        self.assertIn("confirmada para esta resposta", response.lower())
        self.assertIn("citao", response.lower())


if __name__ == "__main__":
    unittest.main()
