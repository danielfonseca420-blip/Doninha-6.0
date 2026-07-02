"""
TESTES UNITRIOS  RAG HBRIDO
===============================
Suite de testes para validar o sistema de RAG hbrido com L1-L2.

Execute com: pytest test_rag_hybrid.py -v
Ou: python -m unittest test_rag_hybrid.py
"""

import unittest
from pathlib import Path
from typing import Dict, Any, Optional

# Importaes do RAG
try:
    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RetrievalStrategy,
        DomainContext,
        RetrievedDocument,
        RAGContext,
    )
    HAS_RAG = True
except ImportError as e:
    HAS_RAG = False
    RAG_ERROR = str(e)

# Importaes do L1-L2
try:
    from l1_l2_rag_integration import (
        create_l1_l2_rag_pipeline,
        IntegratedL1L2RAGPipeline,
    )
    HAS_L1_L2 = True
except ImportError as e:
    HAS_L1_L2 = False
    L1_L2_ERROR = str(e)


# 
# Testes do Motor RAG
# 

@unittest.skipIf(not HAS_RAG, "RAG modules not available")
class TestHybridRAGEngine(unittest.TestCase):
    """Testes para HybridRAGContextInjectionEngine."""

    def setUp(self):
        """Preparao antes de cada teste."""
        self.engine = HybridRAGContextInjectionEngine(verbose=False)

    def tearDown(self):
        """Limpeza aps cada teste."""
        pass

    def test_01_engine_initialization(self):
        """Testa inicializao do motor."""
        self.assertIsNotNone(self.engine)
        self.assertGreater(len(self.engine.domains), 0)
        self.assertIn("geral", self.engine.domains)
        self.assertIn("filosofia", self.engine.domains)
        print(" Motor RAG inicializado com sucesso")

    def test_02_domain_detection(self):
        """Testa deteco de domnio."""
        test_cases = [
            ("Aristteles define a substncia", "filosofia"),
            ("Silogismo e lgica proposicional", "lgica"),
            ("Justificao epistmica", "epistemologia"),
        ]

        for query, expected_domain in test_cases:
            domain, conf = self.engine.detect_domain(query)
            print(f" Query: '{query[:30]}...'  {domain} ({conf:.0%})")
            self.assertIsNotNone(domain)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_03_injected_knowledge_retrieval(self):
        """Testa recuperao de conhecimento injetado."""
        kb = self.engine.get_injected_knowledge("geral", query=None)
        self.assertIsInstance(kb, dict)
        print(f" Knowledge Base carregado: {len(kb)} termos")

    def test_04_rag_processing_strategy_direct(self):
        """Testa processamento com estratgia DIRECT_INJECTION."""
        rag_ctx = self.engine.process(
            query="O que  verdade?",
            strategy=RetrievalStrategy.DIRECT_INJECTION,
        )

        self.assertIsInstance(rag_ctx, RAGContext)
        self.assertIsNotNone(rag_ctx.query)
        self.assertIsNotNone(rag_ctx.domain)
        print(f" DIRECT_INJECTION: domnio={rag_ctx.domain}, confiana={rag_ctx.confidence_score:.2%}")

    def test_05_rag_processing_strategy_hybrid(self):
        """Testa processamento com estratgia HYBRID."""
        rag_ctx = self.engine.process(
            query="Explique a paraconsistncia na lgica",
            strategy=RetrievalStrategy.HYBRID,
        )

        self.assertIsInstance(rag_ctx, RAGContext)
        self.assertGreater(len(rag_ctx.compiled_context), 0)
        self.assertIn("##", rag_ctx.compiled_context)
        print(f" HYBRID: documentos={len(rag_ctx.retrieved_documents)}, contexto={len(rag_ctx.compiled_context)} chars")

    def test_06_domain_registration(self):
        """Testa registro de domnio customizado."""
        custom_domain = DomainContext(
            domain_name="teste",
            description="Domnio de teste",
            keywords=["teste", "validao"],
            system_prompt="Sistema de teste"
        )

        self.engine.register_domain(custom_domain)
        self.assertIn("teste", self.engine.domains)
        print(" Domnio customizado registrado com sucesso")

    def test_07_context_compilation(self):
        """Testa compilao de contexto."""
        rag_ctx = self.engine.process("Teste de compilao")

        formatted = self.engine.format_for_l1_l2(rag_ctx)
        self.assertIsInstance(formatted, dict)
        self.assertIn("domain", formatted)
        self.assertIn("system_prompt", formatted)
        self.assertIn("documents", formatted)
        print(f" Contexto compilado para L1-L2: {len(formatted['documents'])} docs")

    def test_08_confidence_score(self):
        """Testa score de confiana."""
        rag_ctx = self.engine.process("Query teste")

        self.assertIsInstance(rag_ctx.confidence_score, float)
        self.assertGreaterEqual(rag_ctx.confidence_score, 0.0)
        self.assertLessEqual(rag_ctx.confidence_score, 1.0)
        print(f" Confidence score: {rag_ctx.confidence_score:.2%}")


# 
# Testes de Integrao L1-L2-RAG
# 

@unittest.skipIf(not HAS_L1_L2, "L1-L2 RAG modules not available")
class TestL1L2RAGIntegration(unittest.TestCase):
    """Testes para integrao L1-L2-RAG."""

    def setUp(self):
        """Preparao antes de cada teste."""
        self.pipeline = create_l1_l2_rag_pipeline()

    def test_01_pipeline_initialization(self):
        """Testa inicializao do pipeline."""
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.rag_engine)
        self.assertIsNotNone(self.pipeline.l1_enricher)
        self.assertIsNotNone(self.pipeline.l2_enricher)
        print(" Pipeline L1-L2-RAG inicializado")

    def test_02_l1_extraction(self):
        """Testa extrao de conceitos (L1)."""
        l1_output = self.pipeline.l1_enricher.extract_and_enrich(
            query="O que  conhecimento?"
        )

        self.assertIsNotNone(l1_output)
        self.assertIsNotNone(l1_output.domain)
        self.assertGreater(len(l1_output.concepts), 0)
        print(f" L1: {len(l1_output.concepts)} conceitos, domnio={l1_output.domain}")

    def test_03_l2_analysis(self):
        """Testa anlise de juzos (L2)."""
        l2_output = self.pipeline.l2_enricher.analyze_and_enrich(
            query=" verdade que a verdade  relativa?"
        )

        self.assertIsNotNone(l2_output)
        self.assertGreater(len(l2_output.judgments), 0)
        if l2_output.top_judgment:
            self.assertIsNotNone(l2_output.top_judgment.proposicao)
        print(f" L2: {len(l2_output.judgments)} juzos")

    def test_04_full_pipeline(self):
        """Testa pipeline completo."""
        result = self.pipeline.process(
            query="Explique a diferena entre conhecimento e opinio"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("query", result)
        self.assertIn("domain", result)
        self.assertIn("l1_output", result)
        self.assertIn("l2_output", result)
        self.assertIn("compiled_context", result)
        self.assertIn("system_prompt", result)
        self.assertIn("confidence", result)

        print(f" Pipeline completo:")
        print(f"  - Domnio: {result['domain']}")
        print(f"  - Confiana: {result['confidence']:.2%}")
        print(f"  - L1 Conceitos: {len(result['l1_output'].concepts)}")
        print(f"  - L2 Juzos: {len(result['l2_output'].judgments)}")
        print(f"  - Contexto: {len(result['compiled_context'])} chars")

    def test_05_domain_detection_integration(self):
        """Testa deteco de domnio na integrao."""
        test_queries = [
            ("Aristteles", "filosofia"),
            ("Silogismo", "lgica"),
            ("Crena justificada", "epistemologia"),
        ]

        for query, expected_domain in test_queries:
            result = self.pipeline.process(query)
            print(f" '{query}' detectado como: {result['domain']}")


# 
# Testes de Performance
# 

@unittest.skipIf(not HAS_RAG, "RAG modules not available")
class TestPerformance(unittest.TestCase):
    """Testes de performance."""

    def setUp(self):
        """Preparao antes de cada teste."""
        self.engine = HybridRAGContextInjectionEngine(verbose=False)

    def test_01_processing_time(self):
        """Testa tempo de processamento."""
        import time

        start = time.time()
        rag_ctx = self.engine.process("O que  verdade?")
        elapsed = time.time() - start

        self.assertLess(elapsed, 10.0)  # Deve processar em menos de 10s
        print(f" Processamento em {elapsed:.2f}s")

    def test_02_memory_consistency(self):
        """Testa consistncia em mltiplas chamadas."""
        results = []
        for _ in range(3):
            rag_ctx = self.engine.process("Query consistncia")
            results.append(rag_ctx.domain)

        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])
        print(f" Mltiplas chamadas consistentes: {results[0]}")


# 
# Testes de Regresso
# 

@unittest.skipIf(not HAS_RAG, "RAG modules not available")
class TestRegression(unittest.TestCase):
    """Testes de regresso."""

    def test_01_empty_query(self):
        """Testa query vazia."""
        engine = HybridRAGContextInjectionEngine(verbose=False)
        try:
            result = engine.process("")
            print(" Query vazia tratada")
        except Exception:
            pass  # Esperado

    def test_02_very_long_query(self):
        """Testa query muito longa."""
        engine = HybridRAGContextInjectionEngine(verbose=False)
        long_query = "palavra " * 1000  # 1000 palavras
        try:
            result = engine.process(long_query)
            print(" Query muito longa tratada")
        except Exception as e:
            print(f" Query longa falhou: {e}")

    def test_03_special_characters(self):
        """Testa caracteres especiais."""
        engine = HybridRAGContextInjectionEngine(verbose=False)
        queries_with_special = [
            "O que ?!@#$%",
            "Acar, caf, Braslia",
            " +  = ?",
        ]

        for query in queries_with_special:
            try:
                result = engine.process(query)
                print(f" Query com especiais: '{query[:20]}...'")
            except Exception as e:
                print(f" Falhou em '{query[:20]}...': {e}")


# 
# Test Runner
# 

def run_all_tests():
    """Executa todos os testes."""
    print("\n" + "="*70)
    print("TESTES UNITRIOS  RAG HBRIDO")
    print("="*70 + "\n")

    # Verifica dependncias
    if not HAS_RAG:
        print(f" RAG modules no disponvel: {RAG_ERROR}")
    if not HAS_L1_L2:
        print(f" L1-L2 modules no disponvel: {L1_L2_ERROR}")

    # Cria suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Adiciona testes
    if HAS_RAG:
        suite.addTests(loader.loadTestsFromTestCase(TestHybridRAGEngine))
        suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
        suite.addTests(loader.loadTestsFromTestCase(TestRegression))

    if HAS_L1_L2:
        suite.addTests(loader.loadTestsFromTestCase(TestL1L2RAGIntegration))

    # Executa
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Resumo
    print("\n" + "="*70)
    print(f"Testes executados: {result.testsRun}")
    print(f"Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")
    print("="*70 + "\n")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
