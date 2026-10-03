"""Codex Systems Test Suite: High-Speed Fuzzy Entity Disambiguation & Atomic Persistence.
Authored by Codex (OpenAI Systems Perspective) for TrustedAI.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import time
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from engine import MovieDataEngine
from codex_systems_core import (
    CodexFuzzyEntityResolver,
    CodexAtomicMemoryJournal
)


def run_codex_systems_tests():
    print("=" * 70)
    print("CODEX'S SYSTEMS TEST SUITE: FUZZY ENTITY & PERSISTENCE BENCHMARK")
    print("=" * 70)

    # 1. Initialize Engine & Resolver
    print("\n[Step 1] Initializing MovieDataEngine & Codex Resolver...")
    engine = MovieDataEngine()
    resolver = CodexFuzzyEntityResolver(movies_df=engine.movies)
    print(f"-> Indexed {len(resolver._title_lookup)} clean title variations.")

    # ---------------------------------------------------------------------
    # Test 1: Bilingual Vietnamese Cinema Aliases
    # ---------------------------------------------------------------------
    print("\n[Test 1] Testing Bilingual Vietnamese Aliases...")
    test_aliases = [
        ("đảo kinh hoàng", "Shutter Island"),
        ("dao kinh hoang", "Shutter Island"),
        ("bố già", "Godfather"),
        ("võ sĩ giác đấu", "Gladiator"),
        ("kẻ cắp giấc mơ", "Inception"),
    ]
    for alias_in, expected_sub in test_aliases:
        mid, title, conf = resolver.resolve_title(alias_in)
        print(f"Query: '{alias_in}' -> Resolved: '{title}' (ID: {mid}, Conf: {conf})")
        assert mid is not None, f"Failed to resolve alias: {alias_in}"
        assert expected_sub.lower() in title.lower(), f"Expected '{expected_sub}' in '{title}'"
    print("-> STATUS: [PASS - Bilingual alias resolution verified]")

    # ---------------------------------------------------------------------
    # Test 2: Misspelled & Typo Robustness (Fuzzy Matching)
    # ---------------------------------------------------------------------
    print("\n[Test 2] Testing Misspelled Typo Robustness...")
    typos = [
        ("Inceptoin", "Inception"),
        ("Shuter Island", "Shutter Island"),
        ("Pulp Ficiton", "Pulp Fiction"),
        ("Figth Club", "Fight Club"),
    ]
    for typo_in, expected_sub in typos:
        mid, title, conf = resolver.resolve_title(typo_in)
        print(f"Typo: '{typo_in}' -> Resolved: '{title}' (ID: {mid}, Conf: {conf})")
        assert mid is not None, f"Failed to resolve typo: {typo_in}"
        assert expected_sub.lower() in title.lower()
        assert conf >= 0.75, f"Expected confidence >= 0.75, got {conf}"
    print("-> STATUS: [PASS - Fuzzy typo matching verified]")

    # ---------------------------------------------------------------------
    # Test 3: False-Positive Immunity (Must NOT match generic words)
    # ---------------------------------------------------------------------
    print("\n[Test 3] Testing False-Positive Immunity...")
    non_movies = ["phim kinh dị", "hôm nay xem gì", "chào bạn", "tôi muốn", "tâm lý đen tối"]
    for nm in non_movies:
        mid, title, conf = resolver.resolve_title(nm)
        print(f"Non-movie: '{nm}' -> Resolved: {title} (Conf: {conf})")
        assert mid is None, f"False positive! '{nm}' incorrectly matched to '{title}' (Conf: {conf})"
    print("-> STATUS: [PASS - Immunity to false positives verified]")

    # ---------------------------------------------------------------------
    # Test 4: Atomic Memory Journal Integrity
    # ---------------------------------------------------------------------
    print("\n[Test 4] Testing Atomic Journal Persistence...")
    test_journal = root_dir / "data" / "codex_test_journal.json"
    dummy_payload = {"user_id": 999, "reviews": [{"movie": "Inception", "stars": 5}]}
    
    ok = CodexAtomicMemoryJournal.atomic_save(test_journal, dummy_payload)
    assert ok and test_journal.exists(), "Atomic save failed"
    loaded = CodexAtomicMemoryJournal.safe_load(test_journal)
    assert loaded.get("user_id") == 999
    # Clean up test artifact
    test_journal.unlink(missing_ok=True)
    print("-> STATUS: [PASS - Atomic journal save & load verified]")

    # ---------------------------------------------------------------------
    # Test 5: Microsecond Throughput & Latency Benchmark
    # ---------------------------------------------------------------------
    print("\n[Test 5] Running 500-Query Latency Benchmark...")
    queries = ["Inception", "Shutter Island", "Pulp Fiction", "The Godfather", "Toy Story"]
    t0 = time.perf_counter()
    for _ in range(100):
        for q in queries:
            resolver.resolve_title(q)
    elapsed = time.perf_counter() - t0
    avg_us = (elapsed / 500) * 1_000_000
    print(f"-> 500 Queries completed in {elapsed*1000:.2f} ms")
    print(f"-> Average latency per query: {avg_us:.2f} microseconds (µs)")
    assert avg_us < 200, f"Latency too high: {avg_us} µs"
    print("-> STATUS: [PASS - Sub-millisecond performance target achieved]")

    print("\n" + "=" * 70)
    print("CODEX'S SYSTEMS TEST SUITE PASSED (100%)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_codex_systems_tests()
    exit(0 if success else 1)
