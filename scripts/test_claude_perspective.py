"""Claude's Perspective Test Suite: Nuance Emotion Analysis & Grounding Audit.
Authored by Claude (Anthropic Perspective) for TrustedAI.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from claude_conversational_core import (
    ClaudeNuanceAnalyzer,
    ClaudeNuanceEmotion,
    ClaudeEmpatheticSynthesizer,
    ClaudeGroundingGuard
)


def run_claude_test_suite():
    print("=" * 70)
    print("CLAUDE'S PERSPECTIVE TEST SUITE: NUANCE, EMPATHY & GROUNDING AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Test 1: Fine-Grained Micro-Emotion & Causal Attribution Extraction
    # ---------------------------------------------------------------------
    print("\n[Test 1] Testing Micro-Emotion & Causal Attribution Extraction...")
    sample_1 = "Hôm qua xem Shutter Island, đoạn kết hụt hẫng và xem nhức cả đầu thật sự."
    res_1 = ClaudeNuanceAnalyzer.extract_nuance_and_causality(sample_1)
    print(f"Input: \"{sample_1}\"")
    print(f"-> Primary Emotion: {res_1['primary_emotion']}")
    print(f"-> Causal Anchors: {res_1['causal_anchors']}")
    print(f"-> Polarity: {res_1['polarity']}")

    assert res_1["primary_emotion"] in (ClaudeNuanceEmotion.COGNITIVE_FATIGUE, ClaudeNuanceEmotion.EMOTIONAL_SHOCK)
    assert "plot_twist_ending" in res_1["causal_anchors"]
    assert res_1["polarity"] == "negative"
    print("-> STATUS: [PASS - Correctly extracted cognitive fatigue and plot twist anchor]")

    sample_2 = "Phim The Godfather nhịp phim quá lê thê và buồn ngủ, xem mãi không hết."
    res_2 = ClaudeNuanceAnalyzer.extract_nuance_and_causality(sample_2)
    print(f"\nInput: \"{sample_2}\"")
    print(f"-> Primary Emotion: {res_2['primary_emotion']}")
    print(f"-> Causal Anchors: {res_2['causal_anchors']}")
    assert res_2["primary_emotion"] == ClaudeNuanceEmotion.DISAPPOINTMENT_PACING
    assert "pacing_and_flow" in res_2["causal_anchors"]
    print("-> STATUS: [PASS - Correctly attributed disappointment to pacing]")

    # ---------------------------------------------------------------------
    # Test 2: Empathetic Non-Confrontational Dialogue Synthesis
    # ---------------------------------------------------------------------
    print("\n[Test 2] Testing Empathetic Inquiry Synthesis on Preference Shift...")
    past_review = {
        "title": "Shutter Island",
        "user_feeling": "đoạn kết hụt hẫng và xem nhức cả đầu thật sự",
        "sentiment": "negative",
        "date_str": "03/10/2026"
    }
    user_query = "Cái phim Shutter Island đó ổn nhỉ, bạn thấy sao?"
    synth_reply = ClaudeEmpatheticSynthesizer.synthesize_recollection_response(
        movie_title="Shutter Island",
        user_query=user_query,
        past_review=past_review,
        community_stats={"avg_rating": 4.02, "num_ratings": 67}
    )
    print(f"Synthesized Response:\n{synth_reply}")

    assert "Shutter Island" in synth_reply
    assert "nhức cả đầu" in synth_reply or "hụt hẫng" in synth_reply
    assert "tò mò" in synth_reply or "góc nhìn" in synth_reply
    print("-> STATUS: [PASS - Non-confrontational curious inquiry verified]")

    # ---------------------------------------------------------------------
    # Test 3: Grounding Guardrail Audit (Detecting Hallucination)
    # ---------------------------------------------------------------------
    print("\n[Test 3] Testing Grounding Guardrail on Hallucinated vs Grounded Claims...")
    
    # Sub-case A: Hallucinated recollection when no past review existed
    fake_response = "Lần trước bạn từng chia sẻ là bạn rất mê phim Interstellar!"
    is_safe, reason = ClaudeGroundingGuard.audit_response(fake_response, actual_past_review=None)
    print(f"Audit on Hallucination -> Is Safe: {is_safe} | Reason: {reason}")
    assert not is_safe, "Guardrail must flag hallucinated recollection"
    print("-> Sub-case A: [PASS - Correctly caught emotional hallucination]")

    # Sub-case B: Grounded response matching actual review
    is_safe_b, reason_b = ClaudeGroundingGuard.audit_response(synth_reply, actual_past_review=past_review)
    print(f"Audit on Grounded Reply -> Is Safe: {is_safe_b} | Reason: {reason_b}")
    assert is_safe_b, f"Guardrail falsely flagged safe response: {reason_b}"
    print("-> Sub-case B: [PASS - Verified faithful grounding]")

    print("\n" + "=" * 70)
    print("CLAUDE'S PERSPECTIVE TEST SUITE PASSED (100%)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_claude_test_suite()
    exit(0 if success else 1)
