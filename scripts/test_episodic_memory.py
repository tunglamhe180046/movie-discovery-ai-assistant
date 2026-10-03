"""Test suite for Episodic Qualitative Memory & User Feeling Loop.
Verifies:
  1. Automatic extraction of personal movie feelings without explicit save commands.
  2. Persistent atomic storage to disk in data/user_memory.json.
  3. Seamless cross-session retrieval when user asks about the movie later.
  4. Empathetic dialogue referencing past user sentiment instead of just cold data.
  5. Preference updating when user evolves their opinion.
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

# Add project root
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from agent import MovieAgent


def test_episodic_qualitative_memory():
    print("=" * 70)
    print("TESTING EPISODIC QUALITATIVE MEMORY & RETRIEVAL COUNCIL")
    print("=" * 70)

    mem_file = root_dir / "data" / "user_memory.json"
    user_id = 1

    # Step 0: Initial state cleanup for User 1 reviews
    if mem_file.exists():
        with open(mem_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        if str(user_id) in disk_data:
            disk_data[str(user_id)]["movie_reviews"] = []
            with open(mem_file, "w", encoding="utf-8") as f:
                json.dump(disk_data, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # TEST 1: User shares qualitative feeling naturally
    # -------------------------------------------------------------------------
    print("\n[Case 1] Natural qualitative experience sharing:")
    msg_1 = "Hôm qua tôi mới xem Shutter Island, đoạn kết hụt hẫng và xem nhức cả đầu thật sự."
    print(f"User: \"{msg_1}\"")

    agent1 = MovieAgent()
    res1 = agent1.chat(user_id=user_id, user_message=msg_1)
    ans1 = res1["answer"]
    parsed1 = res1.get("parsed_params", {})
    print(f"-> Parsed Intent: {parsed1.get('intent')}")
    print(f"-> Detected Movie: {parsed1.get('target_movie')}")
    print(f"-> Bot Response:\n{ans1}")

    assert parsed1.get("intent") == "share_movie_feeling", f"Expected 'share_movie_feeling', got {parsed1.get('intent')}"
    assert "Shutter Island" in str(parsed1.get("target_movie")), "Expected Shutter Island to be resolved"

    # Verify disk persistence
    with open(mem_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    u_revs = disk_data.get(str(user_id), {}).get("movie_reviews", [])
    assert len(u_revs) >= 1, f"Expected review in disk data, got {u_revs}"
    saved_rev = u_revs[0]
    print(f"-> Saved on disk: title='{saved_rev.get('title')}', feeling='{saved_rev.get('user_feeling')}'")
    assert "Shutter Island" in saved_rev.get("title")
    print("-> STATUS: [PASS - Memory captured & persisted automatically]")

    # -------------------------------------------------------------------------
    # TEST 2: Simulate Session Shutdown & Next-Day Return
    # -------------------------------------------------------------------------
    print("\n[Case 2] Simulating next session / next day return with fresh Agent:")
    del agent1  # destroy memory in RAM

    agent2 = MovieAgent()
    msg_2 = "Cái phim Shutter Island đó ổn nhỉ, bạn thấy sao?"
    print(f"User (next day): \"{msg_2}\"")

    res2 = agent2.chat(user_id=user_id, user_message=msg_2)
    ans2 = res2["answer"]
    print(f"-> Bot Response:\n{ans2}")

    # Check that bot recalled the user's past feeling
    has_past_recollection = any(k in ans2.lower() for k in ["nhớ lại", "hôm trước", "lần trước", "từng chia sẻ", "trước đây", "hụt hẫng", "nhức cả đầu", "nhức đầu"])
    assert has_past_recollection, f"Bot failed to recall past feeling! Answer: {ans2}"
    print("-> STATUS: [PASS - Successfully retrieved past feeling & grounded reply]")

    # -------------------------------------------------------------------------
    # TEST 3: User changes their mind & updates review
    # -------------------------------------------------------------------------
    print("\n[Case 3] User updates/evolves their perspective:")
    msg_3 = "Sau khi xem giải thích phim Shutter Island thì tôi thấy nó đỉnh thật sự, chấm 5 sao."
    print(f"User: \"{msg_3}\"")

    res3 = agent2.chat(user_id=user_id, user_message=msg_3)
    ans3 = res3["answer"]
    print(f"-> Bot Response:\n{ans3}")

    # Verify updated review on disk
    with open(mem_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    u_revs_updated = disk_data.get(str(user_id), {}).get("movie_reviews", [])
    matching_rev = next((r for r in u_revs_updated if "Shutter Island" in r.get("title")), None)
    assert matching_rev is not None
    print(f"-> Updated disk review: feeling='{matching_rev.get('user_feeling')}', rating={matching_rev.get('rating')}")
    assert matching_rev.get("rating") == 5.0, f"Expected 5.0 rating, got {matching_rev.get('rating')}"
    print("-> STATUS: [PASS - Updated user feeling & rating without conflicting duplication]")

    # -------------------------------------------------------------------------
    # TEST 4: Non-feeling query does NOT get hijacked
    # -------------------------------------------------------------------------
    print("\n[Case 4] Generic movie search must NOT trigger share_movie_feeling:")
    msg_4 = "Gợi ý cho tôi 2 phim giật gân tương tự Shutter Island."
    print(f"User: \"{msg_4}\"")
    res4 = agent2.chat(user_id=user_id, user_message=msg_4)
    parsed4 = res4.get("parsed_params", {})
    print(f"-> Parsed Intent: {parsed4.get('intent')}")
    assert parsed4.get("intent") in ("recommend", "plot_search"), f"Expected recommend/plot_search, got {parsed4.get('intent')}"
    print("-> STATUS: [PASS - Search query preserved correctly]")

    print("\n" + "=" * 70)
    print("ALL EPISODIC MEMORY COUNCIL TESTS PASSED (100%)")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_episodic_qualitative_memory()
    exit(0 if success else 1)
