"""Test suite verifying persistent taste editing across terminal restarts.
Simulates shutting down the process and reloading from disk in a fresh instance.
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
from engine import MovieDataEngine


def test_user_memory_persistence():
    print("=" * 60)
    print("TESTING USER MEMORY PERSISTENCE ACROSS RESTARTS")
    print("=" * 60)

    mem_file = root_dir / "data" / "user_memory.json"

    # Step 0: Clean state
    if mem_file.exists():
        mem_file.write_text("{}", encoding="utf-8")

    # -------------------------------------------------------------
    # Test 1: User 1 baseline check
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing Terminal 1 for User 1...")
    agent1 = MovieAgent()
    p1 = agent1.engine.get_user_profile(1)
    base_genres = [g["genre"] for g in p1["top_genres"]]
    print(f"-> User 1 initial genres ({len(base_genres)}): {base_genres}")
    assert len(base_genres) >= 5, f"Expected at least 5 baseline genres, got {len(base_genres)}"

    # -------------------------------------------------------------
    # Test 2: Modify User 1 taste to 2 genres
    # -------------------------------------------------------------
    print("\n[Step 2] User 1 requests: 'đổi sở thích thành 2 cái: Hành động và Khoa học viễn tưởng'")
    res = agent1.chat(user_id=1, user_message="đổi sở thích thành 2 cái: Hành động và Khoa học viễn tưởng")
    print(f"-> Bot response:\n{res['answer']}")

    # Check memory file on disk immediately
    assert mem_file.exists(), "Memory file must exist on disk"
    with open(mem_file, "r", encoding="utf-8") as f:
        disk_data = json.load(f)
    print(f"-> Disk contents for User 1: {disk_data.get('1')}")
    u1_disk_favs = disk_data.get("1", {}).get("favorite_genres", [])
    assert u1_disk_favs == ["Action", "Sci-Fi"], f"Expected ['Action', 'Sci-Fi'], got {u1_disk_favs}"

    # -------------------------------------------------------------
    # Test 3: SIMULATE TERMINAL RESTART (kill agent1, start agent2)
    # -------------------------------------------------------------
    print("\n[Step 3] Simulating Terminal Shutdown & Restarting in a New Terminal...")
    del agent1  # destroy previous process memory

    # Fresh agent instance loading from disk
    agent2 = MovieAgent()
    p2 = agent2.engine.get_user_profile(1)
    new_genres = [g["genre"] for g in p2["top_genres"]]
    print(f"-> User 1 genres after restart ({len(new_genres)}): {new_genres}")

    assert len(new_genres) == 2, f"FAILED: Expected 2 genres after restart, got {len(new_genres)} ({new_genres})"
    assert new_genres == ["Action", "Sci-Fi"], f"FAILED: Expected ['Action', 'Sci-Fi'], got {new_genres}"
    print("-> SUCCESS: Terminal restart preserved the exact 2 updated genres!")

    # -------------------------------------------------------------
    # Test 4: Cold-Start User 0 taste addition and restart
    # -------------------------------------------------------------
    print("\n[Step 4] Testing User 0 (Cold-Start)...")
    p0 = agent2.engine.get_user_profile(0)
    print(f"-> User 0 initial genres: {[g['genre'] for g in p0['top_genres']]}")

    print("-> User 0 requests: 'Thêm sở thích phim hoạt hình và hài'")
    res0 = agent2.chat(user_id=0, user_message="Thêm sở thích phim hoạt hình và hài")
    print(f"-> Bot response:\n{res0['answer']}")

    # Restart terminal again
    del agent2
    agent3 = MovieAgent()
    p0_after = agent3.engine.get_user_profile(0)
    u0_genres = [g["genre"] for g in p0_after["top_genres"]]
    print(f"-> User 0 genres after restart: {u0_genres}")
    assert "Animation" in u0_genres and "Comedy" in u0_genres, f"Expected Animation and Comedy, got {u0_genres}"
    print("-> SUCCESS: User 0 preferences persisted across restart!")

    # -------------------------------------------------------------
    # Test 5: Reset / Clear preferences
    # -------------------------------------------------------------
    print("\n[Step 5] User 1 requests: 'Xóa toàn bộ gu của tôi'")
    res_reset = agent3.chat(user_id=1, user_message="Xóa toàn bộ gu của tôi")
    print(f"-> Bot response:\n{res_reset['answer']}")

    del agent3
    agent4 = MovieAgent()
    p1_reset = agent4.engine.get_user_profile(1)
    reset_genres = [g["genre"] for g in p1_reset["top_genres"]]
    print(f"-> User 1 genres after reset and restart ({len(reset_genres)}): {reset_genres}")
    assert len(reset_genres) >= 5, "Expected baseline 5 genres after reset"
    print("-> SUCCESS: Reset restored default historical profile after restart!")

    print("\n" + "=" * 60)
    print("ALL PERSISTENCE TESTS PASSED (100%)")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_user_memory_persistence()
    exit(0 if success else 1)
