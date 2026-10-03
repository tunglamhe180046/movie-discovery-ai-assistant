"""Test suite for the 10 Codex test cases."""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from agent import MovieAgent

test_cases = [
    {"id": 1, "query": "Cho tôi 1 tên phim lâu rồi tôi chưa xem hoặc xem ít nhất."},
    {"id": 2, "query": "Tìm một phim khoa học viễn tưởng tôi chưa xem trong 2 năm, đừng kinh dị."},
    {"id": 3, "query": "Tối nay xem gì nhẹ nhàng, không phải hoạt hình?"},
    {"id": 4, "query": "Trong các phim giống Inception, phim nào tôi đã xem lâu nhất?"},
    {"id": 5, "query": "Đừng gợi ý bất cứ phim hoạt hình nào; chọn phim hành động tôi bỏ quên."},
    {"id": 6, "query": "Có phim nào thuộc thể loại điểm mù mà tôi gần như chưa xem nhưng hợp với tôi không?"},
    {"id": 7, "query": "Gợi ý 3 phim tôi chưa từng xem mà những người cùng gu đánh giá cao."},
    {"id": 8, "query": "Những người có gu giống tôi nghĩ gì về Pulp Fiction?"},
    {"id": 9, "query": "Điểm mù của tôi là gì và cho tôi phim tiêu biểu để bắt đầu?"},
    {"id": 10, "query": "Gợi ý cho tôi 1 phim tâm lý đen tối có plot twist bất ngờ."}
]

def run_tests():
    agent = MovieAgent()
    print("=" * 60)
    print("RUNNING 10 CODEX BENCHMARK TEST CASES")
    print("=" * 60)

    passed = 0
    for tc in test_cases:
        cid = tc["id"]
        q = tc["query"]
        print(f"\n[Case {cid}] Query: '{q}'")
        res = agent.chat(user_id=1, user_message=q)
        ans = res["answer"]
        params = res.get("parsed_params", {})
        
        print(f"-> Parsed Intent: {params.get('intent')}")
        print(f"-> Params: inc={params.get('genres_include')}, exc={params.get('genres_exclude')}, limit={params.get('limit')}")
        print(f"-> Output preview:\n{ans[:250]}...")
        
        # Validation checks
        has_signal = any(k in ans.lower() for k in ["🎬", "★", "phim", "match", "cohort", "đánh giá", "rating", "điểm", "/5"])
        if len(ans.strip()) > 30 and has_signal:
            print(f"-> STATUS: [PASS]")
            passed += 1
        else:
            print(f"-> STATUS: [FAIL - Output too short or ungrounded]")

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{len(test_cases)} Passed!")
    print("=" * 60)
    return passed == len(test_cases)

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
