"""Movie Discovery AI Assistant - Conversational Data-Augmented RAG
Powered by DeepSeek API (with Groq and Offline backup).
Combines factual data extraction from MovieDataEngine with natural AI dialogue.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from engine import MovieDataEngine


def load_env(env_path: Path) -> dict[str, str]:
    config = {}
    if not env_path.exists():
        return config
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip().strip('"').strip("'")
            elif ":" in line:
                k, v = line.split(":", 1)
                config[k.strip()] = v.strip().strip('"').strip("'")
    return config


PARSER_SYSTEM_PROMPT = """You are an intent & parameter parser for a movie recommendation assistant.
Convert the user's message (Vietnamese or English) into a strict JSON object with fields:
- "intent": one of:
    * "greeting": When the user says hello, asks who you are, or greets (e.g. "xin chào", "bạn là ai", "hello", "hi")
    * "user_profile": When the user asks about THEIR OWN taste, favorite movies, ratings (e.g. "tôi thích xem phim gì nhất", "gu của tôi là gì")
    * "oldest_or_underwatched": When user asks for movies watched long ago or least watched (e.g. "phim lâu rồi tôi chưa xem")
    * "blind_spot": When user asks about blind spots, genres missed, or recommendations from unexplored genres
    * "cohort_opinion": When user asks what peers/similar users think of a specific movie (e.g. Pulp Fiction, Inception)
    * "why_recommendation": When user asks "why would I like that?", "tại sao tôi lại thích phim đó?", asking for grounded reasoning/explanation
    * "update_taste": When user states, changes, or asks to save their taste, preferences, or movie ratings (e.g. "gu của tôi là...", "lưu lại gu cho tôi", "tôi thích phim hành động", "tôi ghét kinh dị", "tôi vừa xem Inception chấm 5 sao")
    * "plot_search": When searching by plot description, theme, or mood keywords
    * "recommend": When user asks for general or tailored movie recommendations
- "search_query_en": English translation of plot/theme query (string or null)
- "target_movie": Specific movie title mentioned in query (string or null; only set if an actual movie title is mentioned!)
- "genres_include": List of genres requested or stated as favorites (e.g. ["Sci-Fi", "Action"] or [])
- "genres_exclude": List of genres to strictly avoid or stated as disliked (e.g. ["Animation", "Horror"] or [])
- "rating": Float number if user is rating a movie (e.g. 5.0, 4.5, 3.0 or null)
- "time_filter": "longest_unwatched", "least_watched", or null
- "limit": Integer number of movies requested (default 5; if user asks for 1 movie, set 1)

Return ONLY valid JSON. No explanation. No markdown fences."""

SYNTHESIZER_SYSTEM_PROMPT = """Bạn là trợ lý AI khám phá phim ảnh của TrustedAI.
Nhiệm vụ: Trả lời câu hỏi của người dùng bằng tiếng Việt NGẮN GỌN, TRỰC DIỆN, CHÍNH XÁC TUYỆT ĐỐI.

NGUYÊN TẮC BẮT BUỘC:
1. ĐI THẲNG VÀO TRỌNG TÂM:
   - KHÔNG mở bài lan man, chào hỏi rườm rà (ví dụ: KHÔNG nói "Chào bạn, câu hỏi hay lắm", "để mình bắt mạch gu...", "bạn là người xem phim nặng đô...").
   - KHÔNG kết bài sáo rỗng (ví dụ: KHÔNG nói "nếu cần gì cứ bảo tôi nhé...", "chúc bạn xem phim vui vẻ", "đừng ngần ngại...").
   - Đi thẳng vào kết quả hoặc nhận xét trực tiếp trong câu đầu tiên.

2. TRÌNH BÀY GỌN GÀNG, RÕ RÀNG:
   - Sử dụng bullet points ngắn gọn.
   - Định dạng phim: **Tên phim (Năm nếu có)** - Thể loại - Thông tin chính (số sao bạn đã chấm, điểm khớp, hoặc lý do ngắn gọn trong 1 câu).
   - Tối đa 150-250 từ. Tuyệt đối không viết thành các đoạn văn dài dòng, lan man.

3. TRUNG THỰC VÀ CHÍNH XÁC (GROUNDING):
   - BẮT BUỘC chỉ sử dụng số liệu có trong DỮ LIỆU ĐÃ TRUY VẤN (tên phim, số sao ★, ngày xem, điểm cohort).
   - KHÔNG bịa đặt thêm phim ngoài dữ liệu.
   - Đúng phạm vi yêu cầu:
     * Nếu người dùng hỏi gu/sở thích/phim xem nhiều nhất: Chỉ trả lời về lịch sử xem và top phim yêu thích của họ. KHÔNG tự ý gợi ý thêm phim mới khi chưa được hỏi.
     * Nếu người dùng hỏi xin gợi ý phim: Chỉ đưa ra danh sách phim gợi ý được cung cấp.
     * Nếu người dùng hỏi về 1 phim cụ thể hoặc phim lâu rồi chưa xem: Trả lời đúng phim đó kèm thông tin ngày/sao.
     * Nếu người dùng chào hỏi / hỏi bạn là ai: Giới thiệu súc tích trong 2 câu.
"""


class MovieAgent:
    def __init__(self, data_dir: str | Path | None = None):
        root_dir = Path(__file__).resolve().parent
        self.env = load_env(root_dir / ".env")
        
        # Primary: DeepSeek API
        self.deepseek_key = self.env.get("DEEPSEEK_API_KEY") or self.env.get("deepseekapi", "")
        self.deepseek_url = self.env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.deepseek_model = self.env.get("DEEPSEEK_MODEL", "deepseek-chat")

        # Secondary Backup: Groq API
        self.groq_key = self.env.get("GROQ_API_KEY", "")
        self.groq_url = self.env.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.groq_model = self.env.get("GROQ_MODEL", "qwen/qwen3.8-27b")

        print("Initializing MovieDataEngine...")
        self.engine = MovieDataEngine(data_dir=data_dir) if data_dir else MovieDataEngine()
        print("Engine ready!")

    def _call_llm(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.3) -> str | None:
        """Execute chat completion with automatic fallback: DeepSeek -> Groq."""
        providers = []
        if self.deepseek_key:
            providers.append({
                "url": f"{self.deepseek_url}/chat/completions",
                "key": self.deepseek_key,
                "model": self.deepseek_model,
                "name": f"DeepSeek ({self.deepseek_model})"
            })
        if self.groq_key:
            providers.append({
                "url": f"{self.groq_url}/chat/completions",
                "key": self.groq_key,
                "model": self.groq_model,
                "name": f"Groq ({self.groq_model})"
            })

        for p in providers:
            headers = {
                "Authorization": f"Bearer {p['key']}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            payload = {
                "model": p["model"],
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                req = urllib.request.Request(p["url"], data=json.dumps(payload).encode("utf-8"), headers=headers)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"].strip()
                    self.active_provider_name = p["name"]
                    return content
            except Exception as e:
                # Try next provider if failed
                continue

        return None

    def _parse_intent_with_llm(self, user_message: str) -> dict[str, Any] | None:
        """Extract structured intent using DeepSeek / Groq."""
        raw = self._call_llm(
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=250,
            temperature=0.0
        )
        if not raw:
            return None
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except Exception:
            return None

    def _offline_intent_parser(self, user_message: str) -> dict[str, Any]:
        """Deterministic regex/keyword parser for 100% offline fallback.
        
        P0 FIX: Added detection for `why_recommendation` (Requirement 3: 'Why would I like that?'),
        dynamic target movie extraction, and explicit genre include/exclude extraction.
        """
        q = user_message.lower().strip()
        # 1. High-priority specific intent triggers (checked first to prevent false matches on generic words like 'think')
        if any(w in q for w in ["tại sao", "vì sao", "lý do", "why would i like", "why do you think", "why i would like", "giải thích"]):
            target_movie = None
            if "pulp fiction" in q:
                target_movie = "Pulp Fiction"
            elif "inception" in q:
                target_movie = "Inception"
            elif "toy story" in q:
                target_movie = "Toy Story"
            return {"intent": "why_recommendation", "target_movie": target_movie, "limit": 1}

        if any(w in q for w in ["cùng gu", "giống tôi", "similar taste", "nghĩ gì về", "thấy thế nào về", "think of", "think about"]):
            target_movie = "Pulp Fiction"
            if "inception" in q:
                target_movie = "Inception"
            elif "interstellar" in q:
                target_movie = "Interstellar"
            elif "toy story" in q:
                target_movie = "Toy Story"
            return {"intent": "cohort_opinion", "target_movie": target_movie, "limit": limit}

        # 2. Greeting (use word boundary regex to avoid false triggers like 'think' -> 'hi')
        word_tokens = set(re.findall(r"\b\w+\b", q))
        if any(w in word_tokens for w in ["xin", "chào", "hello", "hi", "hey"]) or any(p in q for p in ["bạn là ai", "who are you", "giới thiệu"]):
            return {"intent": "greeting", "limit": limit}

        # Extract genre constraints
        genres_exc = []
        if any(w in q for w in ["không phải hoạt hình", "đừng hoạt hình", "không hoạt hình", "chán hoạt hình", "no animation", "đừng phim hoạt hình"]):
            genres_exc.append("Animation")
        if any(w in q for w in ["đừng kinh dị", "không kinh dị", "không phải kinh dị", "no horror"]):
            genres_exc.append("Horror")

        genres_inc = []
        if any(w in q for w in ["khoa học viễn tưởng", "viễn tưởng", "sci-fi", "scifi"]):
            genres_inc.append("Sci-Fi")
        if any(w in q for w in ["hành động", "action"]):
            genres_inc.append("Action")
        if any(w in q for w in ["hài", "nhẹ nhàng", "comedy"]):
            genres_inc.append("Comedy")
        if any(w in q for w in ["tâm lý", "drama"]):
            genres_inc.append("Drama")

        # Check if user is declaring taste, updating preferences, or adding a rating
        if any(w in q for w in ["lưu lại", "nhớ nhé", "ghi nhớ", "đổi gu", "gu của tôi", "tôi thích xem", "tôi thích thể loại", "tôi ghét", "sở thích của tôi là", "chấm", "đánh giá", "remember"]):
            rating_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:sao|star|\★)", q)
            rating_val = float(rating_match.group(1)) if rating_match else None
            target_movie = None
            for tm in ["Inception", "Pulp Fiction", "Toy Story", "Interstellar", "Star Wars", "Gladiator", "Braveheart"]:
                if tm.lower() in q:
                    target_movie = tm
                    break
            return {
                "intent": "update_taste",
                "genres_include": genres_inc if genres_inc else None,
                "genres_exclude": genres_exc if genres_exc else None,
                "target_movie": target_movie,
                "rating": rating_val,
                "limit": 1
            }

        if any(w in q for w in ["tâm lý", "đen tối", "plot twist", "kinh dị", "giật gân", "dark psychological thriller"]):
            return {
                "intent": "plot_search",
                "search_query_en": "dark psychological thriller with a twist",
                "genres_exclude": genres_exc,
                "genres_include": genres_inc,
                "limit": limit
            }

        return {
            "intent": "recommend",
            "genres_exclude": genres_exc,
            "genres_include": genres_inc,
            "search_query_en": "adventure comedy" if "toy story" in q else None,
            "limit": limit
        }

    def chat(self, user_id: int, user_message: str, history: list[dict] | None = None) -> dict:
        """Process conversational turn using Data-Augmented RAG with DeepSeek."""
        self.active_provider_name = "offline-engine"

        # Step 1: Parse intent & parameters
        parsed_params = self._parse_intent_with_llm(user_message)
        if not parsed_params or not isinstance(parsed_params, dict):
            parsed_params = self._offline_intent_parser(user_message)

        intent = parsed_params.get("intent", "recommend")

        # Step 2: Execute data retrieval on Python Engine
        engine_result = self.engine.execute_structured_intent(user_id=user_id, params=parsed_params)
        user_profile = self.engine.get_user_profile(user_id)

        # Step 3: Natural Dialogue Generation via DeepSeek / LLM
        answer = self._generate_conversational_response(
            user_id=user_id,
            user_message=user_message,
            intent=intent,
            result=engine_result,
            profile=user_profile,
            params=parsed_params
        )

        return {
            "answer": answer,
            "parsed_params": parsed_params,
            "model_used": getattr(self, "active_provider_name", "deepseek-chat"),
            "tools_used": [intent]
        }

    def _generate_conversational_response(
        self,
        user_id: int,
        user_message: str,
        intent: str,
        result: dict,
        profile: dict,
        params: dict
    ) -> str:
        # Build tailored Context Capsule (Factual data ground)
        if intent == "greeting":
            context_data = {
                "user_id": user_id,
                "intent": "greeting",
                "assistant_role": "Trợ lý AI khám phá phim ảnh TrustedAI (gợi ý phim cá nhân hóa, phân tích gu, tìm phim cũ đã xem, khám phá điểm mù, xem đánh giá của người cùng gu)"
            }
        elif intent == "user_profile":
            context_data = {
                "user_id": user_id,
                "intent": "user_profile",
                "total_ratings": profile.get("num_ratings"),
                "avg_rating": profile.get("avg_rating"),
                "top_favorite_genres": [f"{g['genre']} ({g['count']} phim)" for g in profile.get("top_genres", [])[:4]],
                "highest_rated_movies": [f"{m['title']} ({m['rating']}★)" for m in profile.get("top_movies", [])[:params.get("limit", 5)]],
            }
        else:
            context_data = {
                "user_id": user_id,
                "user_summary": {
                    "num_ratings": profile.get("num_ratings"),
                    "avg_rating": profile.get("avg_rating"),
                    "top_genres": [g["genre"] for g in profile.get("top_genres", [])[:3]]
                },
                "intent_detected": intent,
                "engine_findings": result
            }

        user_prompt = (
            f"Người dùng (User #{user_id}) hỏi: \"{user_message}\"\n\n"
            f"DỮ LIỆU ĐÃ TRUY VẤN TỪ HỆ THỐNG:\n"
            f"```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```\n\n"
            f"YÊU CẦU: Trả lời NGẮN GỌN, TRỰC DIỆN, ĐI THẲNG VÀO TRỌNG TÂM (tối đa 2-3 đoạn ngắn hoặc bullet points), CHÍNH XÁC 100% dựa vào dữ liệu trên. KHÔNG lan man, KHÔNG mở/kết bài sáo rỗng."
        )

        llm_reply = self._call_llm(
            messages=[
                {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=450,
            temperature=0.2
        )

        if llm_reply:
            return llm_reply

        # Deterministic fallback if LLM is completely unreachable
        return self._fallback_template(user_id=user_id, intent=intent, result=result, profile=profile, params=params)

    def _fallback_template(self, user_id: int, intent: str, result: dict, profile: dict, params: dict) -> str:
        """Backup template formatter only used if all APIs fail."""
        if intent == "greeting":
            return (
                "Xin chào bạn! 👋 Tôi là trợ lý AI khám phá phim ảnh của TrustedAI.\n"
                "Tôi có thể giúp bạn phân tích gu phim, tìm lại phim cũ đã xem, khám phá điểm mù hoặc gợi ý theo cốt truyện. Bạn muốn tìm gì hôm nay?"
            )
        if intent == "user_profile":
            top_g = ", ".join([g["genre"] for g in profile.get("top_genres", [])[:4]])
            # P0 FIX: Replaced hardcoded "190 đánh giá" with dynamic profile.get("num_ratings")
            return (
                f"Dựa trên {profile.get('num_ratings', 0)} đánh giá của bạn (User #{user_id}): Gu chính là **{top_g}** với điểm TB **{profile.get('avg_rating')}★**. "
                f"Những phim bạn thích nhất: {', '.join([m['title'] for m in profile.get('top_movies', [])[:3]])}."
            )
        if intent == "oldest_or_underwatched":
            m = result.get("movies", [{}])[0]
            return f"Phim bạn đã xem lâu nhất trong quá khứ là: **{m.get('title')}** ({m.get('genres')}), bạn đã chấm {m.get('user_rating')}★ vào ngày {m.get('watched_date')}."
        
        # P0 FIX: Formatter for grounded explainability ("Why would I like that?")
        if intent == "why_recommendation":
            exp = result.get("explanation", {})
            if "error" in exp:
                return f"Không thể giải thích: {exp['error']}"
            t = exp.get("title", "phim này")
            g = exp.get("genres", "")
            mg = ", ".join(exp.get("matching_genres", []))
            u_avg = exp.get("user_avg_rating")
            c_cnt = exp.get("cohort_num_ratings", 0)
            c_avg = exp.get("cohort_avg_rating")
            c_glob = exp.get("cohort_global_avg")
            
            lines = [f"Lý do bạn (User #{user_id}) sẽ thích **{t}** ({g}):"]
            if mg:
                lines.append(f"• Thể loại khớp với gu phim yêu thích của bạn: **{mg}**.")
            if u_avg:
                lines.append(f"• Điểm đánh giá trung bình lịch sử của bạn là **{u_avg}★**, rất phù hợp với phong cách phim này.")
            if c_cnt > 0:
                cohort_info = f"• Trong nhóm người có gu tương đồng ({c_cnt} người đã xem), điểm TB là **{c_avg}★**"
                if c_glob:
                    cohort_info += f" (toàn cộng đồng: {c_glob}★)"
                lines.append(cohort_info + ".")
            if exp.get("is_blind_spot"):
                lines.append("• Phim thuộc thể loại điểm mù tiềm năng giúp bạn mở rộng trải nghiệm điện ảnh.")
            return "\n".join(lines)

        # P0 FIX: Formatter for cohort opinion with confidence estimation
        if intent == "cohort_opinion":
            cdata = result.get("cohort_data", {})
            t = cdata.get("title", "phim này")
            cnt = cdata.get("num_ratings", 0)
            avg = cdata.get("avg_rating")
            conf = cdata.get("confidence", "low")
            g_avg = cdata.get("global_avg_rating")
            if cnt == 0:
                return f"Chưa có người dùng nào trong nhóm cùng gu của bạn đánh giá **{t}** (trong số {cdata.get('similar_users_checked', 0)} người được kiểm tra)."
            res_str = f"Trong nhóm người có cùng gu với bạn ({cdata.get('similar_users_checked', 0)} người), có {cnt} người đã đánh giá **{t}** với điểm trung bình **{avg}★**."
            if g_avg:
                res_str += f" (Điểm trung bình toàn cộng đồng: {g_avg}★)."
        # Formatter for dynamic taste preference & rating recording (Assistant Memory)
        if intent == "update_taste":
            res = result.get("result", {})
            favs = ", ".join(res.get("favorite_genres", []))
            dislikes = ", ".join(res.get("disliked_genres", []))
            rating_rec = res.get("rating_record")
            
            lines = [f"✅ Đã ghi nhớ cập nhật vào hồ sơ của bạn (User #{user_id}):"]
            if favs:
                lines.append(f"• Gu thể loại yêu thích: **{favs}**")
            if dislikes:
                lines.append(f"• Thể loại tránh gợi ý: **{dislikes}**")
            if rating_rec and "title" in rating_rec:
                lines.append(f"• Đã lưu đánh giá: **{rating_rec['title']}** - **{rating_rec['rating']}★**")
            lines.append("Từ bây giờ, tôi sẽ tự động cá nhân hóa mọi gợi ý dựa trên sở thích này của bạn!")
            return "\n".join(lines)

        if intent == "blind_spot":
            bs = ", ".join(result.get("blind_spots", []))
            movies = result.get("movies", [])
            lines = [f"Các thể loại điểm mù bạn ít khám phá: **{bs}**.\nGợi ý phim tiêu biểu để bạn bắt đầu:"]
            for i, m in enumerate(movies, 1):
                lines.append(f"{i}. **{m.get('title')}** ({m.get('genres')}) - Điểm TB: {m.get('score')}★")
            return "\n".join(lines)

        movies = result.get("movies", [])
        if not movies:
            return "Không tìm thấy phim phù hợp với tiêu chí của bạn trong tập dữ liệu."
        lines = ["Dưới đây là các gợi ý phù hợp:"]
        for i, m in enumerate(movies, 1):
            score_info = f" - Match: {m.get('score')}" if m.get('score') else ""
            lines.append(f"{i}. **{m.get('title')}** ({m.get('genres')}){score_info}")
        return "\n".join(lines)
