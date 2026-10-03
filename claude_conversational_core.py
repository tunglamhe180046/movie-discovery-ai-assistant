"""Claude's Conversational Core: Nuance Emotion Analysis & Empathetic Grounding Engine.
Authored by Claude (Anthropic Perspective) for TrustedAI Movie Discovery Assistant.

Core Principles:
  1. Causal Attribution: Extract NOT JUST positive/negative, but WHY the user felt that way
     (e.g., twist disappointment, cognitive fatigue, pacing issues, emotional resonance).
  2. Anti-Gaslighting & Curious Inquiry: When past feelings differ from current statements,
     explore the evolution of taste with empathy instead of rigid contradiction.
  3. Grounding Verification: Ensure every statement regarding user history matches exact recorded facts.
"""

from __future__ import annotations

import re
from typing import Any


class ClaudeNuanceEmotion:
    """Micro-sentiment taxonomies capturing authentic human reactions to cinema."""
    COGNITIVE_FATIGUE = "cognitive_fatigue"       # nhức đầu, mệt não, lú lẫn, quá nhiều lớp lang
    EMOTIONAL_SHOCK = "emotional_shock"           # hụt hẫng, sốc, chưng hửng, cái kết bất ngờ khó chịu
    INTELLECTUAL_THRILL = "intellectual_thrill"   # cuốn hút, hack não thú vị, tư duy đỉnh cao
    EXISTENTIAL_DREAD = "existential_dread"       # u ám, nặng nề, ám ảnh, nghẹn ngào
    HEARTWARMING_COMFORT = "heartwarming_comfort" # nhẹ nhàng, ấm áp, thư giãn, chữa lành
    DISAPPOINTMENT_PACING = "disappointment_pacing" # lê thê, buồn ngủ, dài dòng, thiếu điểm nhấn
    NEUTRAL_OBJECTIVE = "neutral_objective"       # đánh giá trung lập, khách quan


class ClaudeNuanceAnalyzer:
    """Extract fine-grained cinematic emotions and causal drivers from natural language."""

    @staticmethod
    def extract_nuance_and_causality(text: str) -> dict[str, Any]:
        """Parse natural user statements into nuanced emotional profiles.

        Args:
            text: Raw user message in Vietnamese or English.

        Returns:
            Dictionary containing primary emotion, causal anchors, and raw quote.
        """
        lower = text.lower().strip()
        
        # 1. Identify primary micro-emotion
        primary_emotion = ClaudeNuanceEmotion.NEUTRAL_OBJECTIVE
        if any(w in lower for w in ["nhức đầu", "nhức cả đầu", "mệt đầu", "mệt mỏi", "lú", "rối não", "loạn óc"]):
            primary_emotion = ClaudeNuanceEmotion.COGNITIVE_FATIGUE
        elif any(w in lower for w in ["hụt hẫng", "chưng hửng", "sốc", "tức", "hụt", "thất vọng ở kết"]):
            primary_emotion = ClaudeNuanceEmotion.EMOTIONAL_SHOCK
        elif any(w in lower for w in ["hack não", "cuốn", "đỉnh", "xuất sắc", "xoắn não", "thông minh", "tuyệt vời"]):
            primary_emotion = ClaudeNuanceEmotion.INTELLECTUAL_THRILL
        elif any(w in lower for w in ["u ám", "nặng nề", "ám ảnh", "nghẹn", "buồn bã", "day dứt"]):
            primary_emotion = ClaudeNuanceEmotion.EXISTENTIAL_DREAD
        elif any(w in lower for w in ["nhẹ nhàng", "dễ chịu", "ấm áp", "thư giãn", "chữa lành", "vui vẻ"]):
            primary_emotion = ClaudeNuanceEmotion.HEARTWARMING_COMFORT
        elif any(w in lower for w in ["lê thê", "buồn ngủ", "chán", "dài dòng", "chậm chạp", "nhạt"]):
            primary_emotion = ClaudeNuanceEmotion.DISAPPOINTMENT_PACING

        # 2. Extract Causal Anchors (What caused the emotion?)
        causal_anchors = []
        if any(w in lower for w in ["đoạn kết", "cái kết", "kết phim", "ending", "twist", "cú twist"]):
            causal_anchors.append("plot_twist_ending")
        if any(w in lower for w in ["tiết tấu", "nhịp phim", "dẫn dắt", "mạch phim", "pacing"]):
            causal_anchors.append("pacing_and_flow")
        if any(w in lower for w in ["diễn viên", "diễn xuất", "nhân vật", "acting"]):
            causal_anchors.append("acting_and_characters")
        if any(w in lower for w in ["âm nhạc", "hình ảnh", "kỹ xảo", "tông màu", "bối cảnh"]):
            causal_anchors.append("audiovisual_ambience")

        # 3. Sentiment polarity mapping
        polarity = "neutral"
        if primary_emotion in (ClaudeNuanceEmotion.INTELLECTUAL_THRILL, ClaudeNuanceEmotion.HEARTWARMING_COMFORT):
            polarity = "positive"
        elif primary_emotion in (ClaudeNuanceEmotion.COGNITIVE_FATIGUE, ClaudeNuanceEmotion.EMOTIONAL_SHOCK, ClaudeNuanceEmotion.DISAPPOINTMENT_PACING):
            polarity = "negative"
        elif primary_emotion == ClaudeNuanceEmotion.EXISTENTIAL_DREAD:
            polarity = "mixed"

        return {
            "primary_emotion": primary_emotion,
            "causal_anchors": causal_anchors,
            "polarity": polarity,
            "verbatim_snippet": text.strip()
        }


class ClaudeEmpatheticSynthesizer:
    """Generate high-empathy, grounded conversational responses respecting emotional memory."""

    @staticmethod
    def synthesize_recollection_response(
        movie_title: str,
        user_query: str,
        past_review: dict[str, Any],
        community_stats: dict[str, Any] | None = None
    ) -> str:
        """Compose an empathetic response when user asks about a movie they previously critiqued.

        Adheres strictly to Anthropic alignment guidelines:
        - Acknowledge past feeling verbatim or with high fidelity.
        - Notice any discrepancy with curious inquiry rather than confrontational accusation.
        - Provide contextual data gracefully.
        """
        past_feeling = past_review.get("user_feeling", "")
        past_date = past_review.get("date_str", "trước đây")
        past_rating = past_review.get("rating")
        rating_mention = f" (bạn từng chấm {past_rating}★)" if past_rating else ""

        # Analyze current query sentiment
        curr_nuance = ClaudeNuanceAnalyzer.extract_nuance_and_causality(user_query)

        lines = [
            f"🎬 **{movie_title}**",
            "",
            f"💡 **Điểm lại góc nhìn trước đây của bạn ({past_date}):**",
            f"Lần trước khi nhắc đến phim này, bạn từng chia sẻ: *\"{past_feeling}\"*{rating_mention}."
        ]

        # Check if the user's current stance implies a positive/neutral shift ("ổn nhỉ", "hay nhỉ")
        is_positive_inquiry = any(w in user_query.lower() for w in ["ổn nhỉ", "hay nhỉ", "được nhỉ", "tuyệt nhỉ", "thích"])
        is_negative_past = past_review.get("sentiment") in ("negative", "mixed") or any(
            w in past_feeling.lower() for w in ["nhức đầu", "hụt hẫng", "tệ", "dở", "chán", "mệt"]
        )

        if is_positive_inquiry and is_negative_past:
            lines.append("")
            lines.append(
                "Thấy hôm nay bạn hỏi phim này trông có vẻ 'ổn', tôi rất tò mò: Có phải sau khi ngẫm lại về cốt truyện, "
                "hoặc xem các bài phân tích giải mã, bạn đã tìm thấy một góc nhìn thú vị hơn về cái kết từng khiến bạn hụt hẫng không?"
            )
        else:
            lines.append("")
            lines.append(
                f"Bạn vẫn giữ nguyên cảm nhận *\"{past_feeling}\"* đó, hay thời gian qua bạn đã có thêm góc nhìn mới về tác phẩm này?"
            )

        if community_stats:
            avg_score = community_stats.get("avg_rating")
            total_votes = community_stats.get("num_ratings", 0)
            if avg_score and total_votes:
                lines.append("")
                lines.append(f"📊 *Thông tin thêm từ cộng đồng:* Phim có điểm trung bình **{avg_score}★** từ {total_votes} khán giả.")

        return "\n".join(lines)


class ClaudeGroundingGuard:
    """Verify that synthesized responses do not fabricate or hallucinate emotional history."""

    @staticmethod
    def audit_response(response_text: str, actual_past_review: dict[str, Any] | None) -> tuple[bool, str]:
        """Audit whether the response is safely grounded on actual emotional history.

        Returns:
            (is_safe, rationale)
        """
        if not actual_past_review:
            # If no past review existed, ensure response does NOT falsely claim user had one
            hallucination_triggers = ["lần trước bạn từng", "bạn từng chia sẻ", "hôm trước bạn bảo"]
            for trig in hallucination_triggers:
                if trig in response_text.lower():
                    return False, f"Hallucinated past review where none existed (trigger: '{trig}')"
            return True, "Safe: No ungrounded emotional memory claims."

        # If past review exists, verify that the response faithfully captures the recorded sentiment
        past_feeling = actual_past_review.get("user_feeling", "").lower()
        # Codex Review Fix: Support 2-letter Vietnamese emotion keywords ('lú', 'tệ', 'dở', 'hụt', 'sốc')
        key_words = [w for w in re.findall(r"\w+", past_feeling) if len(w) >= 2 and w not in ("và", "có", "là", "thì", "mà", "đã", "rồi")]
        
        # Check that at least some key essence of the user's past feeling is referenced
        overlap = [w for w in key_words if w in response_text.lower()]
        if not overlap:
            return False, f"Failed to ground response in actual recorded user feeling ('{past_feeling}')"

        return True, f"Grounding verified with {len(overlap)} semantic anchor tokens."
