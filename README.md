# 🎬 Movie Discovery AI Assistant (TrustedAI Technical Assessment)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Evaluation: 10/10 Passed](https://img.shields.io/badge/Benchmark-10%2F10%20Passed-brightgreen.svg)](scripts/test_codex_cases.py)

An intelligent, data-grounded conversational AI assistant designed to help users discover films by actively **investigating and reasoning over historical dataset signals** (MovieLens: 74k ratings, 5.1k movies with plots, genres, and user tags).

Built for the **TrustedAI - AI Engineer Challenge** by candidate **Tùng Lâm**.

---

## 🌟 Key Highlights & Requirements Fulfillment

Unlike standard search engines or generic LLM chatbots that hallucinate movie facts and ratings, this assistant implements a **Two-Tier Decoupled Architecture**: the LLM handles natural language parsing and conversational synthesis, while all data retrieval, cohort aggregation, and score calculations remain **100% deterministic, grounded, and verifiable**.

| Requirement | Implementation & Technical Mechanism | Status |
|---|---|:---:|
| **1. Personalization by User ID** | Pearson Correlation Collaborative Filtering with mean-centering and positive-similarity weighting ($\ge 3$ co-rated movies). Historical profile aggregation. | ✅ **Done** |
| **2. Multi-Signal Synthesis** | Combines user taste vectors, plot TF-IDF semantic search ($25k$ features), cohort opinions, and strict genre inclusion/exclusion constraints. | ✅ **Done** |
| **3. Grounded Explainability ("Why would I like that?")** | `explain_recommendation()` traces back exact matching genres from the user's top-rated history, peer cohort ratings with sample confidence, and blind spot status. | ✅ **Done** |
| **4. Rigorous Evaluation & Evidence** | Automated benchmark test suite (`10/10 passed`), reproducible JSON artifacts, confidence gating for small sample sizes ($n < 3$), and transparent failure analysis. | ✅ **Done** |

---

## 🤖 Multi-Agent "Vibe Coding" Workflow (Antigravity + Claude Code + Codex)

This project was built, audited, and refined using a **Multi-Agent Collaborative Engineering** workflow, demonstrating how modern AI coding assistants can be orchestrated synergistically to produce production-grade, verifiable software:

```
                  ┌──────────────────────────────────────────────┐
                  │          Google Antigravity (Center)         │
                  │    • Project Lead & Central Orchestrator     │
                  │    • System Architecture & Context Manager   │
                  │    • Task Delegation & Integration Testing   │
                  └───────────────┬──────────────┬───────────────┘
                                  │              │
                   Dispatches Work│              │Requests Audit
                                  ▼              ▼
       ┌──────────────────────────────┐      ┌──────────────────────────────┐
       │       Claude Code CLI        │      │          Codex CLI           │
       │    • Core Implementation     │      │   • Adversarial Reviewer     │
       │    • Logic Refactoring       │      │   • Tech Lead Counter-Check  │
       │    • Code Documentation      │      │   • P0 Edge-Case Hunting     │
       └──────────────────────────────┘      └──────────────────────────────┘
```

### Division of Responsibilities in Development:
1. **Google Antigravity (Central Orchestrator & System Architect):**
   - Served as the master conductor managing repository context, directory structure, prompt requirements, and local execution environment.
   - Designed the Two-Tier Decoupled System architecture based on TrustedAI requirements.
   - Programmatically invoked and coordinated the CLI sub-agents (`claude`, `codex`), merged code changes, resolved conflicts, and executed end-to-end benchmark validations.
2. **Claude Code CLI (Implementation & Refactoring Engine):**
   - Served as the primary implementation engineer, authoring and refactoring core algorithmic modules (`agent.py`, `engine.py`).
   - Added comprehensive technical code comments detailing data flows and fallback routines.
   - Ensured high fault-tolerance with automated offline recovery mechanisms when API limits are reached.
3. **Codex CLI (Adversarial Code Reviewer & Tech Lead):**
   - Executed non-interactively (`codex exec` on `gpt-5.6-terra`) providing critical, adversarial code review from a Tech Lead perspective.
   - **Key P0 Bugs Discovered & Mitigated by Codex**:
     * Caught the candidate leakage bug in `engine.py:649-654` where `genres_include` post-filtering failed when top-K lacked matching genres (causing Sci-Fi queries to return non-Sci-Fi movies).
     * Identified that `explain_recommendation()` was orphaned and not connected to intent parsing for Requirement 3.
     * Flagged statistical vulnerability on sparse cohort queries ($n < 3$ for *Pulp Fiction*), leading to the addition of sample confidence indicators.
     * Flagged hardcoded profile strings (`"190 đánh giá"`) in fallback formatters.
   - Through this adversarial feedback loop, Antigravity directed full remediation of all P0 issues, bringing the test suite to **10/10 Passed**.

---

## 🏛 Architecture

```
[ User / Terminal CLI ]
          │
          ▼  (Natural Language Query in Vietnamese / English)
┌────────────────────────────────────────────────────────────────────────┐
│  Tier 1: Conversational Agent (agent.py)                                │
│  • Structured Intent & Parameter Parser (DeepSeek / Groq / Offline)   │
│  • Intent Routing: profile, recommend, cohort, why_recommendation...   │
│  • Fact-Constrained Natural Synthesizer (No Hallucinations)            │
│  • Automatic API Key Rotation + Graceful Offline Fallback Engine       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Structured Parameters (JSON)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Tier 2: Deterministic Data Engine (engine.py)                         │
│  • User Profiling & Global Genre Distribution Caching                  │
│  • User-User Pearson Collaborative Filtering (User Taste Vectors)      │
│  • Content Search: TF-IDF Plot Vectorizer + Cosine Similarity          │
│  • Strict Constraint Engine (Hard genre include/exclude gating)        │
│  • Sample Confidence Estimator (High / Medium / Low for cohorts)       │
│  • Grounded Reasoner: Explains recommendations with historical proof    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
Ensure Python 3.10 or higher is installed:
```bash
git clone https://github.com/tunglamhe180046/movie-discovery-ai-assistant.git
cd movie-discovery-ai-assistant
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)
Create a `.env` file from `.env.example`:
```bash
copy .env.example .env   # Windows
# or cp .env.example .env # Linux/macOS
```
Add your **DeepSeek** or **Groq** API key:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
GROQ_API_KEY=your_groq_api_key
```
> **Note:** If no API key is provided or if network limits are encountered, the system **automatically and seamlessly switches to the deterministic offline engine** without crashing.

---

## 💻 Running the Application

### 1. Interactive Terminal CLI
Launch the assistant interactively:
```bash
python main.py
```
This displays an interactive account selector for **Users 0 to 4**:
* **User 0**: *Cold-Start Persona (0 ratings)*. Starts with a blank slate. You can add, remove, and reset preferences anytime!
* **User 1**: *Action & Adventure Fan (190 ratings, avg 4.33★)*. Missing: Documentary, IMAX.
* **User 2**: *Drama & Action Enthusiast (15 ratings, avg 4.00★)*.
* **User 3**: *Critical Sci-Fi Viewer (32 ratings, avg 2.08★)*.
* **User 4**: *Drama & Romance Lover (166 ratings, avg 3.60★)*.

Or launch directly with a specific user:
```bash
python main.py --user 0
```

#### Dynamic Preference Management (Add, Remove, Reset):
* **Add Taste:** *"Gu của tôi là thích phim Sci-Fi và Action"* or *"Thêm sở thích phim hoạt hình"*
* **Remove Taste:** *"Xóa sở thích Sci-Fi khỏi gu của tôi"* or *"Bỏ phim hành động"*
* **Dislike & Avoid:** *"Tôi ghét phim kinh dị Horror"* (Adds to avoidance list)
* **Remove Dislike:** *"Bỏ ghét phim kinh dị"* (Removes from avoidance list)
* **Reset / Clear All:** *"Xóa toàn bộ gu của tôi"* or *"Reset sở thích"* (Restores blank slate)
* **In-Session Switching:** Type `/switch <0-4>` to instantly swap users without restarting. Type `/profile` to view updated taste.

#### Natural Language Discovery Queries:
* *"What should I watch tonight? Something light, no animation."* (Constraints: Comedy/Drama, Exclude: Animation)
* *"What do people with similar taste to mine think about Pulp Fiction?"* (Cohort aggregation & sample confidence)
* *"Why do you think I'd like Inception?"* (Grounded explainability using rating history & cohort scores)
* *"What's my blind spot? What genres am I missing?"* (Statistical distribution contrast)
* *"Find me a sci-fi movie I haven't watched, no horror."* (Strict include Sci-Fi, exclude Horror)

### 2. Run Automated Test Suite
Run the 10 benchmark test cases:
```bash
python scripts/test_codex_cases.py
```
Output:
```text
============================================================
RUNNING 10 CODEX BENCHMARK TEST CASES
============================================================
[Case 1] Query: 'Cho tôi 1 tên phim lâu rồi tôi chưa xem...' -> [PASS]
[Case 2] Query: 'Tìm phim khoa học viễn tưởng... đừng kinh dị' -> [PASS]
...
============================================================
TEST SUMMARY: 10/10 Passed!
============================================================
```

### 3. Generate Benchmark JSON Report
Run full benchmark evaluation across multiple profiles:
```bash
python main.py --eval
```
Generates detailed metrics in `benchmark_evaluation.json`.

---

## 📊 Dataset Overview

Dataset located at `data/ml-latest-small-filtered/`:
* **`movies_with_plots.csv`**: 5,135 movies with synopsis plots (avg ~3,200 chars), genres, and release years.
* **`ratings.csv`**: 74,064 ratings from 610 users (range 0.5 - 5.0).
* **`tags.csv`**: 2,440 user-generated tags.

### Key Dataset Characteristic & Design Rationale
The sparsity in this filtered dataset is on the **movie side**, not the user side (~51% of movies have fewer than 5 ratings, but users average ~121 ratings).
* **Pure Collaborative Filtering** struggles on obscure, long-tail movies due to lack of peer overlap.
* **Our Solution:** A **Hybrid Recommender** that fuses Pearson Collaborative Filtering (60% weight) to identify cohort favorites with **TF-IDF Plot Semantic Search** (40% weight) to surface thematic hidden gems.

---

## 🔍 Key Technical Fixes & Improvements (P0/P1)

1. **Strict Genre Include Gating**: Resolved the candidate leakage issue where genre-specific queries (e.g. Sci-Fi) previously fell back to non-matching genres. Candidates are now strictly pre-filtered and backfilled with top-rated genre titles.
2. **Grounded Explainability Pipeline**: Added the `why_recommendation` intent, hooking [`explain_recommendation()`](engine.py) directly into the agent. Users asking *"Why would I like that?"* receive exact matching genres, personal rating comparison, and cohort rating breakdown.
3. **Cohort Sample Confidence Gate**: For cult films with sparse peer ratings (e.g. *Pulp Fiction* where only 2 peers rated it), the system explicitly flags `confidence: "low (sparse cohort sample)"` and prints community-wide averages (4.2★) alongside cohort scores to prevent small-sample bias.
4. **Dynamic User Profiling**: Removed hardcoded assumptions in fallback templates; rating counts, favorite genres, and timestamps are dynamically generated for all 610 users.

---

## 📁 Repository Structure

```
├── agent.py                   # Conversational Agent, Intent Parser & LLM Synthesizer
├── engine.py                  # Deterministic Recommendation & Analytics Engine
├── main.py                    # Interactive Rich Terminal CLI & Evaluation Runner
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and documentation
├── REPORT_TEMPLATE.md         # Comprehensive engineering report & failure analysis
├── PROBLEM.md                 # Original TrustedAI problem requirements
├── benchmark_evaluation.json  # Exported evaluation results
├── scripts/
│   ├── test_codex_cases.py    # 10 benchmark verification test cases
│   ├── test_groq_connection.py # API latency & connectivity verification
│   └── verify_dataset.py      # Dataset integrity verification script
└── data/                      # Filtered MovieLens dataset
```

---

## 👤 Author
* **Candidate:** Tùng Lâm (AI Engineer)
* **Target:** TrustedAI - AI Engineer Take-Home Assessment
* **Evaluation Date:** October 2026
