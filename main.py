"""Movie Discovery AI Assistant - Terminal CLI
Interactive terminal interface powered by Rich and MovieAgent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from agent import MovieAgent

console = Console()


def display_welcome_banner(user_id: int, profile: dict) -> None:
    """Render an attractive user taste card using Rich."""
    console.print()
    console.rule("[bold cyan]🎬 TrustedAI - Movie Discovery AI Assistant[/bold cyan]")
    console.print()

    # User taste summary table
    table = Table(title=f"User #{user_id} Taste Profile", show_header=True, header_style="bold magenta")
    table.add_column("Attribute", style="dim", width=22)
    table.add_column("Details", style="cyan")

    num_ratings = profile.get("num_ratings", 0)
    if num_ratings == 0:
        table.add_row("Ratings Count", "0 (Người dùng mới / Cold-Start)")
        table.add_row("Average Rating", "Chưa có đánh giá")
        favs = ", ".join([g["genre"] for g in profile.get("top_genres", [])])
        table.add_row("Gu đã lưu", favs or "Chưa thiết lập (Hãy chat để tôi ghi nhớ!)")
        table.add_row("Thể loại tránh", ", ".join(profile.get("disliked_genres", [])) or "Không có")
        table.add_row("Ghi chú cá nhân", profile.get("notes") or "Chưa có")
    else:
        table.add_row("Ratings Count", str(num_ratings))
        table.add_row("Average Rating", f"{profile.get('avg_rating', 'N/A')} ★")
        top_genres_str = ", ".join([f"{g['genre']} ({g['count']})" for g in profile.get("top_genres", [])[:4]])
        table.add_row("Favorite Genres", top_genres_str or "None")
        top_movies_str = ", ".join([f"{m['title']} ({m['rating']}★)" for m in profile.get("top_movies", [])[:3]])
        table.add_row("Top Rated Movies", top_movies_str or "None")
        blind_spots_str = ", ".join([b["genre"] for b in profile.get("blind_spots", [])[:3]])
        table.add_row("Blind Spots (Missing)", blind_spots_str or "None")

    console.print(table)
    console.print(
        Panel.fit(
            "[bold green]Try sample queries:[/bold green]\n"
            "• [italic]What should I watch tonight?[/italic]\n"
            "• [italic]Gu của tôi là phim hành động và khoa học viễn tưởng, nhớ nhé[/italic] (Cập nhật gu)\n"
            "• [italic]Tôi vừa xem Inception và chấm 5 sao[/italic] (Lưu đánh giá mới)\n"
            "• [italic]What do people with similar taste to mine think about Pulp Fiction?[/italic]\n"
            "• [italic]I liked Toy Story but I'm tired of animated movies — what else?[/italic]\n"
            "• [italic]Why do you think I'd like Inception?[/italic] (Grounded explainability)\n"
            "• [yellow]exit[/yellow] or [yellow]quit[/yellow] to end session.",
            title="💡 Quick Tips",
            border_style="blue"
        )
    )
    console.print()


def run_interactive_cli(user_id: int, agent: MovieAgent) -> None:
    """Run interactive REPL loop."""
    profile = agent.engine.get_user_profile(user_id)
    display_welcome_banner(user_id, profile)

    history: list[dict] = []

    while True:
        try:
            user_input = Prompt.ask(f"[bold yellow]User #{user_id}[/bold yellow]")
            if not user_input or not user_input.strip():
                continue

            clean_input = user_input.strip()
            if clean_input.lower() in ("exit", "quit", "q"):
                console.print("[cyan]Goodbye! Happy movie watching! 🍿[/cyan]")
                break

            with console.status("[bold cyan]AI Agent is investigating the dataset...[/bold cyan]", spinner="dots"):
                res = agent.chat(user_id=user_id, user_message=clean_input, history=history)

            # Show parsed intent and execution trace
            parsed = res.get("parsed_params", {})
            if parsed:
                inc_str = f", inc={parsed.get('genres_include')}" if parsed.get('genres_include') else ""
                exc_str = f", exc={parsed.get('genres_exclude')}" if parsed.get('genres_exclude') else ""
                lim_str = f", limit={parsed.get('limit')}" if parsed.get('limit') else ""
                console.print(f"[dim]🎯 Intent: {parsed.get('intent')}{inc_str}{exc_str}{lim_str} | Brain: {res.get('model_used')}[/dim]")

            # Render answer
            answer_text = res.get("answer", "")
            console.print()
            console.print(Panel(Markdown(answer_text), title="🤖 AI Assistant", border_style="cyan"))
            console.print()

            # Record turn in short memory
            history.append({"role": "user", "content": clean_input})
            history.append({"role": "assistant", "content": answer_text})
            if len(history) > 6:
                history = history[-6:]

        except KeyboardInterrupt:
            console.print("\n[cyan]Session closed by user.[/cyan]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


def run_evaluation_suite(agent: MovieAgent, output_file: str = "benchmark_evaluation.json") -> None:
    """Run automated evaluation on standard test users and sample queries."""
    console.rule("[bold green]Running Evaluation Benchmark[/bold green]")

    test_cases = [
        {"user_id": 1, "query": "Cho tôi 1 tên phim lâu rồi tôi chưa xem hoặc xem ít nhất."},
        {"user_id": 1, "query": "Tìm một phim khoa học viễn tưởng tôi chưa xem trong 2 năm, đừng kinh dị."},
        {"user_id": 1, "query": "Tối nay xem gì nhẹ nhàng, không phải hoạt hình?"},
        {"user_id": 1, "query": "Trong các phim giống Inception, phim nào tôi đã xem lâu nhất?"},
        {"user_id": 1, "query": "Đừng gợi ý bất cứ phim hoạt hình nào; chọn phim hành động tôi bỏ quên."},
        {"user_id": 1, "query": "Có phim nào thuộc thể loại điểm mù mà tôi gần như chưa xem nhưng hợp với tôi không?"},
        {"user_id": 1, "query": "Những người có gu giống tôi nghĩ gì về Pulp Fiction?"},
        {"user_id": 1, "query": "Điểm mù của tôi là gì và cho tôi phim tiêu biểu để bắt đầu?"},
        {"user_id": 1, "query": "Gợi ý cho tôi 1 phim tâm lý đen tối có plot twist bất ngờ."},
        {"user_id": 15, "query": "What should I watch tonight?"},
        {"user_id": 30, "query": "What should I watch tonight?"},
    ]

    results = []
    for idx, tc in enumerate(test_cases, 1):
        uid = tc["user_id"]
        q = tc["query"]
        console.print(f"\n[bold]Test {idx}/{len(test_cases)}: User {uid} -> \"{q}\"[/bold]")
        with console.status("Querying agent...", spinner="line"):
            res = agent.chat(user_id=uid, user_message=q)
        
        tools_called = [t["name"] if isinstance(t, dict) else str(t) for t in res.get("tools_used", [])]
        console.print(f"Tools called: {tools_called}")
        console.print(Panel(Markdown(res.get("answer", "")[:350] + "..."), title=f"Result {idx}", border_style="green"))

        results.append({
            "test_id": idx,
            "user_id": uid,
            "query": q,
            "tools_called": tools_called,
            "answer": res.get("answer", ""),
            "model_used": res.get("model_used")
        })

    out_path = Path(__file__).resolve().parent / output_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    console.print(f"\n[bold green]Evaluation benchmark complete! Output saved to: {out_path.name}[/bold green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="TrustedAI Movie Discovery Assistant")
    parser.add_argument("--user", type=int, default=1, help="User ID (e.g. 1, 15, 30)")
    parser.add_argument("--eval", action="store_true", help="Run automated evaluation benchmark and exit")
    args = parser.parse_args()

    agent = MovieAgent()

    if args.eval:
        run_evaluation_suite(agent)
    else:
        run_interactive_cli(user_id=args.user, agent=agent)


if __name__ == "__main__":
    main()
