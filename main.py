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


AVAILABLE_USERS = [0, 1, 2, 3, 4]


def select_user_prompt(engine) -> int:
    """Display available users (0 to 4) and prompt user to choose one."""
    console.print()
    console.rule("[bold cyan]👥 Danh Sách Tài Khoản Người Dùng Khả Dụng (User 0 - 4)[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("User ID", style="bold yellow", width=10, justify="center")
    table.add_column("Loại Hồ Sơ", style="bold cyan", width=26)
    table.add_column("Đặc Điểm Gu Phim & Dữ Liệu", style="white")

    descriptions = {
        0: ("Người dùng mới (Cold-Start)", "0 đánh giá. Chưa có gu ban đầu — Tùy ý thêm, xóa và đổi gu qua chat!"),
        1: ("Fan Hành Động & Phiêu Lưu", "190 đánh giá (4.33★). Thích: Action, Adventure, Comedy. Điểm mù: IMAX, Documentary."),
        2: ("Người xem Tâm Lý & Kịch Tính", "15 đánh giá (4.00★). Thích: Drama, Action, Comedy."),
        3: ("Khán Giả Viễn Tưởng Khắt Khe", "32 đánh giá (2.08★). Thích: Sci-Fi, Action, Thriller."),
        4: ("Tín Đồ Tâm Lý & Lãng Mạn", "166 đánh giá (3.60★). Thích: Drama, Comedy, Romance."),
    }

    for uid in AVAILABLE_USERS:
        title_str, desc_str = descriptions[uid]
        table.add_row(f"User #{uid}", title_str, desc_str)

    console.print(table)
    console.print("[dim]Nhập số [0, 1, 2, 3, 4] để vào phòng chat, hoặc gõ trực tiếp --user <id> khi khởi chạy.[/dim]\n")

    while True:
        choice = Prompt.ask("[bold green]Chọn User ID để bắt đầu[/bold green]", choices=["0", "1", "2", "3", "4"], default="0")
        try:
            return int(choice)
        except ValueError:
            continue


def display_welcome_banner(user_id: int, profile: dict) -> None:
    """Render an attractive user taste card using Rich."""
    console.print()
    console.rule(f"[bold cyan]🎬 TrustedAI - Movie Discovery AI Assistant | Active: User #{user_id}[/bold cyan]")
    console.print()

    # User taste summary table
    table = Table(title=f"User #{user_id} Taste Profile", show_header=True, header_style="bold magenta")
    table.add_column("Attribute", style="dim", width=22)
    table.add_column("Details", style="cyan")

    num_ratings = profile.get("num_ratings", 0)
    favs = ", ".join([g["genre"] for g in profile.get("top_genres", [])])
    dislikes = ", ".join(profile.get("disliked_genres", []))
    notes = profile.get("notes", "")

    if num_ratings == 0:
        table.add_row("Ratings Count", "0 (Người dùng mới / Cold-Start)")
        table.add_row("Average Rating", "Chưa có đánh giá")
        table.add_row("Gu yêu thích đã lưu", favs or "Chưa thiết lập (Hãy chat để thêm!)")
        table.add_row("Thể loại tránh", dislikes or "Không có")
        table.add_row("Ghi chú cá nhân", notes or "Chưa có")
    else:
        table.add_row("Ratings Count", str(num_ratings))
        table.add_row("Average Rating", f"{profile.get('avg_rating', 'N/A')} ★")
        top_genres = profile.get("top_genres", [])
        if any(g.get("custom") for g in top_genres):
            top_genres_str = ", ".join([f"{g['genre']} (Tùy chỉnh)" for g in top_genres])
            table.add_row("Favorite Genres (Tự đặt)", top_genres_str or "None")
        else:
            top_genres_str = ", ".join([f"{g['genre']} ({g['count']})" for g in top_genres[:5]])
            table.add_row("Favorite Genres", top_genres_str or "None")
        top_movies_str = ", ".join([f"{m['title']} ({m['rating']}★)" for m in profile.get("top_movies", [])[:3]])
        table.add_row("Top Rated Movies", top_movies_str or "None")
        blind_spots_str = ", ".join([b["genre"] for b in profile.get("blind_spots", [])[:3]])
        table.add_row("Blind Spots (Missing)", blind_spots_str or "None")
        if dislikes:
            table.add_row("Thể loại tránh (Tự đặt)", dislikes)

    console.print(table)
    console.print(
        Panel.fit(
            "[bold green]Gợi ý tương tác & Quản lý sở thích:[/bold green]\n"
            "• [italic]Thêm sở thích phim hành động và khoa học viễn tưởng[/italic] (Thêm gu yêu thích)\n"
            "• [italic]Xóa sở thích khoa học viễn tưởng[/italic] (Xóa gu khỏi danh sách thích)\n"
            "• [italic]Tôi ghét phim kinh dị[/italic] (Thêm thể loại tránh gợi ý)\n"
            "• [italic]Bỏ ghét phim kinh dị[/italic] (Xóa khỏi danh sách tránh)\n"
            "• [italic]Xóa toàn bộ gu / Reset sở thích[/italic] (Xóa hết gu đã lưu để làm lại)\n"
            "• [italic]Tôi vừa xem Inception và chấm 5 sao[/italic] (Lưu đánh giá mới)\n"
            "• [italic]Tối nay xem gì hợp gu tôi?[/italic] | [italic]Why do you think I'd like Inception?[/italic]\n"
            "• Lệnh nhanh: [yellow]/switch <0-4>[/yellow] (Đổi user), [yellow]/profile[/yellow] (Xem lại gu), [yellow]exit[/yellow] (Thoát).",
            title="💡 Hướng Dẫn & Lệnh Nhanh",
            border_style="blue"
        )
    )
    console.print()


def run_interactive_cli(user_id: int, agent: MovieAgent) -> None:
    """Run interactive REPL loop."""
    current_uid = user_id
    profile = agent.engine.get_user_profile(current_uid)
    display_welcome_banner(current_uid, profile)

    history: list[dict] = []

    while True:
        try:
            user_input = Prompt.ask(f"[bold yellow]User #{current_uid}[/bold yellow]")
            if not user_input or not user_input.strip():
                continue

            clean_input = user_input.strip()
            lower_input = clean_input.lower()

            if lower_input in ("exit", "quit", "q"):
                console.print("[cyan]Goodbye! Happy movie watching! 🍿[/cyan]")
                break

            # Handle shortcut slash commands
            if lower_input.startswith("/switch ") or lower_input.startswith("/user "):
                parts = clean_input.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    new_uid = int(parts[1])
                    if new_uid in AVAILABLE_USERS:
                        current_uid = new_uid
                        history.clear()
                        console.print(f"[bold green]Đã chuyển sang tài khoản User #{current_uid}![/bold green]")
                        profile = agent.engine.get_user_profile(current_uid)
                        display_welcome_banner(current_uid, profile)
                        continue
                    else:
                        console.print(f"[bold red]Chỉ hỗ trợ User từ 0 đến 4! Hãy chọn một trong: {AVAILABLE_USERS}[/bold red]")
                        continue

            if lower_input in ("/profile", "/p", "/me"):
                profile = agent.engine.get_user_profile(current_uid)
                display_welcome_banner(current_uid, profile)
                continue

            if lower_input in ("/users", "/list"):
                select_user_prompt(agent.engine)
                continue

            with console.status("[bold cyan]AI Agent is investigating the dataset...[/bold cyan]", spinner="dots"):
                res = agent.chat(user_id=current_uid, user_message=clean_input, history=history)

            # Show parsed intent and execution trace
            parsed = res.get("parsed_params", {})
            if parsed:
                inc_str = f", inc={parsed.get('genres_include')}" if parsed.get('genres_include') else ""
                exc_str = f", exc={parsed.get('genres_exclude')}" if parsed.get('genres_exclude') else ""
                rem_inc_str = f", rem_inc={parsed.get('remove_genres_include')}" if parsed.get('remove_genres_include') else ""
                rem_exc_str = f", rem_exc={parsed.get('remove_genres_exclude')}" if parsed.get('remove_genres_exclude') else ""
                lim_str = f", limit={parsed.get('limit')}" if parsed.get('limit') else ""
                console.print(f"[dim]🎯 Intent: {parsed.get('intent')}{inc_str}{exc_str}{rem_inc_str}{rem_exc_str}{lim_str} | Brain: {res.get('model_used')}[/dim]")

            # Render answer
            answer_text = res.get("answer", "")
            console.print()
            console.print(Panel(Markdown(answer_text), title=f"🤖 AI Assistant (for User #{current_uid})", border_style="cyan"))
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
    parser = argparse.ArgumentParser(description="TrustedAI Movie Discovery Assistant (Users 0 - 4)")
    parser.add_argument("--user", type=int, choices=AVAILABLE_USERS, default=None, help="User ID (0: New User, 1: Action/Adventure, 2: Drama/Action, 3: Sci-Fi, 4: Romance)")
    parser.add_argument("--eval", action="store_true", help="Run automated evaluation benchmark and exit")
    args = parser.parse_args()

    agent = MovieAgent()

    if args.eval:
        run_evaluation_suite(agent)
    else:
        user_id = args.user if args.user is not None else select_user_prompt(agent.engine)
        run_interactive_cli(user_id=user_id, agent=agent)


if __name__ == "__main__":
    main()
