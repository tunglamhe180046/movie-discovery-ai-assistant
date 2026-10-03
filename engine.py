"""MovieDataEngine: data loading, user profiling, and recommendation logic
for the MovieLens ml-latest-small-filtered dataset.
Optimized and hardened with review feedback from Codex.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small-filtered"
POPULAR_GENRE_MIN_COUNT = 20

ALL_CANONICAL_GENRES: dict[str, str] = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "children": "Children",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "film-noir": "Film-Noir",
    "noir": "Film-Noir",
    "horror": "Horror",
    "musical": "Musical",
    "mystery": "Mystery",
    "romance": "Romance",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "thriller": "Thriller",
    "war": "War",
    "western": "Western",
    "imax": "IMAX",
}


class MovieDataEngine:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)

        # 1. Load CSVs
        self.movies = pd.read_csv(self.data_dir / "movies_with_plots.csv")
        self.ratings = pd.read_csv(self.data_dir / "ratings.csv")
        self.tags = pd.read_csv(self.data_dir / "tags.csv")

        # Fill NaNs
        self.movies["plot"] = self.movies["plot"].fillna("")
        self.movies["genres"] = self.movies["genres"].fillna("")
        self.movies["genre_list"] = self.movies["genres"].apply(
            lambda g: [x for x in g.split("|") if x and x != "(no genres listed)"]
        )

        # Fast lookup indices
        self._movies_by_id = self.movies.set_index("movieId", drop=False)

        # Precompute movie stats (popularity & average rating) for cold-start fallback & ranking
        stats = self.ratings.groupby("movieId").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()
        self.movie_stats = self.movies.merge(stats, on="movieId", how="left")
        self.movie_stats["rating_count"] = self.movie_stats["rating_count"].fillna(0).astype(int)
        self.movie_stats["avg_rating"] = self.movie_stats["avg_rating"].fillna(0.0)

        # Precompute cached global genre counts (Codex recommendation: avoid repeated merge/explode)
        merged_genres = self.ratings.merge(self.movies[["movieId", "genre_list"]], on="movieId")
        self._cached_global_genre_counts = merged_genres.explode("genre_list")["genre_list"].value_counts()

        # Dynamic User Memory (Personalization notes, custom ratings, and stated preferences)
        self.user_memory_path = Path(__file__).resolve().parent / "data" / "user_memory.json"
        self.user_memory: dict[str, dict] = self._load_user_memory()
        self._inject_custom_memory_ratings()

        # Build user-item matrix and TF-IDF index
        self._user_item_matrix: pd.DataFrame | None = None
        self._tfidf_vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None

        self._build_user_item_matrix()
        self._build_tfidf_index()

    def _canonicalize_genre(self, g: str) -> str:
        """Standardize raw or colloquial genre names to canonical MovieLens casing."""
        if not g:
            return ""
        clean = str(g).strip().lower()
        return ALL_CANONICAL_GENRES.get(clean, str(g).strip().title())

    def _get_historical_top_genres(self, user_id: int, top_n: int = 5) -> list[str]:
        """Extract baseline top genres from historical rating CSVs."""
        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            return []
        rated = user_ratings.merge(self.movies[["movieId", "genre_list"]], on="movieId")
        liked = rated[rated["rating"] >= 4.0]
        counts = (liked if not liked.empty else rated).explode("genre_list")["genre_list"].value_counts()
        return [str(g) for g in counts.head(top_n).index if g]

    def _load_user_memory(self) -> dict[str, dict]:
        """Load persistent working memory from disk safely."""
        if self.user_memory_path.exists():
            try:
                with open(self.user_memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                return {}
        return {}

    def _save_user_memory(self) -> None:
        """Persist working memory to disk atomically to prevent empty 0-byte file corruption."""
        try:
            self.user_memory_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.user_memory_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.user_memory, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.user_memory_path)
        except Exception:
            try:
                with open(self.user_memory_path, "w", encoding="utf-8") as f:
                    json.dump(self.user_memory, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def _inject_custom_memory_ratings(self) -> None:
        """Inject user ratings saved in memory into the active ratings DataFrame without duplicates."""
        rows_to_add = []
        for uid_str, udata in self.user_memory.items():
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            for cr in udata.get("custom_ratings", []):
                mid = int(cr["movieId"])
                # Avoid duplicate accumulation across restarts: drop existing rating row if present
                self.ratings = self.ratings[~((self.ratings["userId"] == uid) & (self.ratings["movieId"] == mid))]
                rows_to_add.append({
                    "userId": uid,
                    "movieId": mid,
                    "rating": float(cr["rating"]),
                    "timestamp": cr.get("timestamp", int(pd.Timestamp.now().timestamp()))
                })
        if rows_to_add:
            self.ratings = pd.concat([self.ratings, pd.DataFrame(rows_to_add)], ignore_index=True)

    def record_user_preference(
        self,
        user_id: int,
        set_favorite_genres: list[str] | None = None,
        favorite_genres: list[str] | None = None,
        disliked_genres: list[str] | None = None,
        remove_favorite_genres: list[str] | None = None,
        remove_disliked_genres: list[str] | None = None,
        clear_all: bool = False,
        notes: str | None = None,
    ) -> dict:
        """Dynamically record, add, remove, or clear user's stated taste in persistent memory."""
        uid_str = str(user_id)
        if uid_str not in self.user_memory:
            self.user_memory[uid_str] = {
                "favorite_genres": [],
                "disliked_genres": [],
                "custom_ratings": [],
                "has_explicit_preference": False,
                "notes": ""
            }

        mem = self.user_memory[uid_str]

        # Reset / Clear all preferences
        if clear_all:
            mem["favorite_genres"] = []
            mem["disliked_genres"] = []
            mem["has_explicit_preference"] = False
            mem["notes"] = ""
            self._save_user_memory()
            return {
                "user_id": user_id,
                "status": "cleared",
                "message": "Đã xóa toàn bộ sở thích và ghi chú cá nhân của bạn.",
                "favorite_genres": [],
                "disliked_genres": [],
                "notes": ""
            }

        # Case 1: Explicitly overwrite / set entire favorite list
        if set_favorite_genres is not None:
            clean_set = [self._canonicalize_genre(g) for g in set_favorite_genres if g]
            seen = set()
            mem["favorite_genres"] = [g for g in clean_set if not (g.lower() in seen or seen.add(g.lower()))]
            mem["has_explicit_preference"] = True

        # Case 2: Removals from favorites
        removed_favs = []
        if remove_favorite_genres:
            # Seed from historical baseline if user hasn't customized yet
            if not mem.get("has_explicit_preference") and not mem.get("favorite_genres"):
                mem["favorite_genres"] = self._get_historical_top_genres(user_id)

            rem_f_lower = {g.lower() for g in remove_favorite_genres}
            curr_favs = mem.get("favorite_genres", [])
            new_favs = [g for g in curr_favs if g.lower() not in rem_f_lower]
            removed_favs = [g for g in curr_favs if g.lower() in rem_f_lower]
            mem["favorite_genres"] = new_favs
            mem["has_explicit_preference"] = True

        # Case 3: Removals from dislikes
        removed_dislikes = []
        if remove_disliked_genres:
            rem_d_lower = {g.lower() for g in remove_disliked_genres}
            curr_dislikes = mem.get("disliked_genres", [])
            new_dislikes = [g for g in curr_dislikes if g.lower() not in rem_d_lower]
            removed_dislikes = [g for g in curr_dislikes if g.lower() in rem_d_lower]
            mem["disliked_genres"] = new_dislikes

        # Case 4: Additions to favorites
        added_favs = []
        if favorite_genres:
            if not mem.get("has_explicit_preference") and not mem.get("favorite_genres"):
                mem["favorite_genres"] = self._get_historical_top_genres(user_id)

            existing = {g.lower() for g in mem.get("favorite_genres", [])}
            for g in favorite_genres:
                canon = self._canonicalize_genre(g)
                if canon and canon.lower() not in existing:
                    mem["favorite_genres"].append(canon)
                    added_favs.append(canon)
                    existing.add(canon.lower())
            mem["has_explicit_preference"] = True

        # Case 5: Additions to dislikes (avoidance)
        added_dislikes = []
        if disliked_genres:
            existing_d = {g.lower() for g in mem.get("disliked_genres", [])}
            for g in disliked_genres:
                canon = self._canonicalize_genre(g)
                if canon and canon.lower() not in existing_d:
                    mem["disliked_genres"].append(canon)
                    added_dislikes.append(canon)
                    existing_d.add(canon.lower())
                # Also remove from favorites if user now dislikes it
                mem["favorite_genres"] = [f for f in mem.get("favorite_genres", []) if f.lower() != canon.lower()]

        if notes is not None:
            mem["notes"] = notes

        self._save_user_memory()
        return {
            "user_id": user_id,
            "status": "updated",
            "added_favorites": added_favs,
            "removed_favorites": removed_favs,
            "added_dislikes": added_dislikes,
            "removed_dislikes": removed_dislikes,
            "favorite_genres": mem.get("favorite_genres", []),
            "disliked_genres": mem.get("disliked_genres", []),
            "notes": mem.get("notes", "")
        }

    def add_user_rating(self, user_id: int, movie_title_or_id: str | int, rating: float) -> dict:
        """Record a live movie rating dynamically into persistent memory and live DataFrame."""
        movie_id = self._resolve_movie_id(movie_title_or_id)
        if movie_id is None:
            return {"error": f"Không tìm thấy phim '{movie_title_or_id}' trong hệ thống."}

        movie_row = self._get_movie_row(movie_id)
        rating_val = max(0.5, min(5.0, float(rating)))

        uid_str = str(user_id)
        if uid_str not in self.user_memory:
            self.user_memory[uid_str] = {
                "favorite_genres": [],
                "disliked_genres": [],
                "custom_ratings": [],
                "notes": ""
            }

        # Deduplicate and append
        self.user_memory[uid_str]["custom_ratings"] = [
            r for r in self.user_memory[uid_str].get("custom_ratings", []) if r.get("movieId") != movie_id
        ]
        self.user_memory[uid_str]["custom_ratings"].append({
            "movieId": int(movie_id),
            "title": movie_row["title"],
            "rating": rating_val,
            "genres": movie_row["genres"],
            "timestamp": int(pd.Timestamp.now().timestamp())
        })
        self._save_user_memory()

        # Append to live self.ratings DataFrame & re-pivot
        new_row = pd.DataFrame([{
            "userId": user_id,
            "movieId": movie_id,
            "rating": rating_val,
            "timestamp": int(pd.Timestamp.now().timestamp())
        }])
        self.ratings = pd.concat([self.ratings, new_row], ignore_index=True)
        self._build_user_item_matrix()

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "title": movie_row["title"],
            "rating": rating_val,
            "status": "rating_saved"
        }

    def _build_user_item_matrix(self) -> None:
        self._user_item_matrix = self.ratings.pivot_table(
            index="userId", columns="movieId", values="rating"
        )

    def _build_tfidf_index(self) -> None:
        self._tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=25000)
        corpus = (self.movies["plot"] + " " + self.movies["genres"].str.replace("|", " ", regex=False))
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(corpus)

    def _get_movie_row(self, movie_id: int) -> pd.Series | None:
        if movie_id in self._movies_by_id.index:
            row = self._movies_by_id.loc[movie_id]
            # Handle potential duplicate index defensively
            if isinstance(row, pd.DataFrame):
                return row.iloc[0]
            return row
        return None

    def _resolve_movie_id(self, movie_title_or_id: Any) -> int | None:
        if isinstance(movie_title_or_id, (int, np.integer)):
            return int(movie_title_or_id) if movie_title_or_id in self._movies_by_id.index else None
        
        query_str = str(movie_title_or_id).strip()
        if not query_str:
            return None

        # Check numeric string
        try:
            val = int(query_str)
            if val in self._movies_by_id.index:
                return val
        except ValueError:
            pass

        query_lower = query_str.lower()

        # 1. Exact match (case-insensitive)
        exact = self.movies[self.movies["title"].str.lower() == query_lower]
        if not exact.empty:
            return int(exact.iloc[0]["movieId"])

        # 2. Exact match without year (e.g. "Inception" matches "Inception (2010)")
        exact_no_year = self.movies[
            self.movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.lower() == query_lower
        ]
        if not exact_no_year.empty:
            return int(exact_no_year.iloc[0]["movieId"])

        # 3. Substring match (regex=False to avoid special regex character crashes)
        matches = self.movies[self.movies["title"].str.contains(query_str, case=False, na=False, regex=False)]
        if not matches.empty:
            # Sort by rating count to return the most notable movie
            notable = matches.merge(self.movie_stats[["movieId", "rating_count"]], on="movieId")
            notable = notable.sort_values("rating_count", ascending=False)
            return int(notable.iloc[0]["movieId"])

        return None

    # ------------------------------------------------------------------
    # User Profile & Blind Spots
    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: int) -> dict:
        uid_str = str(user_id)
        custom_mem = self.user_memory.get(uid_str, {})
        custom_favs = custom_mem.get("favorite_genres", [])
        custom_dislikes = custom_mem.get("disliked_genres", [])

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            return {
                "user_id": user_id,
                "num_ratings": 0,
                "avg_rating": None,
                "top_genres": [{"genre": g, "count": 1} for g in custom_favs],
                "disliked_genres": custom_dislikes,
                "top_movies": [],
                "blind_spots": [],
                "has_custom_profile": bool(custom_favs or custom_dislikes),
                "notes": custom_mem.get("notes", ""),
            }

        num_ratings = len(user_ratings)
        avg_rating = round(float(user_ratings["rating"].mean()), 2)

        rated = user_ratings.merge(self.movies[["movieId", "title", "genre_list"]], on="movieId")

        # Top genres based on ratings >= 4.0
        liked = rated[rated["rating"] >= 4.0]
        if not liked.empty:
            genre_counts = liked.explode("genre_list")["genre_list"].value_counts()
        else:
            genre_counts = rated.explode("genre_list")["genre_list"].value_counts()

        has_explicit = custom_mem.get("has_explicit_preference", False)
        if has_explicit and custom_favs:
            # User explicitly declared or customized their active favorite genres
            top_genres = [
                {"genre": g, "count": int(genre_counts.get(g, 1)), "custom": True}
                for g in custom_favs
            ]
        else:
            dislike_lower = {d.lower() for d in custom_dislikes}
            top_genres = [
                {"genre": genre, "count": int(count)}
                for genre, count in genre_counts.head(5).items()
                if genre and genre.lower() not in dislike_lower
            ]

        # Top rated movies the user has watched
        top_movies_df = rated.sort_values(["rating", "title"], ascending=[False, True]).head(5)
        top_movies = [
            {"movieId": int(r.movieId), "title": r.title, "rating": float(r.rating)}
            for r in top_movies_df.itertuples()
        ]

        # Blind spots using pre-cached global counts
        user_genre_counts = rated.explode("genre_list")["genre_list"].value_counts()
        global_counts = self._cached_global_genre_counts[self._cached_global_genre_counts >= POPULAR_GENRE_MIN_COUNT]

        blind_spots = []
        for genre, global_count in global_counts.sort_values(ascending=False).items():
            if not genre:
                continue
            user_count = int(user_genre_counts.get(genre, 0))
            exposure_ratio = user_count / num_ratings
            if user_count == 0 or exposure_ratio < 0.05:
                blind_spots.append(
                    {
                        "genre": genre,
                        "user_watch_count": user_count,
                        "global_rating_count": int(global_count),
                    }
                )
        blind_spots = blind_spots[:5]

        notes = custom_mem.get("notes", "")
        return {
            "user_id": user_id,
            "num_ratings": num_ratings,
            "avg_rating": avg_rating,
            "top_genres": top_genres,
            "disliked_genres": custom_dislikes,
            "top_movies": top_movies,
            "blind_spots": blind_spots,
            "has_custom_profile": bool(has_explicit or custom_favs or custom_dislikes or notes),
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Collaborative Filtering: Similar Users
    # ------------------------------------------------------------------

    def get_similar_users(self, user_id: int, top_n: int = 10, min_common_ratings: int = 3) -> list[dict]:
        if self._user_item_matrix is None or user_id not in self._user_item_matrix.index:
            return []

        target = self._user_item_matrix.loc[user_id]
        target_rated_mask = target.notna()

        results = []
        for other_id, other_row in self._user_item_matrix.iterrows():
            if other_id == user_id:
                continue

            common_mask = target_rated_mask & other_row.notna()
            n_common = int(common_mask.sum())
            if n_common < min_common_ratings:
                continue

            a = target[common_mask].to_numpy(dtype=float)
            b = other_row[common_mask].to_numpy(dtype=float)

            if np.std(a) == 0 or np.std(b) == 0:
                similarity = 1.0 if np.allclose(a, b) else 0.0
            else:
                similarity = float(np.corrcoef(a, b)[0, 1])

            if np.isnan(similarity):
                continue

            # Codex review: Filter out negative or zero similarity to prevent inverse recommendation
            if similarity > 0:
                results.append(
                    {
                        "userId": int(other_id),
                        "similarity": round(similarity, 4),
                        "common_ratings": n_common,
                    }
                )

        results.sort(key=lambda r: (r["similarity"], r["common_ratings"]), reverse=True)
        return results[:max(1, top_n)]

    # ------------------------------------------------------------------
    # Cohort Ratings for a Specific Movie
    # ------------------------------------------------------------------

    def get_cohort_ratings(self, movie_title_or_id: Any, user_ids: list[int]) -> dict:
        movie_id = self._resolve_movie_id(movie_title_or_id)
        if movie_id is None:
            return {"movie_id": None, "title": None, "ratings": [], "avg_rating": None, "num_ratings": 0}

        movie_row = self._get_movie_row(movie_id)
        subset = self.ratings[(self.ratings["movieId"] == movie_id) & (self.ratings["userId"].isin(user_ids))]

        ratings_list = [
            {"userId": int(r.userId), "rating": float(r.rating)}
            for r in subset.itertuples()
        ]

        # P0 FIX: Add confidence estimation & global comparison to prevent misleading small-sample bias (e.g. n=1 for Pulp Fiction)
        global_stats = self.movie_stats[self.movie_stats["movieId"] == movie_id]
        global_avg = round(float(global_stats["avg_rating"].values[0]), 2) if not global_stats.empty else None
        global_count = int(global_stats["rating_count"].values[0]) if not global_stats.empty else 0

        num_cohort = len(ratings_list)
        if num_cohort >= 5:
            confidence = "high"
        elif num_cohort >= 3:
            confidence = "medium"
        else:
            confidence = "low (sparse cohort sample)"

        return {
            "movie_id": movie_id,
            "title": movie_row["title"] if movie_row is not None else None,
            "ratings": ratings_list,
            "avg_rating": round(float(subset["rating"].mean()), 2) if not subset.empty else None,
            "num_ratings": num_cohort,
            "confidence": confidence,
            "global_avg_rating": global_avg,
            "global_rating_count": global_count,
        }

    # ------------------------------------------------------------------
    # Semantic Content Search (TF-IDF on Plot + Genres)
    # ------------------------------------------------------------------

    def search_by_plot(
        self,
        query: str,
        filter_genres: list[str] | None = None,
        exclude_genres: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        top_k = max(1, top_k)
        if not query or not query.strip():
            return []

        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).ravel()

        candidates = self.movies.copy()
        candidates["score"] = scores

        # Case-insensitive genre handling (Codex recommendation)
        if filter_genres:
            fg_lower = [g.lower() for g in filter_genres]
            candidates = candidates[
                candidates["genre_list"].apply(lambda gl: any(g.lower() in fg_lower for g in gl))
            ]
        if exclude_genres:
            eg_lower = [g.lower() for g in exclude_genres]
            candidates = candidates[
                candidates["genre_list"].apply(lambda gl: not any(g.lower() in eg_lower for g in gl))
            ]

        candidates = candidates[candidates["score"] > 0].sort_values("score", ascending=False).head(top_k)

        return [
            {
                "movieId": int(row.movieId),
                "title": row.title,
                "genres": row.genres,
                "score": round(float(row.score), 4),
            }
            for row in candidates.itertuples()
        ]

    # ------------------------------------------------------------------
    # Hybrid Recommendation Engine with Cold-Start Fallback
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: int,
        query: str | None = None,
        exclude_genres: list[str] | None = None,
        include_genres: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Hybrid personalized recommendation combining Collaborative Filtering (60%) and Content TF-IDF (40%).
        
        P0 FIX: Added strict `include_genres` constraint enforcement.
        Candidates are pre-filtered so that when a user asks for Sci-Fi or Comedy, non-matching movies
        (e.g., Antonia's Line) are never returned.
        """
        top_k = max(1, top_k)
        watched_ids = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])

        # Normalize exclude_genres & include_genres for case-insensitive matching
        eg_lower = [g.lower() for g in exclude_genres] if exclude_genres else []
        ig_lower = [g.lower() for g in include_genres] if include_genres else []

        # Check user memory for dynamic preferences (Assistant Memory Feature)
        uid_str = str(user_id)
        custom_mem = self.user_memory.get(uid_str, {})
        mem_dislikes = custom_mem.get("disliked_genres", [])
        mem_favs = custom_mem.get("favorite_genres", [])
        if mem_dislikes:
            eg_lower = list(set(eg_lower + [g.lower() for g in mem_dislikes]))
        if not ig_lower and not query and mem_favs:
            include_genres = mem_favs
            ig_lower = [g.lower() for g in mem_favs]

        similar_users = self.get_similar_users(user_id, top_n=25, min_common_ratings=3)
        similar_user_ids = [u["userId"] for u in similar_users]
        similarity_by_user = {u["userId"]: u["similarity"] for u in similar_users}

        # 1. Collaborative Filtering scores (Pearson correlation weighted)
        cf_scores: dict[int, float] = {}
        if similar_user_ids:
            cohort_ratings = self.ratings[
                self.ratings["userId"].isin(similar_user_ids) & ~self.ratings["movieId"].isin(watched_ids)
            ]
            for movie_id, group in cohort_ratings.groupby("movieId"):
                # Only use positive similarity
                weights = group["userId"].map(similarity_by_user).fillna(0.1)
                if weights.sum() > 0:
                    weighted_avg = float(np.average(group["rating"], weights=weights))
                    # Quality gate: only recommend movies with average >= 3.5 from cohort
                    if weighted_avg >= 3.5:
                        cf_scores[int(movie_id)] = weighted_avg

        # 2. Content-based scores (TF-IDF plot + genre similarity)
        content_scores: dict[int, float] = {}
        if query and query.strip():
            hits = self.search_by_plot(query, filter_genres=include_genres, exclude_genres=exclude_genres, top_k=50)
            content_scores = {h["movieId"]: h["score"] for h in hits if h["movieId"] not in watched_ids}
        else:
            profile = self.get_user_profile(user_id)
            # Only use movies rated >= 3.5 for building user taste vector
            user_ratings = self.ratings[self.ratings["userId"] == user_id]
            liked_ratings = user_ratings[user_ratings["rating"] >= 3.5]
            liked_movie_ids = list(liked_ratings["movieId"])

            if liked_movie_ids:
                liked_idx = self.movies[self.movies["movieId"].isin(liked_movie_ids)].index
                if len(liked_idx) > 0:
                    profile_vec = self._tfidf_matrix[liked_idx].mean(axis=0)
                    profile_vec = np.asarray(profile_vec)
                    sims = cosine_similarity(profile_vec, self._tfidf_matrix).ravel()
                    for mid, score in zip(self.movies["movieId"], sims):
                        if mid not in watched_ids and score > 0.05:
                            content_scores[int(mid)] = float(score)

        candidate_ids = (set(cf_scores) | set(content_scores)) - watched_ids

        # Filter candidates strictly by include_genres if requested
        if ig_lower:
            valid_candidates = set()
            for mid in candidate_ids:
                row = self._get_movie_row(mid)
                if row is not None and any(g.lower() in ig_lower for g in row["genre_list"]):
                    valid_candidates.add(mid)
            candidate_ids = valid_candidates

            # If candidates are fewer than top_k, fetch popular unwatched movies in these genres
            if len(candidate_ids) < top_k:
                genre_candidates = self.movie_stats[
                    (~self.movie_stats["movieId"].isin(watched_ids)) &
                    (self.movie_stats["genre_list"].apply(lambda gl: any(g.lower() in ig_lower for g in gl)))
                ]
                if eg_lower:
                    genre_candidates = genre_candidates[
                        genre_candidates["genre_list"].apply(lambda gl: not any(g.lower() in eg_lower for g in gl))
                    ]
                top_genre_mids = genre_candidates.sort_values(["avg_rating", "rating_count"], ascending=False)["movieId"].head(top_k * 2)
                for mid in top_genre_mids:
                    candidate_ids.add(int(mid))

        # COLD-START FALLBACK: If user has no CF candidates and no query hits
        if not candidate_ids:
            popular = self.movie_stats[
                (~self.movie_stats["movieId"].isin(watched_ids)) &
                (self.movie_stats["rating_count"] >= 10)
            ].sort_values(["avg_rating", "rating_count"], ascending=False)

            if eg_lower:
                popular = popular[
                    popular["genre_list"].apply(lambda gl: not any(g.lower() in eg_lower for g in gl))
                ]
            if ig_lower:
                popular = popular[
                    popular["genre_list"].apply(lambda gl: any(g.lower() in ig_lower for g in gl))
                ]
            
            top_fallback = popular.head(top_k)
            return [
                {
                    "movieId": int(r.movieId),
                    "title": r.title,
                    "genres": r.genres,
                    "score": round(float(r.avg_rating), 2),
                    "cf_score": None,
                    "content_score": None,
                    "is_fallback": True,
                }
                for r in top_fallback.itertuples()
            ]

        def normalize(d: dict[int, float]) -> dict[int, float]:
            if not d:
                return {}
            values = np.array(list(d.values()))
            lo, hi = values.min(), values.max()
            if hi == lo:
                return {k: 1.0 for k in d}
            return {k: (v - lo) / (hi - lo) for k, v in d.items()}

        cf_norm = normalize(cf_scores)
        content_norm = normalize(content_scores)

        cf_weight = 0.6 if cf_scores else 0.0
        content_weight = 0.4 if content_scores else 0.0
        if cf_weight == 0.0 and content_weight > 0.0:
            content_weight = 1.0
        elif content_weight == 0.0 and cf_weight > 0.0:
            cf_weight = 1.0

        results = []
        for movie_id in candidate_ids:
            movie_row = self._get_movie_row(movie_id)
            if movie_row is None:
                continue
            # Enforce genre exclusions
            if eg_lower and any(g.lower() in eg_lower for g in movie_row["genre_list"]):
                continue
            # Enforce genre inclusions
            if ig_lower and not any(g.lower() in ig_lower for g in movie_row["genre_list"]):
                continue

            final_score = cf_weight * cf_norm.get(movie_id, 0.0) + content_weight * content_norm.get(movie_id, 0.0)
            results.append(
                {
                    "movieId": int(movie_id),
                    "title": movie_row["title"],
                    "genres": movie_row["genres"],
                    "score": round(float(final_score), 4),
                    "cf_score": round(cf_scores.get(movie_id, 0.0), 2) if movie_id in cf_scores else None,
                    "content_score": round(content_scores.get(movie_id, 0.0), 3) if movie_id in content_scores else None,
                    "is_fallback": False,
                }
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Grounded Explainability
    # ------------------------------------------------------------------

    def explain_recommendation(self, user_id: int, movie_id: int) -> dict:
        movie_row = self._get_movie_row(movie_id)
        if movie_row is None:
            return {"user_id": user_id, "movie_id": movie_id, "error": "movie not found"}

        profile = self.get_user_profile(user_id)
        movie_genres = movie_row["genre_list"]

        matching_genres = [g["genre"] for g in profile["top_genres"] if g["genre"] in movie_genres]

        similar_users = self.get_similar_users(user_id, top_n=15, min_common_ratings=3)
        similar_user_ids = [u["userId"] for u in similar_users]
        cohort = self.get_cohort_ratings(movie_id, similar_user_ids)

        similarity_by_user = {u["userId"]: u["similarity"] for u in similar_users}
        cohort_detail = [
            {
                "userId": r["userId"],
                "rating": r["rating"],
                "similarity": similarity_by_user.get(r["userId"]),
            }
            for r in cohort["ratings"]
        ]

        # Check if movie belongs to blind spot
        blind_spot_genres = [b["genre"] for b in profile["blind_spots"]]
        is_blind_spot = any(g in blind_spot_genres for g in movie_genres)

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "title": movie_row["title"],
            "genres": movie_row["genres"],
            "matching_genres": matching_genres,
            "user_avg_rating": profile["avg_rating"],
            "cohort_avg_rating": cohort["avg_rating"],
            "cohort_num_ratings": cohort["num_ratings"],
            "cohort_confidence": cohort.get("confidence"),
            "cohort_global_avg": cohort.get("global_avg_rating"),
            "cohort_ratings": cohort_detail,
            "is_blind_spot": is_blind_spot,
        }

    # ------------------------------------------------------------------
    # 8. Earliest Watched Movies (Timestamp-based)
    # ------------------------------------------------------------------

    def get_oldest_watched_movies(self, user_id: int, min_rating: float = 3.5, limit: int = 1) -> list[dict]:
        """Find movies the user watched and rated longest ago in the past."""
        user_ratings = self.ratings[(self.ratings["userId"] == user_id) & (self.ratings["rating"] >= min_rating)]
        if user_ratings.empty:
            # Fallback to any rated movie
            user_ratings = self.ratings[self.ratings["userId"] == user_id]
        if user_ratings.empty:
            return []

        # Sort by timestamp ascending (earliest first)
        oldest_df = user_ratings.sort_values("timestamp", ascending=True).head(max(1, limit))
        results = []
        for r in oldest_df.itertuples():
            movie_row = self._get_movie_row(int(r.movieId))
            if movie_row is None:
                continue
            
            # Format timestamp to readable date and dynamic delta
            dt = datetime.fromtimestamp(r.timestamp)
            date_str = dt.strftime("%d/%m/%Y")
            years_ago = max(1, datetime.now().year - dt.year)
            
            results.append({
                "movieId": int(r.movieId),
                "title": movie_row["title"],
                "year": int(movie_row["year"]) if "year" in movie_row and pd.notna(movie_row["year"]) else None,
                "genres": movie_row["genres"],
                "user_rating": float(r.rating),
                "watched_date": date_str,
                "years_ago": years_ago,
                "timestamp": int(r.timestamp),
            })
        return results

    # ------------------------------------------------------------------
    # 9. Specific Movie Recommendations for Blind Spots
    # ------------------------------------------------------------------

    def get_blind_spot_recommendations(self, user_id: int, limit: int = 3) -> list[dict]:
        """Recommend specific top-rated movies in the user's blind spot genres."""
        profile = self.get_user_profile(user_id)
        blind_spots = [b["genre"] for b in profile.get("blind_spots", [])]
        if not blind_spots:
            return []

        watched_ids = set(self.ratings.loc[self.ratings["userId"] == user_id, "movieId"])

        # Filter movies belonging to blind spot genres that user hasn't watched
        candidates = self.movie_stats[
            (~self.movie_stats["movieId"].isin(watched_ids)) &
            (self.movie_stats["rating_count"] >= 5) &
            (self.movie_stats["avg_rating"] >= 3.5)
        ].copy()

        # Match at least one blind spot genre
        candidates = candidates[
            candidates["genre_list"].apply(lambda gl: any(g in blind_spots for g in gl))
        ]

        # Prioritize highest average rating then popularity
        candidates = candidates.sort_values(["avg_rating", "rating_count"], ascending=[False, False]).head(max(1, limit))

        results = []
        for r in candidates.itertuples():
            matching_blind_genres = [g for g in r.genre_list if g in blind_spots]
            results.append({
                "movieId": int(r.movieId),
                "title": r.title,
                "genres": r.genres,
                "avg_rating": round(float(r.avg_rating), 2),
                "rating_count": int(r.rating_count),
                "blind_spot_genres": matching_blind_genres,
            })
        return results

    # ------------------------------------------------------------------
    # 10. Integrated Structured Intent Execution
    # ------------------------------------------------------------------

    def execute_structured_intent(self, user_id: int, params: dict) -> dict:
        """Execute structured parameters deterministically and return complete grounded facts."""
        intent = params.get("intent", "recommend")
        limit = max(1, int(params.get("limit", 5)))
        genres_inc = params.get("genres_include") or []
        genres_exc = params.get("genres_exclude") or []
        target_movie = params.get("target_movie")
        search_query = params.get("search_query_en")
        time_filter = params.get("time_filter")
        # 0. Greeting / identity inquiry
        if intent == "greeting":
            return {
                "intent": "greeting",
                "user_id": user_id,
            }

        # 1. User profile / taste / history inquiry
        if intent == "user_profile":
            profile = self.get_user_profile(user_id)
            return {
                "intent": "user_profile",
                "user_id": user_id,
                "num_ratings": profile["num_ratings"],
                "avg_rating": profile["avg_rating"],
                "top_genres": profile["top_genres"][:limit],
                "top_rated_movies": profile["top_movies"][:limit],
                "blind_spots": [b["genre"] for b in profile.get("blind_spots", [])[:3]],
            }

        # 2. Oldest / under-watched movies
        if intent == "oldest_or_underwatched" or time_filter in ("longest_unwatched", "least_watched", "oldest"):
            movies = self.get_oldest_watched_movies(user_id=user_id, min_rating=3.5, limit=limit)
            return {
                "intent": "oldest_or_underwatched",
                "movies": movies,
                "user_id": user_id,
                "description": f"Tìm các phim bạn đã xem và đánh giá từ lâu nhất trong quá khứ."
            }

        # 2. Specific blind spot recommendations
        if intent == "blind_spot" or (genres_inc and any(g.lower() in ("blind_spot", "diem_mu") for g in genres_inc)):
            profile = self.get_user_profile(user_id)
            recs = self.get_blind_spot_recommendations(user_id=user_id, limit=limit)
            return {
                "intent": "blind_spot",
                "blind_spots": profile.get("blind_spots", []),
                "movies": recs,
                "user_id": user_id,
                "description": "Các thể loại điểm mù bạn chưa/ít xem và gợi ý phim tiêu biểu nhất."
            }

        # 3. Cohort opinions on a specific movie
        # P0 FIX: Only trigger cohort_opinion if intent is explicitly cohort_opinion
        if intent == "cohort_opinion":
            movie_to_check = target_movie or "Pulp Fiction"
            sim_users = self.get_similar_users(user_id, top_n=20)
            sim_ids = [u["userId"] for u in sim_users]
            cohort_data = self.get_cohort_ratings(movie_to_check, sim_ids)
            cohort_data["similar_users_checked"] = len(sim_ids)
            return {
                "intent": "cohort_opinion",
                "cohort_data": cohort_data,
                "user_id": user_id
            }

        # 3b. Grounded Explainability (Requirement 3): "Why would I like that?"
        # P0 FIX: Hook up explain_recommendation() into structured intent execution
        if intent in ("why_recommendation", "explain"):
            movie_id = None
            if target_movie:
                mid = self._resolve_movie_id(target_movie)
                if mid:
                    movie_id = mid
            if not movie_id:
                # If no target movie is given, explain the top recommended movie for this user
                top_recs = self.recommend_for_user(user_id=user_id, top_k=1)
                if top_recs:
                    movie_id = top_recs[0]["movieId"]

            if movie_id:
                explanation = self.explain_recommendation(user_id=user_id, movie_id=movie_id)
                return {
                    "intent": "why_recommendation",
                    "explanation": explanation,
                    "user_id": user_id
                }
        # 3c. Dynamic Taste Update & Preference Recording (Assistant Memory)
        if intent == "update_taste":
            set_favs = params.get("set_favorite_genres")
            stated_favs = params.get("genres_include") or params.get("favorite_genres")
            stated_dislikes = params.get("genres_exclude") or params.get("disliked_genres")
            remove_favs = params.get("remove_genres_include") or params.get("remove_favorite_genres")
            remove_dislikes = params.get("remove_genres_exclude") or params.get("remove_disliked_genres")
            clear_all = bool(params.get("clear_all", False))
            target_movie = params.get("target_movie")
            rating_val = params.get("rating")
            notes = params.get("notes")

            result_data = {}
            if set_favs or stated_favs or stated_dislikes or remove_favs or remove_dislikes or clear_all or notes is not None:
                pref_res = self.record_user_preference(
                    user_id=user_id,
                    set_favorite_genres=set_favs,
                    favorite_genres=stated_favs,
                    disliked_genres=stated_dislikes,
                    remove_favorite_genres=remove_favs,
                    remove_disliked_genres=remove_dislikes,
                    clear_all=clear_all,
                    notes=notes
                )
                result_data.update(pref_res)

            if target_movie and rating_val:
                rating_res = self.add_user_rating(
                    user_id=user_id,
                    movie_title_or_id=target_movie,
                    rating=float(rating_val)
                )
                result_data["rating_record"] = rating_res

            return {
                "intent": "update_taste",
                "result": result_data,
                "user_id": user_id,
                "profile": self.get_user_profile(user_id)
            }

        # 4. Pure plot semantic search
        if intent == "plot_search" and search_query and not genres_inc:
            hits = self.search_by_plot(
                query=search_query,
                filter_genres=genres_inc,
                exclude_genres=genres_exc,
                top_k=limit
            )
            return {
                "intent": "plot_search",
                "movies": hits,
                "query": search_query,
                "user_id": user_id
            }

        # 5. Default: Personalized Hybrid Recommendation with strict constraints
        # P0 FIX: Pass include_genres directly to recommend_for_user so candidates are properly pre-filtered
        recs = self.recommend_for_user(
            user_id=user_id,
            query=search_query,
            exclude_genres=genres_exc,
            include_genres=genres_inc,
            top_k=limit
        )

        return {
            "intent": "recommend",
            "movies": recs,
            "user_id": user_id,
            "query": search_query,
            "excluded_genres": genres_exc,
            "included_genres": genres_inc
        }

