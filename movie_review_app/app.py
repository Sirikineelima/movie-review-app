"""
Movie Review App
-----------------
Features:
 - Signup / Login (username + password) with validation constraints
 - Session based auth -> successful login redirects to /movies
 - Movie catalogue with CSS-generated posters (zero image dependency)
 - Each movie has its own detail page: star rating (1-5) + written review
 - Reviews are classified Positive / Neutral / Negative with a weighted
   keyword + negation + intensifier sentiment analyzer (no external ML libs)
 - Users can edit or delete their own review at any time
 - Positive / Neutral / Negative counts + average rating recalculated live
 - Admin account (username "admin") gets a dashboard: manage movies,
   view/delete any review, view/delete users
 - All data persisted in SQLite (movie_reviews.db)
"""

import os
import re
import sqlite3
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DB_PATH", os.path.join(BASE_DIR, "movie_reviews.db"))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-fallback-key")  # used to sign session cookies

# ----------------------------------------------------------------------
# Default movie catalogue.
# ----------------------------------------------------------------------
DEFAULT_MOVIES = [
    ("Galaxy Warriors", 2024, "poster-1", "🚀", "Sci-fi / Action"),
    ("Silent Echoes", 2023, "poster-2", "🎭", "Drama / Mystery"),
    ("Midnight Racer", 2025, "poster-3", "🏎️", "Action / Thriller"),
    ("Ocean's Whisper", 2022, "poster-4", "🌊", "Adventure / Romance"),
    ("The Last Bookstore", 2024, "poster-5", "📚", "Comedy / Drama"),
    ("Crimson Peak Trail", 2023, "poster-6", "🏔️", "Horror / Survival"),
]

# ----------------------------------------------------------------------
# Weighted keyword based sentiment analyzer.
# ----------------------------------------------------------------------
POSITIVE_WORDS = {
    "good": 1, "great": 2, "excellent": 2, "amazing": 2, "awesome": 2,
    "fantastic": 2, "love": 2, "loved": 2, "best": 2, "brilliant": 2,
    "superb": 2, "wonderful": 2, "enjoyed": 1, "enjoy": 1, "nice": 1,
    "beautiful": 1, "perfect": 2, "outstanding": 2, "impressive": 1,
    "masterpiece": 3, "entertaining": 1, "fun": 1, "fabulous": 2,
    "solid": 1, "recommend": 2, "worth": 1, "happy": 1, "super": 1,
    "mass": 1, "blockbuster": 1, "hit": 1, "stunning": 2, "epic": 2,
    "top": 1, "flawless": 2, "gripping": 1, "engaging": 1,
    "satisfying": 1, "delightful": 2, "captivating": 2,
}

NEGATIVE_WORDS = {
    "bad": -1, "worst": -2, "terrible": -2, "awful": -2, "boring": -1,
    "hate": -2, "hated": -2, "poor": -1, "disappointing": -2,
    "disappointed": -2, "waste": -2, "horrible": -2, "dull": -1,
    "weak": -1, "flop": -2, "mediocre": -1, "annoying": -1, "lame": -1,
    "cheap": -1, "pathetic": -2, "slow": -1, "confusing": -1,
    "predictable": -1, "cringe": -1, "avoid": -2, "bore": -1,
    "regret": -2, "unwatchable": -2, "disaster": -2, "letdown": -2,
    "overrated": -1, "forgettable": -1,
}

INTENSIFIERS = {"very", "extremely", "really", "so", "incredibly", "absolutely"}
NEGATIONS = {"not", "no", "never", "n't", "isn't", "wasn't", "didn't", "don't", "cant", "can't", "wont", "won't"}


def analyze_sentiment(text: str, rating: int = None) -> str:
    tokens = re.findall(r"[a-z']+", (text or "").lower())
    score = 0
    matched = False

    for i, tok in enumerate(tokens):
        weight = POSITIVE_WORDS.get(tok) or NEGATIVE_WORDS.get(tok)
        if weight is None:
            continue
        matched = True

        window = tokens[max(0, i - 2):i]
        negated = any(w in NEGATIONS for w in window)
        boosted = any(w in INTENSIFIERS for w in window)

        value = -weight if negated else weight
        if boosted:
            value = int(value * 1.5) or value
        score += value

    if rating:
        if rating >= 4:
            score += 2
        elif rating <= 2:
            score -= 2
        matched = True

    if not matched:
        return "Positive" if "!" in (text or "") else "Neutral"

    if score >= 2:
        return "Positive"
    if score <= -2:
        return "Negative"
    return "Neutral"


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year INTEGER NOT NULL,
            poster_class TEXT NOT NULL,
            emoji TEXT NOT NULL DEFAULT '🎬',
            genre TEXT NOT NULL DEFAULT ''
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            rating INTEGER NOT NULL DEFAULT 3,
            review_text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, movie_id),
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY (movie_id) REFERENCES movies (id) ON DELETE CASCADE
        )
        """
    )
    count = db.execute("SELECT COUNT(*) AS c FROM movies").fetchone()[0]
    if count == 0:
        db.executemany(
            "INSERT INTO movies (name, year, poster_class, emoji, genre) VALUES (?, ?, ?, ?, ?)",
            DEFAULT_MOVIES,
        )
    db.commit()
    db.close()


def login_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


def admin_required(view_func):
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        if not session.get("is_admin"):
            flash("Admin access required.", "error")
            return redirect(url_for("movies"))
        return view_func(*args, **kwargs)
    wrapper.__name__ = view_func.__name__
    return wrapper


USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


def validate_signup(username, password, confirm_password):
    errors = []
    if not username or not USERNAME_RE.match(username):
        errors.append("Username must be 3-20 characters (letters, numbers, underscore only).")
    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    if password != confirm_password:
        errors.append("Password and Confirm Password do not match.")
    return errors


@app.context_processor
def inject_globals():
    return {"current_username": session.get("username"), "is_admin": session.get("is_admin", False)}


@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("movies"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = validate_signup(username, password, confirm_password)

        db = get_db()
        if not errors:
            existing = db.execute(
                "SELECT id FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                errors.append("Username already taken. Please choose another.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("signup.html", username=username)

        password_hash = generate_password_hash(password)
        is_admin = 1 if username.lower() == "admin" else 0
        db.execute(
            "INSERT INTO users (username, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, is_admin, datetime.utcnow().isoformat()),
        )
        db.commit()
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html", username="")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid username or password.", "error")
            return render_template("login.html", username=username)

        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["is_admin"] = bool(user["is_admin"])
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(url_for("movies"))

    return render_template("login.html", username="")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


def get_movies():
    db = get_db()
    return db.execute("SELECT * FROM movies ORDER BY id").fetchall()


def get_movie_or_404(movie_id):
    db = get_db()
    movie = db.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
    return movie


def get_stats():
    db = get_db()
    stats = {}
    for m in get_movies():
        stats[m["id"]] = {"positive": 0, "neutral": 0, "negative": 0, "avg_rating": 0, "total": 0}

    rows = db.execute(
        "SELECT movie_id, sentiment, COUNT(*) as cnt FROM reviews GROUP BY movie_id, sentiment"
    ).fetchall()
    for row in rows:
        mid = row["movie_id"]
        if mid in stats:
            key = row["sentiment"].lower()
            if key in stats[mid]:
                stats[mid][key] = row["cnt"]

    avg_rows = db.execute(
        "SELECT movie_id, AVG(rating) as avg_rating, COUNT(*) as total FROM reviews GROUP BY movie_id"
    ).fetchall()
    for row in avg_rows:
        mid = row["movie_id"]
        if mid in stats:
            stats[mid]["avg_rating"] = round(row["avg_rating"], 1)
            stats[mid]["total"] = row["total"]
    return stats


def get_my_reviews(user_id):
    db = get_db()
    rows = db.execute(
        "SELECT movie_id, rating, review_text, sentiment FROM reviews WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    return {row["movie_id"]: dict(row) for row in rows}


@app.route("/movies")
@login_required
def movies():
    stats = get_stats()
    my_reviews = get_my_reviews(session["user_id"])
    return render_template(
        "movies.html",
        movies=get_movies(),
        stats=stats,
        my_reviews=my_reviews,
    )


@app.route("/movie/<int:movie_id>")
@login_required
def movie_detail(movie_id):
    movie = get_movie_or_404(movie_id)
    if movie is None:
        flash("Movie not found.", "error")
        return redirect(url_for("movies"))

    db = get_db()
    reviews = db.execute(
        """SELECT reviews.*, users.username FROM reviews
           JOIN users ON users.id = reviews.user_id
           WHERE movie_id = ? ORDER BY reviews.updated_at DESC""",
        (movie_id,),
    ).fetchall()
    stats = get_stats()[movie_id]
    my_review = get_my_reviews(session["user_id"]).get(movie_id)

    return render_template(
        "movie_detail.html",
        movie=movie,
        reviews=reviews,
        stats=stats,
        my_review=my_review,
    )


@app.route("/movie/<int:movie_id>/review", methods=["POST"])
@login_required
def submit_single_review(movie_id):
    movie = get_movie_or_404(movie_id)
    if movie is None:
        flash("Movie not found.", "error")
        return redirect(url_for("movies"))

    text = request.form.get("review_text", "").strip()
    try:
        rating = int(request.form.get("rating", 0))
    except ValueError:
        rating = 0

    if not text:
        flash("Please write a review before submitting.", "error")
        return redirect(url_for("movie_detail", movie_id=movie_id))
    if rating < 1 or rating > 5:
        flash("Please choose a star rating between 1 and 5.", "error")
        return redirect(url_for("movie_detail", movie_id=movie_id))

    sentiment = analyze_sentiment(text, rating)
    db = get_db()
    user_id = session["user_id"]
    now = datetime.utcnow().isoformat()

    existing = db.execute(
        "SELECT id FROM reviews WHERE user_id = ? AND movie_id = ?", (user_id, movie_id)
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE reviews SET rating = ?, review_text = ?, sentiment = ?, updated_at = ? WHERE id = ?",
            (rating, text, sentiment, now, existing["id"]),
        )
        flash("Your review has been updated!", "success")
    else:
        db.execute(
            """INSERT INTO reviews (user_id, movie_id, rating, review_text, sentiment, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, movie_id, rating, text, sentiment, now, now),
        )
        flash("Your review has been submitted!", "success")

    db.commit()
    return redirect(url_for("movie_detail", movie_id=movie_id))


@app.route("/movie/<int:movie_id>/review/delete", methods=["POST"])
@login_required
def delete_review(movie_id):
    db = get_db()
    db.execute(
        "DELETE FROM reviews WHERE user_id = ? AND movie_id = ?",
        (session["user_id"], movie_id),
    )
    db.commit()
    flash("Your review has been deleted.", "success")
    return redirect(url_for("movie_detail", movie_id=movie_id))


@app.route("/submit_review", methods=["POST"])
@login_required
def submit_review():
    db = get_db()
    user_id = session["user_id"]
    submitted_any = False
    now = datetime.utcnow().isoformat()

    for movie in get_movies():
        field_name = f"review_{movie['id']}"
        text = request.form.get(field_name, "").strip()
        if not text:
            continue
        rating = int(request.form.get(f"rating_{movie['id']}", 3) or 3)
        sentiment = analyze_sentiment(text, rating)
        submitted_any = True

        existing = db.execute(
            "SELECT id FROM reviews WHERE user_id = ? AND movie_id = ?",
            (user_id, movie["id"]),
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE reviews SET rating = ?, review_text = ?, sentiment = ?, updated_at = ? WHERE id = ?",
                (rating, text, sentiment, now, existing["id"]),
            )
        else:
            db.execute(
                """INSERT INTO reviews (user_id, movie_id, rating, review_text, sentiment, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, movie["id"], rating, text, sentiment, now, now),
            )
    db.commit()

    if submitted_any:
        flash("Your review(s) have been submitted!", "success")
    else:
        flash("Please write at least one review before submitting.", "error")

    return redirect(url_for("movies"))


@app.route("/admin")
@admin_required
def admin_dashboard():
    db = get_db()
    users = db.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id").fetchall()
    reviews = db.execute(
        """SELECT reviews.*, users.username, movies.name as movie_name FROM reviews
           JOIN users ON users.id = reviews.user_id
           JOIN movies ON movies.id = reviews.movie_id
           ORDER BY reviews.updated_at DESC"""
    ).fetchall()
    return render_template(
        "admin.html",
        users=users,
        reviews=reviews,
        movies=get_movies(),
        stats=get_stats(),
    )


@app.route("/admin/movie/add", methods=["POST"])
@admin_required
def admin_add_movie():
    name = request.form.get("name", "").strip()
    year = request.form.get("year", "").strip()
    genre = request.form.get("genre", "").strip()
    emoji = request.form.get("emoji", "🎬").strip() or "🎬"

    if not name or not year.isdigit():
        flash("Movie name and a valid year are required.", "error")
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    count = db.execute("SELECT COUNT(*) as c FROM movies").fetchone()["c"]
    poster_class = f"poster-{(count % 6) + 1}"
    db.execute(
        "INSERT INTO movies (name, year, poster_class, emoji, genre) VALUES (?, ?, ?, ?, ?)",
        (name, int(year), poster_class, emoji, genre),
    )
    db.commit()
    flash(f'"{name}" added to the catalogue.', "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/movie/<int:movie_id>/delete", methods=["POST"])
@admin_required
def admin_delete_movie(movie_id):
    db = get_db()
    db.execute("DELETE FROM movies WHERE id = ?", (movie_id,))
    db.commit()
    flash("Movie (and its reviews) removed.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/review/<int:review_id>/delete", methods=["POST"])
@admin_required
def admin_delete_review(review_id):
    db = get_db()
    db.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    db.commit()
    flash("Review removed.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/user/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    if user_id == session.get("user_id"):
        flash("You can't delete your own account while logged in.", "error")
        return redirect(url_for("admin_dashboard"))
    db = get_db()
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    flash("User removed.", "success")
    return redirect(url_for("admin_dashboard"))


if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
else:
    # Also runs when imported by a WSGI server like gunicorn
    init_db()