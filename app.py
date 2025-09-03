import io
import json
import os
import tempfile
from datetime import datetime

import docx
import PyPDF2
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from werkzeug.utils import secure_filename

from forms import ChangePasswordForm, LoginForm, ProfileForm, SignupForm
from models import PlagiarismCheck, User, db, init_db
from plagiarism_detector import PlagiarismDetector, TextPreprocessor

app = Flask(__name__)

# Configuration config
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["SECRET_KEY"] = (
    os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"
)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    os.environ.get("DATABASE_URL") or "sqlite:///plagiarism_detector.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["WTF_CSRF_ENABLED"] = True

# Initialize extensions
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database
init_db(app)

# Initialize plagiarism detector (now handled above with persistence)

# Global variable to store trained model
model_trained = False

# Initialize detector with model persistence
detector = PlagiarismDetector()
model_trained = detector.is_trained


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return User.query.get(int(user_id))


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_path, file_extension):
    """Extract text from uploaded file based on its type."""
    try:
        if file_extension == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_extension == "docx":
            doc = docx.Document(file_path)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])

        elif file_extension == "pdf":
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text

        else:
            return None
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None


def highlight_matching_phrases(text, matches):
    """Highlight matching phrases in text with different colors for exact and semantic matches."""
    # Create a list of all phrases to highlight with their types and positions
    phrases_to_highlight = []

    for match in matches:
        if match["match_type"] == "exact":
            phrase = match["phrase"]
            # Find all occurrences of this phrase
            import re

            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            for match_obj in pattern.finditer(text):
                phrases_to_highlight.append(
                    {
                        "start": match_obj.start(),
                        "end": match_obj.end(),
                        "phrase": phrase,
                        "type": "exact",
                        "length": len(phrase),
                    }
                )
        else:  # semantic match
            # For semantic matches, we have a single phrase to highlight
            phrase = match.get("phrase", "")
            if phrase:
                import re

                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                for match_obj in pattern.finditer(text):
                    phrases_to_highlight.append(
                        {
                            "start": match_obj.start(),
                            "end": match_obj.end(),
                            "phrase": phrase,
                            "type": "semantic",
                            "length": len(phrase),
                            "tooltip": f"Semantic match (similarity: {match.get('similarity', 0):.2f})",
                        }
                    )

    # Sort by start position
    phrases_to_highlight.sort(key=lambda x: x["start"])

    # Remove overlapping highlights (keep the longer ones)
    filtered_phrases = []
    for phrase_info in phrases_to_highlight:
        # Check if this phrase overlaps with any existing phrase
        overlaps = False
        for existing in filtered_phrases:
            if (
                phrase_info["start"] < existing["end"]
                and phrase_info["end"] > existing["start"]
            ):
                overlaps = True
                break

        if not overlaps:
            filtered_phrases.append(phrase_info)

    # Sort by start position again and apply highlighting
    filtered_phrases.sort(key=lambda x: x["start"])

    # Apply highlighting from end to start to avoid position shifting
    highlighted_text = text
    for phrase_info in reversed(filtered_phrases):
        start = phrase_info["start"]
        end = phrase_info["end"]
        phrase = phrase_info["phrase"]
        phrase_type = phrase_info["type"]

        if phrase_type == "exact":
            replacement = f'<span class="highlight-exact">{phrase}</span>'
        else:
            tooltip = phrase_info.get("tooltip", "")
            replacement = (
                f'<span class="highlight-semantic" title="{tooltip}">{phrase}</span>'
            )

        highlighted_text = (
            highlighted_text[:start] + replacement + highlighted_text[end:]
        )

    return highlighted_text


# Authentication Routes


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = LoginForm()
    if form.validate_on_submit():
        # Check if input is email or username
        if "@" in form.username.data:
            user = User.query.filter_by(email=form.username.data).first()
        else:
            user = User.query.filter_by(username=form.username.data).first()

        if user and user.check_password(form.password.data) and user.is_active:
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()

            login_user(user, remember=form.remember_me.data)
            flash(f"Welcome back, {user.get_full_name()}!", "success")

            # Redirect to next page or index
            next_page = request.args.get("next")
            if not next_page or not next_page.startswith("/"):
                next_page = url_for("index")
            return redirect(next_page)
        else:
            flash("Invalid username/email or password.", "error")

    return render_template("auth/login.html", form=form)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = SignupForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
        )
        db.session.add(user)
        db.session.commit()

        flash(
            f"Account created successfully! Welcome, {user.get_full_name()}!", "success"
        )
        login_user(user)
        return redirect(url_for("index"))

    return render_template("auth/signup.html", form=form)


@app.route("/logout")
@login_required
def logout():
    """User logout."""
    username = current_user.username
    logout_user()
    flash(f"You have been logged out successfully, {username}.", "info")
    return redirect(url_for("login"))


@app.route("/profile")
@login_required
def profile():
    """User profile page."""
    # Get user's plagiarism check history
    checks = (
        PlagiarismCheck.query.filter_by(user_id=current_user.id)
        .order_by(PlagiarismCheck.created_at.desc())
        .limit(10)
        .all()
    )

    return render_template("auth/profile.html", checks=checks)


@app.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    """Edit user profile."""
    form = ProfileForm(original_email=current_user.email)

    if form.validate_on_submit():
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Your profile has been updated successfully!", "success")
        return redirect(url_for("profile"))
    elif request.method == "GET":
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email

    return render_template("auth/edit_profile.html", form=form)


@app.route("/profile/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    """Change user password."""
    form = ChangePasswordForm()

    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash("Your password has been updated successfully!", "success")
            return redirect(url_for("profile"))
        else:
            flash("Current password is incorrect.", "error")

    return render_template("auth/change_password.html", form=form)


@app.route("/profile/history")
@login_required
def check_history():
    """View plagiarism check history."""
    page = request.args.get("page", 1, type=int)
    checks = (
        PlagiarismCheck.query.filter_by(user_id=current_user.id)
        .order_by(PlagiarismCheck.created_at.desc())
        .paginate(page=page, per_page=20, error_out=False)
    )

    return render_template("auth/history.html", checks=checks)


@app.route("/")
@login_required
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/train", methods=["POST"])
@login_required
def train_model():
    """Train the plagiarism detection model."""
    global detector, model_trained

    try:
        # Train the model
        results = detector.train("data.csv")
        model_trained = True

        return jsonify(
            {
                "success": True,
                "message": "Model trained successfully!",
                "results": results,
            }
        )
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error training model: {str(e)}"}),
            500,
        )


@app.route("/model/status", methods=["GET"])
@login_required
def get_model_status():
    """Get current model status."""
    global detector
    status = detector.get_model_status()
    return jsonify({"success": True, "status": status})


@app.route("/model/reset", methods=["POST"])
@login_required
def reset_model():
    """Reset the model."""
    global detector, model_trained

    try:
        detector.reset_model()
        model_trained = False
        return jsonify(
            {"success": True, "message": "Model reset successfully. Training required."}
        )
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error resetting model: {str(e)}"}),
            500,
        )


@app.route("/detect", methods=["POST"])
@login_required
def detect_plagiarism():
    """Detect plagiarism between two texts or files."""
    global detector, model_trained

    if not model_trained:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Model not trained. Please train the model first.",
                }
            ),
            400,
        )

    try:
        mode = request.form.get("mode", "pairwise")

        if mode == "pairwise":
            # Pairwise comparison mode
            text1 = request.form.get("text1", "")
            text2 = request.form.get("text2", "")

            # Handle file uploads
            if "file1" in request.files and request.files["file1"].filename:
                file1 = request.files["file1"]
                if file1 and allowed_file(file1.filename):
                    filename = secure_filename(file1.filename)
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file1.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    text1 = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)  # Clean up

            if "file2" in request.files and request.files["file2"].filename:
                file2 = request.files["file2"]
                if file2 and allowed_file(file2.filename):
                    filename = secure_filename(file2.filename)
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file2.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    text2 = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)  # Clean up

            if not text1 or not text2:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Please provide both texts or files for comparison.",
                        }
                    ),
                    400,
                )

            # Detect plagiarism
            result = detector.detect_plagiarism(text1, text2)
            matches = detector.find_matching_phrases(text1, text2)

            # Get detailed matching information with context
            detailed_matches = []
            for match in matches:
                if match["match_type"] == "exact":
                    context = detector.get_matching_context(
                        text1, text2, match["phrase"]
                    )
                    detailed_matches.append(
                        {
                            **match,
                            "context1": context["context1"],
                            "context2": context["context2"],
                            "highlight1": context["highlight1"],
                            "highlight2": context["highlight2"],
                        }
                    )
                else:  # semantic match
                    detailed_matches.append(match)

            # Create separate match lists for highlighting each text
            text1_matches = []
            text2_matches = []

            for match in matches:
                if match["match_type"] == "exact":
                    # Add exact matches to both texts
                    text1_matches.append(match)
                    text2_matches.append(match)
                else:  # semantic match
                    # Find the original phrases in the text (preserving case)
                    import re

                    phrase1_original = match["phrase1"]
                    phrase2_original = match["phrase2"]

                    # Find the actual phrases in the original text
                    pattern1 = re.compile(re.escape(match["phrase1"]), re.IGNORECASE)
                    pattern2 = re.compile(re.escape(match["phrase2"]), re.IGNORECASE)

                    match1 = pattern1.search(text1)
                    match2 = pattern2.search(text2)

                    if match1:
                        phrase1_original = text1[match1.start() : match1.end()]
                    if match2:
                        phrase2_original = text2[match2.start() : match2.end()]

                    # Add semantic matches to respective texts
                    text1_matches.append(
                        {
                            "match_type": "semantic",
                            "phrase": phrase1_original,
                            "similarity": match["similarity"],
                        }
                    )
                    text2_matches.append(
                        {
                            "match_type": "semantic",
                            "phrase": phrase2_original,
                            "similarity": match["similarity"],
                        }
                    )

            # Highlight matching phrases with better highlighting
            highlighted_text1 = highlight_matching_phrases(text1, text1_matches)
            highlighted_text2 = highlight_matching_phrases(text2, text2_matches)

            # Save plagiarism check to history
            try:
                check = PlagiarismCheck(
                    user_id=current_user.id,
                    check_type="pairwise",
                    original_text=text1,
                    reference_text=text2,
                    plagiarism_score=result.get("plagiarism_probability", 0),
                    similarity_score=result.get("similarity_score", 0),
                    total_matches=len(detailed_matches),
                    exact_matches=len(
                        [m for m in detailed_matches if m["match_type"] == "exact"]
                    ),
                    semantic_matches=len(
                        [m for m in detailed_matches if m["match_type"] == "semantic"]
                    ),
                    file_name=(
                        request.files.get("file1", {}).filename
                        if "file1" in request.files
                        else None
                    ),
                    reference_file_name=(
                        request.files.get("file2", {}).filename
                        if "file2" in request.files
                        else None
                    ),
                )
                db.session.add(check)
                db.session.commit()
            except Exception as e:
                print(f"Error saving plagiarism check: {e}")
                # Don't fail the request if history saving fails

            return jsonify(
                {
                    "success": True,
                    "result": result,
                    "matches": detailed_matches,
                    "highlighted_text1": highlighted_text1,
                    "highlighted_text2": highlighted_text2,
                    "summary": {
                        "total_matches": len(detailed_matches),
                        "exact_matches": len(
                            [m for m in detailed_matches if m["match_type"] == "exact"]
                        ),
                        "semantic_matches": len(
                            [
                                m
                                for m in detailed_matches
                                if m["match_type"] == "semantic"
                            ]
                        ),
                        "longest_match": (
                            max([m["length"] for m in detailed_matches])
                            if detailed_matches
                            else 0
                        ),
                    },
                }
            )

        elif mode == "classification":
            # Classification mode - compare against reference documents
            text = request.form.get("text", "")

            # Handle file upload
            if "file" in request.files and request.files["file"].filename:
                file = request.files["file"]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    text = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)  # Clean up

            if not text:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": "Please provide text or file for classification.",
                        }
                    ),
                    400,
                )

            # For classification, we'll use a simple approach:
            # Compare against some reference texts or use the model's features
            # This is a simplified version - in practice, you'd have a reference corpus

            # For now, let's create a dummy reference text and compare
            reference_text = (
                "This is a sample reference text for classification purposes."
            )
            result = detector.detect_plagiarism(text, reference_text)

            return jsonify(
                {
                    "success": True,
                    "result": result,
                    "input_text": text,
                    "reference_text": reference_text,
                }
            )

        else:
            return (
                jsonify({"success": False, "message": "Invalid mode specified."}),
                400,
            )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error during plagiarism detection: {str(e)}",
                }
            ),
            500,
        )


@app.route("/compare_multiple", methods=["POST"])
@login_required
def compare_multiple():
    """Compare uploaded text against multiple reference documents."""
    global detector, model_trained

    if not model_trained:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Model not trained. Please train the model first.",
                }
            ),
            400,
        )

    try:
        text = request.form.get("text", "")

        # Handle original file upload
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                file_extension = filename.rsplit(".", 1)[1].lower()
                text = extract_text_from_file(file_path, file_extension)
                os.remove(file_path)  # Clean up

        if not text:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Please provide original text or file for comparison.",
                    }
                ),
                400,
            )

        # Get reference files
        reference_count = int(request.form.get("reference_count", 0))
        if reference_count == 0:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Please upload at least one reference document.",
                    }
                ),
                400,
            )

        reference_texts = []
        reference_names = []

        for i in range(reference_count):
            file_key = f"reference_file_{i}"
            name_key = f"reference_name_{i}"

            if file_key in request.files and request.files[file_key].filename:
                file = request.files[file_key]
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                    file.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    ref_text = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)  # Clean up

                    reference_texts.append(ref_text)
                    reference_names.append(
                        request.form.get(name_key, f"Reference {i+1}")
                    )
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "message": f"Invalid file format for reference {i+1}.",
                            }
                        ),
                        400,
                    )
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Missing reference file {i+1}.",
                        }
                    ),
                    400,
                )

        results = []
        for i, (ref_text, ref_name) in enumerate(zip(reference_texts, reference_names)):
            result = detector.detect_plagiarism(text, ref_text)
            matches = detector.find_matching_phrases(text, ref_text)

            # Get detailed matching information
            detailed_matches = []
            for match in matches:
                if match["match_type"] == "exact":
                    context = detector.get_matching_context(
                        text, ref_text, match["phrase"]
                    )
                    detailed_matches.append(
                        {
                            **match,
                            "context1": context["context1"],
                            "context2": context["context2"],
                            "highlight1": context["highlight1"],
                            "highlight2": context["highlight2"],
                        }
                    )
                else:
                    detailed_matches.append(match)

            # Create separate match lists for highlighting each text
            text1_matches = []
            text2_matches = []

            for match in matches:
                if match["match_type"] == "exact":
                    text1_matches.append(match)
                    text2_matches.append(match)
                else:
                    text1_matches.append(
                        {
                            "match_type": "semantic",
                            "phrase": match["phrase1"],
                            "similarity": match["similarity"],
                        }
                    )
                    text2_matches.append(
                        {
                            "match_type": "semantic",
                            "phrase": match["phrase2"],
                            "similarity": match["similarity"],
                        }
                    )

            # Highlight matching phrases
            highlighted_text1 = highlight_matching_phrases(text, text1_matches)
            highlighted_text2 = highlight_matching_phrases(ref_text, text2_matches)

            results.append(
                {
                    "reference_id": i + 1,
                    "reference_name": ref_name,
                    "reference_text": ref_text,
                    "result": result,
                    "matches": detailed_matches,
                    "highlighted_text1": highlighted_text1,
                    "highlighted_text2": highlighted_text2,
                    "summary": {
                        "total_matches": len(detailed_matches),
                        "exact_matches": len(
                            [m for m in detailed_matches if m["match_type"] == "exact"]
                        ),
                        "semantic_matches": len(
                            [
                                m
                                for m in detailed_matches
                                if m["match_type"] == "semantic"
                            ]
                        ),
                        "longest_match": (
                            max([m["length"] for m in detailed_matches])
                            if detailed_matches
                            else 0
                        ),
                    },
                }
            )

        # Sort by plagiarism probability (highest first)
        results.sort(key=lambda x: x["result"]["plagiarism_probability"], reverse=True)

        return jsonify(
            {
                "success": True,
                "results": results,
                "input_text": text,
                "original_file_name": (
                    request.files.get("file", {}).filename
                    if "file" in request.files
                    else None
                ),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Error during multiple comparison: {str(e)}",
                }
            ),
            500,
        )


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
