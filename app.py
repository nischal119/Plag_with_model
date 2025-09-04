import os

from flask import Flask, render_template, send_from_directory, url_for
from flask_login import LoginManager, login_required

from blueprints.auth_bp import auth_bp
from blueprints.detect_bp import detect_bp
from models import User, init_db

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
login_manager.login_view = "auth.login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "info"

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize database
init_db(app)

"""Routes are organized via blueprints in `blueprints/`.
Aux services are in `services/`, utils in `utils/`.
"""


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


from utils.highlight_utils import (
    highlight_matching_phrases,  # re-export for templates if needed
)

app.register_blueprint(auth_bp)
app.register_blueprint(detect_bp)


@app.route("/")
@login_required
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files."""
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
