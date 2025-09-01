"""
Database models for the plagiarism detection system.
"""

from datetime import datetime

import bcrypt
from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication."""

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    # Relationship to plagiarism checks
    plagiarism_checks = db.relationship(
        "PlagiarismCheck", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def __init__(self, username, email, password, first_name, last_name):
        self.username = username
        self.email = email
        self.first_name = first_name
        self.last_name = last_name
        self.set_password(password)

    def set_password(self, password):
        """Hash and set the user's password."""
        self.password_hash = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password):
        """Check if the provided password matches the user's password."""
        return bcrypt.checkpw(
            password.encode("utf-8"), self.password_hash.encode("utf-8")
        )

    def get_full_name(self):
        """Get the user's full name."""
        return f"{self.first_name} {self.last_name}"

    def to_dict(self):
        """Convert user object to dictionary."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "full_name": self.get_full_name(),
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

    def __repr__(self):
        return f"<User {self.username}>"


class PlagiarismCheck(db.Model):
    """Model to store plagiarism check history."""

    __tablename__ = "plagiarism_checks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    check_type = db.Column(
        db.String(20), nullable=False
    )  # 'pairwise', 'classification', 'multiple'
    original_text = db.Column(db.Text, nullable=False)
    reference_text = db.Column(db.Text)  # For pairwise comparison
    plagiarism_score = db.Column(db.Float, nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    total_matches = db.Column(db.Integer, default=0)
    exact_matches = db.Column(db.Integer, default=0)
    semantic_matches = db.Column(db.Integer, default=0)
    file_name = db.Column(db.String(255))  # Original file name if uploaded
    reference_file_name = db.Column(db.String(255))  # Reference file name if uploaded
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert plagiarism check object to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "check_type": self.check_type,
            "original_text": (
                self.original_text[:200] + "..."
                if len(self.original_text) > 200
                else self.original_text
            ),
            "reference_text": (
                self.reference_text[:200] + "..."
                if self.reference_text and len(self.reference_text) > 200
                else self.reference_text
            ),
            "plagiarism_score": self.plagiarism_score,
            "similarity_score": self.similarity_score,
            "total_matches": self.total_matches,
            "exact_matches": self.exact_matches,
            "semantic_matches": self.semantic_matches,
            "file_name": self.file_name,
            "reference_file_name": self.reference_file_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<PlagiarismCheck {self.id} by User {self.user_id}>"


def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)

    with app.app_context():
        # Create all tables
        db.create_all()

        # Create admin user if it doesn't exist
        admin_user = User.query.filter_by(username="admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@example.com",
                password="admin123",  # Change this in production!
                first_name="Admin",
                last_name="User",
            )
            admin_user.is_admin = True
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: username='admin', password='admin123'")

        print("Database initialized successfully!")
