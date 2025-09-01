"""
Forms for authentication and user management.
"""

from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError

from models import User


class LoginForm(FlaskForm):
    """Login form."""

    username = StringField(
        "Username or Email",
        validators=[DataRequired(), Length(min=3, max=80)],
        render_kw={"placeholder": "Enter your username or email"},
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter your password"},
    )
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


class SignupForm(FlaskForm):
    """User registration form."""

    first_name = StringField(
        "First Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter your first name"},
    )
    last_name = StringField(
        "Last Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter your last name"},
    )
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)],
        render_kw={"placeholder": "Choose a username"},
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email(), Length(max=120)],
        render_kw={"placeholder": "Enter your email address"},
    )
    password = PasswordField(
        "Password",
        validators=[
            DataRequired(),
            Length(
                min=6, max=128, message="Password must be at least 6 characters long"
            ),
        ],
        render_kw={"placeholder": "Create a password (min 6 characters)"},
    )
    password2 = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match"),
        ],
        render_kw={"placeholder": "Confirm your password"},
    )
    submit = SubmitField("Create Account")

    def validate_username(self, username):
        """Check if username is already taken."""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError(
                "Username already taken. Please choose a different one."
            )

    def validate_email(self, email):
        """Check if email is already registered."""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError(
                "Email address already registered. Please use a different one."
            )


class ProfileForm(FlaskForm):
    """User profile update form."""

    first_name = StringField(
        "First Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter your first name"},
    )
    last_name = StringField(
        "Last Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter your last name"},
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email(), Length(max=120)],
        render_kw={"placeholder": "Enter your email address"},
    )
    submit = SubmitField("Update Profile")

    def __init__(self, original_email, *args, **kwargs):
        super(ProfileForm, self).__init__(*args, **kwargs)
        self.original_email = original_email

    def validate_email(self, email):
        """Check if email is already registered by another user."""
        if email.data != self.original_email:
            user = User.query.filter_by(email=email.data).first()
            if user:
                raise ValidationError(
                    "Email address already registered by another user."
                )


class ChangePasswordForm(FlaskForm):
    """Change password form."""

    current_password = PasswordField(
        "Current Password",
        validators=[DataRequired()],
        render_kw={"placeholder": "Enter your current password"},
    )
    new_password = PasswordField(
        "New Password",
        validators=[
            DataRequired(),
            Length(
                min=6, max=128, message="Password must be at least 6 characters long"
            ),
        ],
        render_kw={"placeholder": "Enter new password (min 6 characters)"},
    )
    new_password2 = PasswordField(
        "Confirm New Password",
        validators=[
            DataRequired(),
            EqualTo("new_password", message="Passwords must match"),
        ],
        render_kw={"placeholder": "Confirm your new password"},
    )
    submit = SubmitField("Change Password")


class AdminUserForm(FlaskForm):
    """Admin form for managing users."""

    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=80)],
        render_kw={"placeholder": "Username"},
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email(), Length(max=120)],
        render_kw={"placeholder": "Email address"},
    )
    first_name = StringField(
        "First Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "First name"},
    )
    last_name = StringField(
        "Last Name",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Last name"},
    )
    is_active = BooleanField("Active User")
    is_admin = BooleanField("Admin User")
    submit = SubmitField("Save User")
