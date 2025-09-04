from datetime import datetime

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from forms import ChangePasswordForm, LoginForm, ProfileForm, SignupForm
from models import PlagiarismCheck, User, db

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    form = LoginForm()
    if form.validate_on_submit():
        if "@" in form.username.data:
            user = User.query.filter_by(email=form.username.data).first()
        else:
            user = User.query.filter_by(username=form.username.data).first()

        if user and user.check_password(form.password.data) and user.is_active:
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user, remember=form.remember_me.data)
            flash(f"Welcome back, {user.get_full_name()}!", "success")
            next_page = request.args.get("next")
            if not next_page or not next_page.startswith("/"):
                next_page = url_for("index")
            return redirect(next_page)
        else:
            flash("Invalid username/email or password.", "error")

    return render_template("auth/login.html", form=form)


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
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


@auth_bp.route("/logout")
@login_required
def logout():
    username = current_user.username
    logout_user()
    flash(f"You have been logged out successfully, {username}.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/profile")
@login_required
def profile():
    checks = (
        PlagiarismCheck.query.filter_by(user_id=current_user.id)
        .order_by(PlagiarismCheck.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template("auth/profile.html", checks=checks)


@auth_bp.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    form = ProfileForm(original_email=current_user.email)

    if form.validate_on_submit():
        current_user.first_name = form.first_name.data
        current_user.last_name = form.last_name.data
        current_user.email = form.email.data
        db.session.commit()
        flash("Your profile has been updated successfully!", "success")
        return redirect(url_for("auth.profile"))
    elif request.method == "GET":
        form.first_name.data = current_user.first_name
        form.last_name.data = current_user.last_name
        form.email.data = current_user.email

    return render_template("auth/edit_profile.html", form=form)


@auth_bp.route("/profile/change_password", methods=["GET", "POST"])
@login_required
def change_password():
    form = ChangePasswordForm()

    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash("Your password has been updated successfully!", "success")
            return redirect(url_for("auth.profile"))
        else:
            flash("Current password is incorrect.", "error")

    return render_template("auth/change_password.html", form=form)


@auth_bp.route("/profile/history")
@login_required
def check_history():
    page = request.args.get("page", 1, type=int)
    checks = (
        PlagiarismCheck.query.filter_by(user_id=current_user.id)
        .order_by(PlagiarismCheck.created_at.desc())
        .paginate(page=page, per_page=20, error_out=False)
    )
    return render_template("auth/history.html", checks=checks)
