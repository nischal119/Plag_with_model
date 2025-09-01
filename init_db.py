#!/usr/bin/env python3
"""
Database Initialization Script
Initializes the database and creates an admin user for the plagiarism detection system.
"""

import os
import sys
from getpass import getpass

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from models import User, db


def create_database():
    """Create the database tables."""
    print("Creating database tables...")
    with app.app_context():
        try:
            db.create_all()
            print("✓ Database tables created successfully!")
            return True
        except Exception as e:
            print(f"✗ Error creating database tables: {e}")
            return False


def create_admin_user():
    """Create an admin user interactively."""
    print("\n" + "=" * 50)
    print("ADMIN USER CREATION")
    print("=" * 50)

    with app.app_context():
        # Check if admin user already exists
        existing_admin = User.query.filter_by(is_admin=True).first()
        if existing_admin:
            print(
                f"✓ Admin user already exists: {existing_admin.username} ({existing_admin.email})"
            )

            response = (
                input("Do you want to create another admin user? (y/N): ")
                .strip()
                .lower()
            )
            if response != "y":
                return True

        print("Creating a new admin user...")

        # Get user input
        while True:
            username = input("Username: ").strip()
            if not username:
                print("Username cannot be empty!")
                continue

            # Check if username exists
            if User.query.filter_by(username=username).first():
                print("Username already exists! Please choose a different one.")
                continue

            if len(username) < 3:
                print("Username must be at least 3 characters long!")
                continue

            break

        while True:
            email = input("Email: ").strip()
            if not email:
                print("Email cannot be empty!")
                continue

            # Basic email validation
            if "@" not in email or "." not in email:
                print("Please enter a valid email address!")
                continue

            # Check if email exists
            if User.query.filter_by(email=email).first():
                print("Email already exists! Please use a different one.")
                continue

            break

        first_name = input("First Name: ").strip()
        while not first_name:
            print("First name cannot be empty!")
            first_name = input("First Name: ").strip()

        last_name = input("Last Name: ").strip()
        while not last_name:
            print("Last name cannot be empty!")
            last_name = input("Last Name: ").strip()

        while True:
            password = getpass("Password (min 6 characters): ")
            if len(password) < 6:
                print("Password must be at least 6 characters long!")
                continue

            password_confirm = getpass("Confirm Password: ")
            if password != password_confirm:
                print("Passwords do not match!")
                continue

            break

        try:
            # Create the admin user
            admin_user = User(
                username=username,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
            )
            admin_user.is_admin = True

            db.session.add(admin_user)
            db.session.commit()

            print(f"\n✓ Admin user created successfully!")
            print(f"  Username: {admin_user.username}")
            print(f"  Email: {admin_user.email}")
            print(f"  Full Name: {admin_user.get_full_name()}")

            return True

        except Exception as e:
            print(f"\n✗ Error creating admin user: {e}")
            db.session.rollback()
            return False


def create_sample_users():
    """Create some sample users for testing."""
    print("\n" + "=" * 50)
    print("SAMPLE USERS CREATION")
    print("=" * 50)

    response = (
        input("Do you want to create sample users for testing? (y/N): ").strip().lower()
    )
    if response != "y":
        return True

    sample_users = [
        {
            "username": "john_doe",
            "email": "john@example.com",
            "password": "password123",
            "first_name": "John",
            "last_name": "Doe",
            "is_admin": False,
        },
        {
            "username": "jane_smith",
            "email": "jane@example.com",
            "password": "password123",
            "first_name": "Jane",
            "last_name": "Smith",
            "is_admin": False,
        },
        {
            "username": "teacher_demo",
            "email": "teacher@example.com",
            "password": "teacher123",
            "first_name": "Demo",
            "last_name": "Teacher",
            "is_admin": False,
        },
    ]

    with app.app_context():
        created_count = 0

        for user_data in sample_users:
            # Check if user already exists
            existing_user = User.query.filter(
                (User.username == user_data["username"])
                | (User.email == user_data["email"])
            ).first()

            if existing_user:
                print(f"  - Skipping {user_data['username']} (already exists)")
                continue

            try:
                user = User(
                    username=user_data["username"],
                    email=user_data["email"],
                    password=user_data["password"],
                    first_name=user_data["first_name"],
                    last_name=user_data["last_name"],
                )
                user.is_admin = user_data["is_admin"]

                db.session.add(user)
                db.session.commit()

                print(f"  ✓ Created user: {user.username} ({user.email})")
                created_count += 1

            except Exception as e:
                print(f"  ✗ Error creating user {user_data['username']}: {e}")
                db.session.rollback()

        if created_count > 0:
            print(f"\n✓ Created {created_count} sample users successfully!")
            print("  All sample users have password: 'password123' or 'teacher123'")
        else:
            print("  No new sample users were created.")

        return True


def show_database_info():
    """Show information about the database."""
    print("\n" + "=" * 50)
    print("DATABASE INFORMATION")
    print("=" * 50)

    with app.app_context():
        try:
            user_count = User.query.count()
            admin_count = User.query.filter_by(is_admin=True).count()

            print(f"Database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
            print(f"Total users: {user_count}")
            print(f"Admin users: {admin_count}")
            print(f"Regular users: {user_count - admin_count}")

            if user_count > 0:
                print("\nExisting users:")
                users = User.query.all()
                for user in users:
                    admin_status = " (Admin)" if user.is_admin else ""
                    print(f"  - {user.username} ({user.email}){admin_status}")

        except Exception as e:
            print(f"Error retrieving database information: {e}")


def main():
    """Main initialization function."""
    print("=" * 60)
    print("PLAGIARISM DETECTION SYSTEM - DATABASE INITIALIZATION")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(
        f"Database URL: {app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')}"
    )

    try:
        # Step 1: Create database tables
        if not create_database():
            print("\n✗ Database initialization failed!")
            return False

        # Step 2: Create admin user
        if not create_admin_user():
            print("\n✗ Admin user creation failed!")
            return False

        # Step 3: Create sample users (optional)
        create_sample_users()

        # Step 4: Show database information
        show_database_info()

        print("\n" + "=" * 60)
        print("✓ DATABASE INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run the application with:")
        print("  python app.py")
        print("\nAccess the application at:")
        print("  http://localhost:5001")

        return True

    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user.")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error during initialization: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
