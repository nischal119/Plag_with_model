import io
import json
import os
import tempfile

import docx
import PyPDF2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from plagiarism_detector import PlagiarismDetector, TextPreprocessor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize plagiarism detector (now handled above with persistence)

# Global variable to store trained model
model_trained = False

# Initialize detector with model persistence
detector = PlagiarismDetector()
model_trained = detector.is_trained


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
    """Highlight matching phrases in text."""
    highlighted_text = text
    for match in matches:
        phrase = match["phrase"]
        # Simple highlighting - in a real implementation, you'd want more sophisticated matching
        highlighted_text = highlighted_text.replace(phrase, f"<mark>{phrase}</mark>")
    return highlighted_text


@app.route("/")
def index():
    """Main page."""
    return render_template("index.html")


@app.route("/train", methods=["POST"])
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
def get_model_status():
    """Get current model status."""
    global detector
    status = detector.get_model_status()
    return jsonify({"success": True, "status": status})


@app.route("/model/reset", methods=["POST"])
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

            # Highlight matching phrases
            highlighted_text1 = highlight_matching_phrases(text1, matches)
            highlighted_text2 = highlight_matching_phrases(text2, matches)

            return jsonify(
                {
                    "success": True,
                    "result": result,
                    "matches": matches,
                    "highlighted_text1": highlighted_text1,
                    "highlighted_text2": highlighted_text2,
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
                        "message": "Please provide text or file for comparison.",
                    }
                ),
                400,
            )

        # For this demo, we'll create some sample reference texts
        # In a real implementation, you'd load these from a reference folder
        reference_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a high-level programming language.",
            "Data science involves statistics, programming, and domain expertise.",
            "Natural language processing enables computers to understand human language.",
        ]

        results = []
        for i, ref_text in enumerate(reference_texts):
            result = detector.detect_plagiarism(text, ref_text)
            results.append(
                {"reference_id": i + 1, "reference_text": ref_text, "result": result}
            )

        # Sort by plagiarism probability (highest first)
        results.sort(key=lambda x: x["result"]["plagiarism_probability"], reverse=True)

        return jsonify({"success": True, "results": results, "input_text": text})

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
