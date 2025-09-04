import os

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from models import PlagiarismCheck, db
from services.detector_service import (
    get_detector,
    is_model_trained,
    reset_model_service,
    set_model_trained,
)
from utils.file_utils import allowed_file, extract_text_from_file
from utils.highlight_utils import highlight_matching_phrases

detect_bp = Blueprint("detect", __name__)


@detect_bp.route("/train", methods=["POST"])
@login_required
def train_model():
    detector = get_detector()
    try:
        results = detector.train("data.csv")
        set_model_trained(True)
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


@detect_bp.route("/model/status", methods=["GET"])
@login_required
def get_model_status():
    detector = get_detector()
    status = detector.get_model_status()
    return jsonify({"success": True, "status": status})


@detect_bp.route("/model/reset", methods=["POST"])
@login_required
def reset_model():
    try:
        reset_model_service()
        return jsonify(
            {"success": True, "message": "Model reset successfully. Training required."}
        )
    except Exception as e:
        return (
            jsonify({"success": False, "message": f"Error resetting model: {str(e)}"}),
            500,
        )


@detect_bp.route("/detect", methods=["POST"])
@login_required
def detect_plagiarism():
    if not is_model_trained():
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Model not trained. Please train the model first.",
                }
            ),
            400,
        )

    detector = get_detector()
    try:
        mode = request.form.get("mode", "pairwise")

        if mode == "pairwise":
            text1 = request.form.get("text1", "")
            text2 = request.form.get("text2", "")

            if "file1" in request.files and request.files["file1"].filename:
                file1 = request.files["file1"]
                if file1 and allowed_file(file1.filename):
                    filename = secure_filename(file1.filename)
                    file_path = os.path.join("uploads", filename)
                    file1.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    text1 = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)

            if "file2" in request.files and request.files["file2"].filename:
                file2 = request.files["file2"]
                if file2 and allowed_file(file2.filename):
                    filename = secure_filename(file2.filename)
                    file_path = os.path.join("uploads", filename)
                    file2.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    text2 = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)

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

            result = detector.detect_plagiarism(text1, text2)
            matches = detector.find_matching_phrases(text1, text2)

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
                            "highlighted_context1": context["highlighted_context1"],
                            "highlighted_context2": context["highlighted_context2"],
                            "highlight1": context["highlight1"],
                            "highlight2": context["highlight2"],
                        }
                    )
                else:
                    detailed_matches.append(match)

            text1_matches = []
            text2_matches = []
            for match in matches:
                if match["match_type"] == "exact":
                    text1_matches.append(match)
                    text2_matches.append(match)
                else:
                    import re

                    phrase1_original = match["phrase1"]
                    phrase2_original = match["phrase2"]
                    pattern1 = re.compile(re.escape(match["phrase1"]), re.IGNORECASE)
                    pattern2 = re.compile(re.escape(match["phrase2"]), re.IGNORECASE)
                    match1 = pattern1.search(text1)
                    match2 = pattern2.search(text2)
                    if match1:
                        phrase1_original = text1[match1.start() : match1.end()]
                    if match2:
                        phrase2_original = text2[match2.start() : match2.end()]
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

            highlighted_text1 = highlight_matching_phrases(text1, text1_matches)
            highlighted_text2 = highlight_matching_phrases(text2, text2_matches)

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
                )
                db.session.add(check)
                db.session.commit()
            except Exception:
                pass

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


@detect_bp.route("/compare_multiple", methods=["POST"])
@login_required
def compare_multiple():
    if not is_model_trained():
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Model not trained. Please train the model first.",
                }
            ),
            400,
        )

    detector = get_detector()
    try:
        text = request.form.get("text", "")

        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join("uploads", filename)
                file.save(file_path)
                file_extension = filename.rsplit(".", 1)[1].lower()
                text = extract_text_from_file(file_path, file_extension)
                os.remove(file_path)

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
                    file_path = os.path.join("uploads", filename)
                    file.save(file_path)
                    file_extension = filename.rsplit(".", 1)[1].lower()
                    ref_text = extract_text_from_file(file_path, file_extension)
                    os.remove(file_path)
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
                        {"success": False, "message": f"Missing reference file {i+1}."}
                    ),
                    400,
                )

        results = []
        for i, (ref_text, ref_name) in enumerate(zip(reference_texts, reference_names)):
            result = detector.detect_plagiarism(text, ref_text)
            matches = detector.find_matching_phrases(text, ref_text)

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
