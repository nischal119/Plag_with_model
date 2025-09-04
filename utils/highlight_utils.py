from typing import Dict, List


def highlight_matching_phrases(text: str, matches: List[Dict]) -> str:
    phrases_to_highlight: List[Dict] = []

    def build_flexible_pattern(p: str):
        import re

        words = [re.escape(w) for w in p.split() if w]
        if not words:
            return None
        pattern_str = r"\b" + r"\W+".join(words) + r"\b"
        return re.compile(pattern_str, re.IGNORECASE)

    for match in matches:
        if match["match_type"] == "exact":
            phrase = match["phrase"]
            import re

            flex = build_flexible_pattern(phrase)
            # Try flexible first, then fall back to strict
            patterns = []
            if flex is not None:
                patterns.append(flex)
            patterns.append(re.compile(re.escape(phrase), re.IGNORECASE))

            found_any = False
            for pattern in patterns:
                for match_obj in pattern.finditer(text):
                    phrases_to_highlight.append(
                        {
                            "start": match_obj.start(),
                            "end": match_obj.end(),
                            "matched": match_obj.group(0),
                            "phrase": phrase,
                            "type": "exact",
                            "length": len(phrase),
                        }
                    )
                    found_any = True
                if found_any:
                    break
        else:
            phrase = match.get("phrase", "")
            if phrase:
                import re

                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                for match_obj in pattern.finditer(text):
                    phrases_to_highlight.append(
                        {
                            "start": match_obj.start(),
                            "end": match_obj.end(),
                            "matched": match_obj.group(0),
                            "phrase": phrase,
                            "type": "semantic",
                            "length": len(phrase),
                            "tooltip": f"Semantic match (similarity: {match.get('similarity', 0):.2f})",
                        }
                    )

    phrases_to_highlight.sort(key=lambda x: x["start"])

    filtered_phrases = []
    for phrase_info in phrases_to_highlight:
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

    filtered_phrases.sort(key=lambda x: x["start"])

    highlighted_text = text
    for phrase_info in reversed(filtered_phrases):
        start = phrase_info["start"]
        end = phrase_info["end"]
        matched_substring = phrase_info.get("matched", phrase_info.get("phrase", ""))
        phrase_type = phrase_info["type"]

        if phrase_type == "exact":
            replacement = f'<span class="highlight-exact">{matched_substring}</span>'
        else:
            tooltip = phrase_info.get("tooltip", "")
            replacement = f'<span class="highlight-semantic" title="{tooltip}">{matched_substring}</span>'

        highlighted_text = (
            highlighted_text[:start] + replacement + highlighted_text[end:]
        )

    return highlighted_text
