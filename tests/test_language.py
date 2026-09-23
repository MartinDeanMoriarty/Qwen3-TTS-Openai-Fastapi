# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for language resolution and language-aware text normalization.
"""

import pytest

from api.services.language import detect_language, resolve_language
from api.services.text_processing import normalize_text


class TestDetectLanguage:
    @pytest.mark.parametrize("text", [
        "Alles klar, ich kümmere mich darum.",
        "Das Wetter in Berlin wird morgen sonnig.",
        "Soll ich dir die Zusammenfassung per E-Mail schicken?",
        "Ja, gerne.",
        "Es gibt 3 neue Termine für heute.",
    ])
    def test_german(self, text):
        assert detect_language(text) == "German"

    @pytest.mark.parametrize("text", [
        "Sure, I have added the meeting to your calendar.",
        "What would you like to do next?",
        "Yes, please.",
        "The weather in Berlin will be sunny tomorrow.",
    ])
    def test_english(self, text):
        assert detect_language(text) == "English"

    @pytest.mark.parametrize("text", ["Berlin.", "OK", "42", ""])
    def test_undecided_returns_none(self, text):
        assert detect_language(text) is None


class TestResolveLanguage:
    def test_model_suffix_wins(self):
        assert resolve_language("English", "German", "Hello there") == "German"

    def test_explicit_request_beats_detection(self):
        assert resolve_language("English", None, "Alles klar, ich mache das.") == "English"

    def test_language_code_is_mapped(self):
        assert resolve_language("de", None, "Hello") == "German"

    def test_auto_falls_back_to_detection(self):
        assert resolve_language("Auto", None, "Alles klar, ich mache das.") == "German"
        assert resolve_language(None, None, "Sure, I will do that.") == "English"

    def test_undetectable_uses_default(self):
        assert resolve_language(None, None, "Berlin.") == "Auto"


class TestLanguageAwareNormalization:
    def test_english_numbers_are_spelled_out(self):
        assert "twenty-two" in normalize_text("It is 22 degrees.", language="English")

    def test_german_numbers_are_left_to_the_model(self):
        text = normalize_text("Es sind 22 Grad & sonnig.", language="German")
        assert text == "Es sind 22 Grad & sonnig."

    def test_german_still_gets_whitespace_and_quote_cleanup(self):
        text = normalize_text("„Hallo“\n  sagte   er.", language="German")
        assert "\n" not in text
        assert "  " not in text

    def test_auto_keeps_previous_english_behaviour(self):
        assert "twenty-two" in normalize_text("It is 22 degrees.", language="Auto")
