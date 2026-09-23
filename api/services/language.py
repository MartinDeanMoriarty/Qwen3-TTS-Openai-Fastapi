# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Resolve the synthesis language of a request.

Clients such as Open WebUI send no language at all. Qwen3-TTS takes the
language as a conditioning token, so guessing it from the text beats leaving it
to chance. The guess is deliberately small: it only has to tell German from
English in assistant replies, and it declines when it is not sure. Words that
are common in both languages (an, am, in, was, war, will) count for neither.
"""

import os
import re
from typing import Optional

# Used when the request, the model name and the text leave the language open.
# "Auto" lets Qwen3-TTS decide on its own.
DEFAULT_LANGUAGE = os.getenv("TTS_DEFAULT_LANGUAGE", "Auto")

LANGUAGE_CODES = {
    "en": "English",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

_GERMAN_WORDS = frozenset(
    "der die das den dem des und ist sind nicht ich du er sie es wir ihr ein eine einen einem einer "
    "zu mit auf für von vom zum zur im um aus bei nach über unter noch auch schon nur sehr "
    "wird werden wurde hat haben habe bin bist dir mir dich mich dein deine mein meine sich "
    "kann soll muss oder aber wenn dass weil wie wer wo hier heute morgen gestern bitte "
    "danke gerne klar ja nein doch alles etwas nichts gibt gut".split()
)
_ENGLISH_WORDS = frozenset(
    "the a and or but is are were be been you your i me my we our they their he she it its "
    "to of on at for with from by this that these those have has had do does did not can could "
    "will would should what which who how where when there here please thanks yes no sure okay "
    "today tomorrow just all".split()
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_language(text: str) -> Optional[str]:
    """Return "German" or "English" when the text clearly is one of them, else None."""
    german = len(re.findall(r"[äöüÄÖÜß]", text))
    english = 0
    for word in _WORD.findall(text.lower()):
        german += word in _GERMAN_WORDS
        english += word in _ENGLISH_WORDS
    if german > english:
        return "German"
    if english > german:
        return "English"
    return None


def resolve_language(requested: Optional[str], model_language: Optional[str], text: str) -> str:
    """Pick the language: model suffix, then explicit request, then the text, then the default."""
    if model_language:
        return model_language
    if requested and requested.lower() != "auto":
        return LANGUAGE_CODES.get(requested.lower(), requested)
    return detect_language(text) or DEFAULT_LANGUAGE
