# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Shared test texts and statistics helpers for the benchmark scripts."""

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Assistant-style utterances. German first, because that is what the
# assistant speaks; one English line keeps the English path honest.
TEXTS = [
    ("de_short", "German", "Alles klar, ich kümmere mich darum."),
    ("de_sentence", "German", "Das Wetter in Berlin wird morgen sonnig mit Temperaturen um die zweiundzwanzig Grad."),
    ("de_question", "German", "Soll ich dir die Zusammenfassung per E-Mail schicken oder lieber hier vorlesen?"),
    ("de_long", "German",
     "Ich habe drei neue Nachrichten gefunden. Die erste ist von deinem Kollegen und betrifft das Projekt am Montag. "
     "Die zweite ist eine Rechnung, und die dritte ist eine Erinnerung an deinen Zahnarzttermin."),
    ("en_sentence", "English", "Sure, I have added the meeting to your calendar for tomorrow at ten."),
]


def summarize(values):
    """Median, mean, min and max of a list of floats."""
    if not values:
        return {}
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def save_results(label: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{label}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
