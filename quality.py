"""Mechanical QA checks used alongside the QA model."""

import re
from collections import Counter

REPETITION_LIMIT = 0.45
SCAFFOLD_MARKERS = (
    "analysis:",
    "status: evaluating",
    "plan:",
    "execution:",
    "drafting:",
    "private context",
    "private instructions",
    "the user wants",
    "i need to output",
    "persona of",
    "in-character",
)
REFUSAL_MARKERS = (
    "as an ai",
    "i am an ai",
    "i'm an ai",
    "as a language model",
    "as a large language model",
    "ai language model",
    "as a machine",
    "as a helpful assistant",
    "i am programmed to",
    "against my programming",
    "programming prohibits",
    "i cannot provide",
    "i cannot generate",
    "i cannot create content",
    "i cannot create reasoning",
    "i cannot write",
    "i cannot compose",
    "i cannot fulfill",
    "i cannot fulfil",
    "i cannot assist",
    "i cannot roleplay",
    "i cannot role-play",
    "i cannot claim",
    "i cannot engage in",
    "i cannot help with",
    "i can't help with",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "i am unable to",
    "i must decline",
    "safety guidelines",
    "safety policies",
    "against my safety",
    "fabricates false",
)

REFUSAL_SUBSTRING = re.compile(
    "|".join(r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])" for marker in REFUSAL_MARKERS),
    re.I,
)
TASK_VERB = (
    r"writ(e|ing)|creat(e|ing)|generat(e|ing)|produc(e|ing)|compos(e|ing)|"
    r"craft(ing)?|fulfil?l|comply|complet(e|ing)|participat(e|ing)|roleplay|"
    r"role-play|fabricat(e|ing)"
)
REFUSAL_TASK = re.compile(
    rf"\bi (?:cannot|can't|will not|won't|am unable to|'m unable to) (?:{TASK_VERB})",
    re.I,
)


def repeated_trigram_fraction(text):
    words = text.split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    return sum(n for n in counts.values() if n > 1) / len(trigrams)


def rejection_reason(text):
    lowered = text.lower()
    marker = next((marker for marker in SCAFFOLD_MARKERS if marker in lowered), None)
    if marker:
        return f"scaffold-leak:{marker}"
    match = REFUSAL_SUBSTRING.search(text) or REFUSAL_TASK.search(text)
    if match:
        return f"refusal:{match.group(0)[:40]}"
    repetition = repeated_trigram_fraction(text)
    if repetition > REPETITION_LIMIT:
        return f"repetition:rep3={repetition:.2f}"
    return None
