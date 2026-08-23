"""Pure helpers for merging tokenizer pieces into natural-language words."""
from __future__ import annotations

import string
from typing import Any

import numpy as np


SPECIAL_TOKENS = {"<pad>", "</s>", "<s>", "<unk>", "[PAD]", "[CLS]", "[SEP]"}
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "she", "that", "the", "their", "this", "to", "was", "with",
}


def clean_token(token: str) -> str:
    """Remove common SentencePiece, GPT-style, and WordPiece markers."""
    return (
        token.lstrip("▁Ġ")
        .removeprefix("##")
        .strip()
        .strip(string.punctuation + "“”‘’")
    )


def is_natural_word(text: str) -> bool:
    """Accept alphabetic words with internal apostrophes or hyphens."""
    normalized = text.strip("'’-–-")
    if len(normalized) < 2:
        return False
    return all(character.isalpha() or character in "'’-–-" for character in normalized)


def merge_subtokens_to_words(
    tokens: list[str], scores: np.ndarray
) -> list[dict[str, Any]]:
    """Merge SentencePiece/WordPiece subtokens into natural-word occurrences."""
    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must have identical lengths")
    denominator = max(1, len(scores))
    words: list[dict[str, Any]] = []
    pieces: list[str] = []
    piece_scores: list[float] = []

    def flush() -> None:
        if not pieces:
            return
        surface = "".join(pieces).strip()
        if is_natural_word(surface):
            score_sum = float(np.sum(piece_scores))
            words.append(
                {
                    "word": surface.casefold(),
                    "surface": surface,
                    "subtokens": list(pieces),
                    "subtoken_count": len(pieces),
                    "mean_piece_attribution": float(np.mean(piece_scores)),
                    "additive_contribution": score_sum / denominator,
                }
            )
        pieces.clear()
        piece_scores.clear()

    for raw_token, raw_score in zip(tokens, scores):
        token = str(raw_token)
        if token in SPECIAL_TOKENS or (token.startswith("<") and token.endswith(">")):
            flush()
            continue
        starts_word = token.startswith(("▁", "Ġ"))
        piece = clean_token(token)
        if starts_word:
            flush()
        if not piece or not any(character.isalpha() for character in piece):
            flush()
            continue
        pieces.append(piece)
        piece_scores.append(float(raw_score))
    flush()
    return words


def summarize_attributions(
    values: dict[str, list[dict[str, float]]], *, minimum_count: int
) -> list[dict[str, Any]]:
    rows = []
    for word, occurrences in values.items():
        if len(occurrences) < minimum_count:
            continue
        mean_scores = [item["mean_piece_attribution"] for item in occurrences]
        additive_scores = [item["additive_contribution"] for item in occurrences]
        subtoken_counts = [item["subtoken_count"] for item in occurrences]
        rows.append(
            {
                "word": word,
                "count": len(occurrences),
                "mean_attribution": float(np.mean(mean_scores)),
                "std_attribution": float(np.std(mean_scores)),
                "mean_additive_contribution": float(np.mean(additive_scores)),
                "mean_subtokens": float(np.mean(subtoken_counts)),
            }
        )
    return rows
