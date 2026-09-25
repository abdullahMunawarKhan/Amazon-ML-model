"""Deterministic text normalization that retains raw fields."""

import re
import unicodedata

import pandas as pd

ABBREVIATIONS = {
    "corp": "corporation", "co": "company", "inc": "incorporated",
    "pvt": "private", "ltd": "limited", "llc": "limited liability company",
    "rd": "road", "st": "street", "str": "street", "ave": "avenue",
    "av": "avenue", "blvd": "boulevard", "dr": "drive", "hwy": "highway",
    "ln": "lane", "ct": "court", "apt": "apartment", "ste": "suite",
}
LEGAL_SUFFIXES = {"corporation", "company", "incorporated", "limited", "private", "llc", "limited liability company"}


def transliterate(value: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", value) if not unicodedata.combining(char))


def normalize_text(value: object, remove_legal_suffixes: bool = False) -> str:
    text = transliterate(str(value)).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [ABBREVIATIONS.get(token, token) for token in text.split()]
    if remove_legal_suffixes:
        while tokens and tokens[-1] in LEGAL_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


def add_normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["name_norm"] = result["business_name"].map(normalize_text)
    result["name_core"] = result["business_name"].map(lambda value: normalize_text(value, True))
    result["address_norm"] = result["business_address"].map(normalize_text)
    result["name_tokens"] = result["name_norm"].str.split()
    result["address_tokens"] = result["address_norm"].str.split()
    return result