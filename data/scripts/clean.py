import ast
import hashlib
import re

SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"sk-[a-zA-Z0-9]{32,}",
    r"-----BEGIN PRIVATE KEY-----",
]

seen_hashes = set()


def has_secret(text):
    return any(re.search(p, text) for p in SECRET_PATTERNS)


def is_valid_python(text):
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def is_minified(text):
    lines = text.splitlines() or [""]
    avg_len = sum(len(l) for l in lines) / len(lines)
    return avg_len > 300


def content_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup(text):
    h = content_hash(text)
    if h in seen_hashes:
        return None
    seen_hashes.add(h)
    return text


def clean_file(text, lang):
    if has_secret(text):
        return None
    if is_minified(text):
        return None
    if lang == "python" and not is_valid_python(text):
        return None
    return dedup(text)


if __name__ == "__main__":
    pass
