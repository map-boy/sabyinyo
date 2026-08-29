"""Decision layer: what the model does with a request, and what it may return.

Implements the enforceable rules of `docs/MODEL_SPEC.md`. Every rule ID here
appears in that document; every ID marked `enforced` there appears here.

Two halves, deliberately separable:

  decide(prompt)            -> Decision      : pre-generation. Answer, clarify,
                                               warn, or refuse. Picks sampling.
  validate(text, decision)  -> Validation    : post-generation. Suppresses
                                               output that breaks a rule.

Both are pure functions over text, so they run without a model and are tested
that way in `eval/behavior_eval.py --policy-only` (which is also why they can
gate CI while the model itself is still broken).

`respond()` wires them to a real model: decide -> generate -> validate -> one
retry -> refuse.
"""

import ast
import hmac
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field

from data.scripts.clean import has_secret
from inference.identity import MODEL_NAME  # the model is named "wandaa"

ASSISTANT_NAME = MODEL_NAME  # re-exported so callers can `from inference.policy import ASSISTANT_NAME`

# ---------------------------------------------------------------------------
# request kinds and their decoding profiles (spec: "Decoding rules")
# ---------------------------------------------------------------------------
CODE_FIX = "code_fix"
CODE_COMPLETION = "code_completion"
SHELL_COMMAND = "shell_command"
CODE_EXPLAIN = "code_explain"
GENERAL_QA = "general_qa"


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_k: int | None
    top_p: float | None
    repetition_penalty: float
    max_new_tokens: int


SAMPLING = {
    # One right answer, and a wrong one is destructive -> greedy.
    CODE_FIX: Sampling(0.0, None, None, 1.0, 256),
    SHELL_COMMAND: Sampling(0.0, None, None, 1.0, 128),
    # Near-deterministic, with just enough slack to escape a repetition loop.
    CODE_COMPLETION: Sampling(0.2, 40, 0.95, 1.05, 192),
    # Prose wants variety; repetition penalty is safe here because there are no
    # legitimately repeated structural tokens to punish.
    CODE_EXPLAIN: Sampling(0.6, 50, 0.95, 1.1, 320),
    GENERAL_QA: Sampling(0.7, 50, 0.95, 1.1, 320),
}

# ---------------------------------------------------------------------------
# S1 -- destructive operations
# ---------------------------------------------------------------------------
# Unscoped: destroys data outside any path the user named. Never emitted.
DESTRUCTIVE_UNSCOPED = [
    (r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rR][a-zA-Z]*f?[a-zA-Z]*\s+(/|/\*|~|\$HOME)\s*$",
     "rm -rf on / or the home directory"),
    (r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*\s+(/|/\*|~)(\s|$)", "rm -f on / or the home directory"),
    (r"\bmkfs(\.\w+)?\b", "mkfs formats a filesystem"),
    (r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|disk)", "dd writing to a raw block device"),
    (r">\s*/dev/(sd|nvme|hd|disk)\w*", "redirect onto a raw block device"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:", "fork bomb"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\s+/\s*$", "chmod 777 on /"),
    (r"\bchown\s+-[a-zA-Z]*R[a-zA-Z]*\s+\S+\s+/\s*$", "recursive chown on /"),
    (r"\bDROP\s+DATABASE\b", "DROP DATABASE"),
    (r"\bgit\s+push\b[^\n]*--force[^\n]*\b(main|master)\b", "force-push to main"),
    (r"\bgit\s+push\b[^\n]*\b(main|master)\b[^\n]*--force", "force-push to main"),
]

# Scoped to a named relative path: allowed, with a warning.
DESTRUCTIVE_SCOPED = [
    (r"\brm\s+(-[a-zA-Z]+\s+)*\.?/?[\w.\-/]+", "recursive delete"),
    (r"\bTRUNCATE\s+TABLE\b", "TRUNCATE TABLE"),
    (r"\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)", "DELETE without WHERE"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard discards local changes"),
]

# ---------------------------------------------------------------------------
# S3 -- malicious tooling. Intent, not subject matter: the same words appear in
# legitimate security work, so AUTHORISED_CONTEXT re-permits those.
# ---------------------------------------------------------------------------
MALICIOUS = [
    (r"\bkeylogger\b", "keylogger"),
    (r"\bransomware\b", "ransomware"),
    (r"\b(steal|exfiltrate|harvest|dump)\s+(the\s+)?(password|credential|cookie|token|browser)",
     "credential theft"),
    (r"\bbotnet\b|\bc2\s+server\b|\bcommand[- ]and[- ]control\b", "botnet / C2"),
    (r"\bcrypto\s*(jack|miner)\b(?![^\n]*\bmy\s+own\b)", "cryptojacking"),
    (r"\bhack\s+(into\s+)?(my\s+)?(ex|neighbou?r|someone|their|his|her)\b",
     "unauthorised access to another person's system"),
    (r"\bbypass\s+(the\s+)?(login|auth\w*|paywall|licen[cs]e)\b", "auth bypass"),
]

AUTHORISED_CONTEXT = [
    r"\bctf\b", r"\bcapture[- ]the[- ]flag\b",
    r"\bpen(etration)?[- ]test", r"\bauthoris?zed\b", r"\bauthori[sz]ation\b",
    r"\bmy own\b", r"\bbug bounty\b", r"\bsecurity research\b",
    r"\bin scope\b", r"\bwritten permission\b",
]

# H3 -- a URL whose host was not in the prompt is ungrounded.
URL_RE = re.compile(r"https?://([\w.-]+)")

# P3 -- filler openers.
FILLER = [
    "great question", "certainly!", "sure thing", "i'd be happy to",
    "of course!", "absolutely!", "let me help you with that",
    "that's a great", "excellent question",
]

# C2 -- placeholder standing in for the substance.
ELLIPSIS_STUBS = [
    r"#\s*(rest of|remaining|the rest|implementation goes|your code|fill in)",
    r"//\s*(rest of|remaining|the rest|implementation goes|your code|fill in)",
    r"#\s*\.\.\.\s*$",
    r"\bTODO:\s*implement\b",
]

LANG_TAG_RE = re.compile(r"<language>(\w+)</language>")
LANG_WORDS = {
    "python": ["python", "py", "django", "flask", "pandas", "numpy", "pytest"],
    "typescript": ["typescript", "ts", "tsx", "react", "angular", "node", "interface"],
    "bash": ["bash", "shell", "sh", "zsh", "shellscript", "command line"],
}


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
ANSWER = "answer"
ANSWER_WITH_WARNING = "answer_with_warning"
CLARIFY = "clarify"
REFUSE = "refuse"


@dataclass
class Decision:
    action: str
    kind: str
    sampling: Sampling
    language: str | None = None
    reason: str = ""
    rules: list[str] = field(default_factory=list)
    message: str = ""     # the refusal or clarifying question, when applicable


@dataclass
class Violation:
    rule: str
    detail: str


@dataclass
class Validation:
    ok: bool
    violations: list[Violation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Admin session (spec: Tier 0 -- Admin)
#
# An authenticated admin bypasses the safety/quality gates. Two hard design
# rules make this an escape hatch rather than a hole:
#
#   1. It is NEVER triggered by prompt text. There is no magic word a user can
#      type to unlock it -- that would be a prompt-injection bypass anyone could
#      trip. It is unlocked only by presenting a token that matches the
#      out-of-band secret in SABYINYO_ADMIN_TOKEN, compared in constant time.
#   2. It is off by default. With no SABYINYO_ADMIN_TOKEN set in the
#      environment, admin can never engage, whatever token a caller presents.
#
# When engaged it still RUNS every check and logs what would have been blocked
# (report-only), so an admin session leaves an audit trail instead of a blind
# spot.
# ---------------------------------------------------------------------------
_audit = logging.getLogger("sabyinyo.policy.audit")


@dataclass(frozen=True)
class AdminSession:
    """Result of admin_session(token). `active` is True only for an
    authenticated admin; pass it into decide()/validate() to bypass the gates.
    """
    active: bool = False
    label: str = ""


def admin_session(token, *, label="admin"):
    """Return an active AdminSession iff `token` matches SABYINYO_ADMIN_TOKEN.

    Off unless the env secret is set. Constant-time compare so a caller cannot
    learn the secret by timing. Never reads the prompt.
    """
    import os

    secret = os.environ.get("SABYINYO_ADMIN_TOKEN")
    if not secret or not token:
        return AdminSession(active=False)
    if hmac.compare_digest(str(token), secret):
        _audit.warning("admin session ACTIVATED (label=%s): gates bypassed", label)
        return AdminSession(active=True, label=label)
    _audit.warning("admin activation REJECTED (label=%s): bad token", label)
    return AdminSession(active=False)


def _matches(patterns, text):
    """Return (pattern_description) for the first match, else None."""
    for pattern, description in patterns:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return description
    return None


def _has_authorised_context(text):
    return any(re.search(p, text, re.IGNORECASE) for p in AUTHORISED_CONTEXT)


def detect_language(prompt):
    """Corpus tag first, then keywords. None when nothing indicates a language."""
    tag = LANG_TAG_RE.search(prompt)
    if tag:
        return tag.group(1).lower()
    lowered = prompt.lower()
    for lang, words in LANG_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words):
            return lang
    return None


def classify(prompt):
    """Which decoding regime this request belongs to (spec: Decoding rules)."""
    lowered = prompt.lower()
    if re.search(r"\b(fix|bug|error|traceback|failing|broken|debug|why does)\b", lowered):
        return CODE_FIX
    if detect_language(prompt) == "bash" or re.search(r"\b(command|one-liner|shell)\b", lowered):
        return SHELL_COMMAND
    if re.search(r"\b(explain|what does|how does|why is|walk me through)\b", lowered):
        return CODE_EXPLAIN
    if re.search(r"(def |class |function |interface |import |=>|\{)", prompt):
        return CODE_COMPLETION
    if detect_language(prompt):
        return CODE_COMPLETION
    return GENERAL_QA


def is_ambiguous(prompt):
    """F3 -- ambiguous only when the ambiguity would change the answer.

    Code context disambiguates, so a prompt carrying code is never ambiguous
    here however terse it is.
    """
    stripped = prompt.strip()
    if re.search(r"(def |class |function |interface |import |=>|\{|\n)", stripped):
        return False
    return len(stripped.split()) < 4


def decide(prompt, admin=None):
    """Pre-generation decision. Pure function of (prompt, admin).

    An active AdminSession bypasses every gate: the request is answered as-is.
    Admin is authenticated out-of-band (see admin_session); prompt text can
    never activate it.
    """
    kind = classify(prompt)
    sampling = SAMPLING[kind]
    language = detect_language(prompt)

    if admin is not None and admin.active:
        _audit.warning("admin decide() bypass (label=%s): %r", admin.label, prompt[:120])
        return Decision(ANSWER, kind, sampling, language,
                        reason="Tier 0: admin session, gates bypassed",
                        rules=["ADMIN"])

    # I1 -- identity is answered deterministically, never left to the model to
    # hallucinate. The model is named "wandaa".
    if re.search(r"\b(your name|who are you|what are you called|what.s your name)\b",
                 prompt, re.IGNORECASE):
        from inference.identity import IDENTITY_LINE
        return Decision(ANSWER, kind, sampling, language,
                        reason="I1: identity question", rules=["I1"],
                        message=IDENTITY_LINE)

    # Tier 1 first, always.
    malicious = _matches(MALICIOUS, prompt)
    if malicious and not _has_authorised_context(prompt):
        return Decision(
            REFUSE, kind, sampling, language,
            reason=f"S3: request is for {malicious}",
            rules=["S3"],
            message=(
                f"I can't help build {malicious}. If this is authorised security "
                f"work (a CTF, a pentest with permission, your own system), say so "
                f"and I can help with the underlying technique."
            ),
        )

    unscoped = _matches(DESTRUCTIVE_UNSCOPED, prompt)
    if unscoped:
        return Decision(
            REFUSE, kind, sampling, language,
            reason=f"S1: unscoped destructive operation ({unscoped})",
            rules=["S1"],
            message=(
                f"I won't produce that: {unscoped}. It destroys data outside any "
                f"path you named. Tell me the specific directory or table you mean "
                f"and I'll scope it."
            ),
        )

    scoped = _matches(DESTRUCTIVE_SCOPED, prompt)
    if scoped:
        return Decision(
            ANSWER_WITH_WARNING, kind, sampling, language,
            reason=f"S1: scoped destructive operation ({scoped})",
            rules=["S1"],
            message=f"Note: this is destructive and not undoable ({scoped}).",
        )

    if is_ambiguous(prompt):
        return Decision(
            CLARIFY, kind, sampling, language,
            reason="F3: under-specified and the readings diverge",
            rules=["F3"],
            message="Which language and what should it do? A line of context is enough.",
        )

    return Decision(ANSWER, kind, sampling, language, reason="no rule triggered")


# ---------------------------------------------------------------------------
# post-generation validation
# ---------------------------------------------------------------------------
def _python_parses(code):
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"line {e.lineno}: {e.msg}"


def _bash_parses(code):
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["bash", "-n", path], capture_output=True, text=True, timeout=10)
        return r.returncode == 0, r.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        # No bash available -- report unknown rather than a false failure.
        return True, f"skipped: {e}"


CODE_START_RE = re.compile(
    r"^\s*(#!/|from\s+\w|import\s+\w|def\s+\w|class\s+\w|@\w|"
    r"(export\s+)?(async\s+)?function\s+\w|(export\s+)?(interface|type|const|let|var)\s+\w)"
)


def extract_code(text, language):
    """Return the part of `text` that is code, or "" if it cannot be delimited.

    Fenced blocks are unambiguous, so they are preferred. Unfenced text counts
    as code only when it *begins* like code -- checking merely that it contains
    "def " somewhere would classify an English sentence mentioning a function as
    Python and fail C1 on every prose answer.

    The cost of being conservative: an unfenced answer that mixes prose and code
    is not syntax-checked, because there is no reliable way to find where the
    code starts. Prompting the model to fence its code buys the check back.
    """
    fences = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if fences:
        return "\n".join(fences)
    if language and CODE_START_RE.match(text):
        return text
    return ""


def validate(text, decision, prompt="", admin=None):
    """Post-generation checks. Returns every violation, not just the first.

    Under an active AdminSession the checks still RUN (so the audit log records
    what would have been blocked), but the result is reported ok=True: the
    output is not suppressed.
    """
    violations = []

    # S2 -- secrets. Reuses data/scripts/clean.py so there is one definition.
    if has_secret(text):
        violations.append(Violation("S2", "output contains a credential-shaped string"))

    # S1 -- the model may not emit a destructive command even if the prompt was clean.
    unscoped = _matches(DESTRUCTIVE_UNSCOPED, text)
    if unscoped:
        violations.append(Violation("S1", f"output contains {unscoped}"))

    # H3 -- URLs to hosts that were not in the prompt are ungrounded.
    prompt_hosts = set(URL_RE.findall(prompt))
    for host in set(URL_RE.findall(text)) - prompt_hosts:
        violations.append(Violation("H3", f"ungrounded URL host: {host}"))

    # C1 -- generated code must parse.
    code = extract_code(text, decision.language)
    if code.strip():
        if decision.language == "python":
            ok, detail = _python_parses(code)
            if not ok:
                violations.append(Violation("C1", f"python does not parse: {detail}"))
        elif decision.language == "bash":
            ok, detail = _bash_parses(code)
            if not ok:
                violations.append(Violation("C1", f"bash does not parse: {detail}"))

        # C2 -- placeholder where the substance belongs.
        for pattern in ELLIPSIS_STUBS:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                violations.append(Violation("C2", "code elides the requested implementation"))
                break

    # F1 -- answered in the wrong language.
    if decision.language:
        emitted = LANG_TAG_RE.search(text)
        if emitted and emitted.group(1).lower() != decision.language:
            violations.append(
                Violation("F1", f"asked for {decision.language}, emitted {emitted.group(1)}")
            )

    # P3 -- filler opener.
    opener = text.lstrip().lower()[:40]
    for phrase in FILLER:
        if opener.startswith(phrase):
            violations.append(Violation("P3", f"filler opener: {phrase!r}"))
            break

    if admin is not None and admin.active:
        if violations:
            _audit.warning("admin validate() bypass (label=%s): would-block %s",
                           admin.label, [v.rule for v in violations])
        return Validation(ok=True, violations=violations)

    return Validation(ok=not violations, violations=violations)


# ---------------------------------------------------------------------------
# wiring it to a model
# ---------------------------------------------------------------------------
@dataclass
class Response:
    action: str
    text: str
    decision: Decision
    validation: Validation | None = None
    attempts: int = 0


def respond(model, tokenizer, prompt, device="cpu", generate_fn=None, max_attempts=2,
            admin=None):
    """decide -> generate -> validate -> retry once -> refuse.

    `generate_fn` defaults to eval.harness.generate; inject a fake for tests.
    Pass an active AdminSession (from admin_session) to bypass the gates.
    """
    if generate_fn is None:
        from eval.harness import generate as generate_fn

    decision = decide(prompt, admin=admin)
    if decision.action in (REFUSE, CLARIFY):
        return Response(decision.action, decision.message, decision)

    s = decision.sampling
    last = None
    for attempt in range(1, max_attempts + 1):
        out = generate_fn(
            model, tokenizer, prompt,
            max_new_tokens=s.max_new_tokens,
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            repetition_penalty=s.repetition_penalty,
            device=device,
            seed=attempt,
        )
        text = out["completion"]
        last = validate(text, decision, prompt, admin=admin)
        if last.ok:
            body = f"{decision.message}\n\n{text}" if decision.message else text
            return Response(decision.action, body, decision, last, attempt)

        # Greedy decoding is deterministic -- a retry would return the same
        # text, so do not spend one.
        if s.temperature == 0.0:
            break

    detail = "; ".join(f"{v.rule}: {v.detail}" for v in last.violations)
    return Response(
        REFUSE,
        f"I could not produce an answer that passes the output checks ({detail}).",
        decision, last, max_attempts,
    )
