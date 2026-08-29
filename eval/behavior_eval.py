"""Score the model and its decision layer against docs/MODEL_SPEC.md.

    # deterministic, no checkpoint needed -- this is the CI gate
    PYTHONPATH=. python eval/behavior_eval.py --policy-only

    # also generate with a real checkpoint and validate what comes back
    PYTHONPATH=. python eval/behavior_eval.py --data-dir /content/data --checkpoint latest

Cases live in eval/behavior_cases.jsonl, one JSON object per line:

  phase="decide"    prompt + expect_action / expect_rules / expect_kind /
                    expect_language / expect_temperature
  phase="validate"  output + language + expect_violations
  phase="admin"     env_secret + token + prompt + expect_active + expect_action

Reports a pass rate per spec rule, so a regression names the rule it broke.
Exits non-zero if any case fails.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

from inference import policy

CASES_PATH = os.path.join(os.path.dirname(__file__), "behavior_cases.jsonl")


def load_cases(path=CASES_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_decide(case):
    """Run the pre-generation decision and compare against expectations."""
    d = policy.decide(case["prompt"])
    problems = []

    if "expect_action" in case and d.action != case["expect_action"]:
        problems.append(f"action={d.action!r} expected {case['expect_action']!r}")

    for rule in case.get("expect_rules", []):
        if rule not in d.rules:
            problems.append(f"rule {rule} did not fire (fired: {d.rules or 'none'})")
    if case.get("expect_rules") == [] and d.rules:
        problems.append(f"unexpected rules fired: {d.rules}")

    if "expect_kind" in case and d.kind != case["expect_kind"]:
        problems.append(f"kind={d.kind!r} expected {case['expect_kind']!r}")

    if "expect_language" in case and d.language != case["expect_language"]:
        problems.append(f"language={d.language!r} expected {case['expect_language']!r}")

    if "expect_temperature" in case:
        got = d.sampling.temperature
        if got != case["expect_temperature"]:
            problems.append(f"temperature={got} expected {case['expect_temperature']}")

    return problems, d.rules


def check_validate(case):
    """Run the post-generation checks against a fixed output string."""
    decision = policy.Decision(
        action=policy.ANSWER,
        kind=policy.CODE_COMPLETION,
        sampling=policy.SAMPLING[policy.CODE_COMPLETION],
        language=case.get("language"),
    )
    v = policy.validate(case["output"], decision, case.get("prompt", ""))
    got = sorted({x.rule for x in v.violations})
    want = sorted(case.get("expect_violations", []))
    problems = []
    if got != want:
        detail = "; ".join(f"{x.rule}: {x.detail}" for x in v.violations) or "none"
        problems.append(f"violations={got} expected {want} ({detail})")
    return problems, got


def check_admin(case):
    """Verify Tier 0 admin behaviour under a controlled env secret.

    Each admin case sets SABYINYO_ADMIN_TOKEN to a known value (or clears it),
    presents `token`, and asserts whether the gate is bypassed. This is how we
    prove the security properties: off by default, prompt text never activates
    it, wrong token stays locked, right token bypasses decide() and validate().
    """
    import os

    prev = os.environ.get("SABYINYO_ADMIN_TOKEN")
    try:
        if case.get("env_secret") is None:
            os.environ.pop("SABYINYO_ADMIN_TOKEN", None)
        else:
            os.environ["SABYINYO_ADMIN_TOKEN"] = case["env_secret"]

        admin = policy.admin_session(case.get("token", ""))
        problems = []
        if admin.active != case["expect_active"]:
            problems.append(f"admin.active={admin.active} expected {case['expect_active']}")

        d = policy.decide(case["prompt"], admin=admin)
        if d.action != case["expect_action"]:
            problems.append(f"action={d.action!r} expected {case['expect_action']!r}")
        return problems, ["ADMIN"]
    finally:
        if prev is None:
            os.environ.pop("SABYINYO_ADMIN_TOKEN", None)
        else:
            os.environ["SABYINYO_ADMIN_TOKEN"] = prev


def rules_for(case):
    """Which spec rules this case exercises, for the per-rule report."""
    return case.get("expect_rules") or case.get("expect_violations") or ["(no-rule)"]


def run_policy_suite(cases, verbose=False):
    per_rule = defaultdict(lambda: [0, 0])   # rule -> [passed, total]
    failures = []

    for case in cases:
        if case["phase"] == "decide":
            problems, _ = check_decide(case)
        elif case["phase"] == "validate":
            problems, _ = check_validate(case)
        elif case["phase"] == "admin":
            problems, _ = check_admin(case)
        else:
            problems = [f"unknown phase {case['phase']!r}"]

        for rule in rules_for(case):
            per_rule[rule][1] += 1
            if not problems:
                per_rule[rule][0] += 1

        if problems:
            failures.append((case["id"], problems))
        elif verbose:
            print(f"  PASS  {case['id']}")

    return per_rule, failures


def run_model_suite(cases, model, tokenizer, device):
    """Generate for every `decide` case that should be answered, then validate.

    This measures the model, not the policy: how often real output clears the
    enforced output rules. On a broken checkpoint expect a low number -- that is
    the point of having it.
    """
    checked = 0
    clean = 0
    per_rule = defaultdict(int)

    for case in cases:
        if case["phase"] != "decide":
            continue
        if case.get("expect_action") not in (policy.ANSWER, policy.ANSWER_WITH_WARNING):
            continue

        r = policy.respond(model, tokenizer, case["prompt"], device=device)
        checked += 1
        if r.validation is None:
            continue
        if r.validation.ok:
            clean += 1
        else:
            for v in r.validation.violations:
                per_rule[v.rule] += 1

    return checked, clean, per_rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-only", action="store_true",
                    help="run the deterministic policy suite; no model needed")
    ap.add_argument("--cases", default=CASES_PATH)
    ap.add_argument("--data-dir", default="/content/data")
    ap.add_argument("--checkpoint", default="latest")
    ap.add_argument("--device", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.cases)

    print("=" * 74)
    print("SPEC COMPLIANCE  (docs/MODEL_SPEC.md)")
    print("=" * 74)
    print(f"\n[1/2] Decision layer -- {len(cases)} cases, no model involved")

    per_rule, failures = run_policy_suite(cases, args.verbose)

    print(f"\n  {'rule':<12}{'passed':>10}{'total':>8}")
    print("  " + "-" * 30)
    for rule in sorted(per_rule):
        passed, total = per_rule[rule]
        flag = "" if passed == total else "   <-- FAIL"
        print(f"  {rule:<12}{passed:>10}{total:>8}{flag}")

    if failures:
        print(f"\n  {len(failures)} failing case(s):")
        for case_id, problems in failures:
            print(f"    {case_id}")
            for p in problems:
                print(f"      - {p}")
    else:
        print(f"\n  all {len(cases)} cases pass")

    if args.policy_only:
        print("\n" + "=" * 74)
        return 1 if failures else 0

    # --- with a model -------------------------------------------------------
    print("\n[2/2] Model output against the enforced output rules")
    import torch

    from eval.harness import load_model, load_tokenizer
    from eval.run_eval import hf_token, resolve_checkpoint

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.data_dir)
    ckpt = resolve_checkpoint(args.checkpoint, hf_token(), policy_repo_id())
    model, meta = load_model(ckpt, device)
    print(f"  checkpoint step={meta['step']}  device={device}")

    checked, clean, rule_counts = run_model_suite(cases, model, tokenizer, device)
    rate = clean / checked if checked else 0.0
    print(f"\n  {clean}/{checked} generations passed every output check ({rate:.0%})")
    if rule_counts:
        print("  violations by rule:")
        for rule in sorted(rule_counts):
            print(f"    {rule}: {rule_counts[rule]}")

    print("\n" + "=" * 74)
    return 1 if failures else 0


def policy_repo_id():
    from eval.harness import HF_REPO_ID

    return HF_REPO_ID


if __name__ == "__main__":
    sys.exit(main())
