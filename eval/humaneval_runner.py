def run_tests(completion, test_code):
    try:
        namespace = {}
        exec(completion, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False


def run_pass_at_k(model, problems, generate_fn, k=5):
    results = []
    for problem in problems:
        completions = [generate_fn(model, problem["prompt"]) for _ in range(k)]
        passed = [run_tests(c, problem["test"]) for c in completions]
        results.append(any(passed))
    return sum(results) / len(results) if results else 0.0


if __name__ == "__main__":
    pass
