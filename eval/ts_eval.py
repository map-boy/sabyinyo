import subprocess


def check_ts_syntax(file_path):
    result = subprocess.run(
        ["npx", "tsc", "--noEmit", file_path], capture_output=True, text=True
    )
    return result.returncode == 0


if __name__ == "__main__":
    pass
