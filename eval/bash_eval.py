import subprocess


def check_bash_syntax(file_path):
    result = subprocess.run(
        ["bash", "-n", file_path], capture_output=True, text=True
    )
    return result.returncode == 0


if __name__ == "__main__":
    pass
