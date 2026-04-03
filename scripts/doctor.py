#!/usr/bin/env python3
"""Environment checker for running brain-learn on WSL or Windows."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from pathlib import Path

REQUIRED_MODULES = [
    "dill",
    "dotenv",
    "pandas",
    "requests",
    "sympy",
]


def is_wsl() -> bool:
    release = platform.release().lower()
    version = platform.version().lower()
    return "microsoft" in release or "microsoft" in version


def windows_repo_hint(repo_root: Path) -> str:
    path = str(repo_root)
    if path.startswith("/mnt/") and len(path) > 6:
        drive = path[5].upper()
        remainder = path[6:].replace("/", "\\")
        return f"{drive}:{remainder}"
    return str(repo_root)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def read_env_keys(env_path: Path) -> set[str]:
    keys: set[str] = set()
    if not env_path.exists():
        return keys

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether brain-learn is ready to run.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "wsl", "windows"),
        default="auto",
        help="Target runtime environment.",
    )
    parser.add_argument(
        "--require-env",
        action="store_true",
        help="Treat a missing or incomplete .env file as an error.",
    )
    parser.add_argument(
        "--require-deps",
        action="store_true",
        help="Treat missing Python packages as an error.",
    )
    parser.add_argument(
        "--check-auth",
        action="store_true",
        help="Attempt a real authentication request with the credentials in .env.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dotenv import load_dotenv
    from src.utils import create_authenticated_session

    env_path = repo_root / ".env"
    example_env_path = repo_root / ".env.example"
    initial_population_path = repo_root / "initial-population.pkl"
    current_dir = Path.cwd().resolve()

    errors: list[str] = []
    warnings: list[str] = []

    mode = args.mode
    if mode == "auto":
        mode = "wsl" if is_wsl() else "windows" if os.name == "nt" else "wsl"

    print("brain-learn doctor")
    print(f"- Repo root: {repo_root}")
    print(f"- Current dir: {current_dir}")
    print(f"- Mode: {mode}")
    print(f"- Python: {sys.executable}")
    print(f"- Python version: {platform.python_version()}")

    if not (repo_root / "main.py").exists():
        errors.append("Missing main.py in repo root.")

    if current_dir != repo_root:
        warnings.append(f"You're not in the repo root. Use: cd {repo_root}")

    if sys.version_info < (3, 12):
        errors.append("Python 3.12+ is required by pyproject.toml.")

    if mode == "wsl":
        if os.name != "posix":
            errors.append("WSL mode expects a Linux shell, not PowerShell or cmd.")
        elif not is_wsl():
            warnings.append("This shell does not look like WSL. If you're on plain Linux, that's still fine.")
        print(f"- WSL cd command: cd {repo_root}")
    else:
        windows_hint = windows_repo_hint(repo_root)
        print(f"- PowerShell cd command: cd {windows_hint}")
        if os.name != "nt":
            warnings.append("Windows mode is mainly for PowerShell usage on Windows.")

    missing_modules = [name for name in REQUIRED_MODULES if not module_available(name)]
    if missing_modules:
        joined = ", ".join(missing_modules)
        warnings.append(f"Missing Python modules: {joined}")
        if mode == "windows":
            print("- Install deps: py -m pip install dill pandas python-dotenv requests sympy")
        else:
            print("- Install deps: python3 -m pip install --user --break-system-packages dill pandas python-dotenv requests sympy")
        if args.require_deps:
            errors.append("Required Python dependencies are not installed.")
    else:
        print("- Python dependencies: OK")

    env_keys = read_env_keys(env_path)
    missing_env_keys = [key for key in ("USERNAME", "PASSWORD") if key not in env_keys]
    if env_path.exists():
        print("- .env file: found")
    else:
        warnings.append(f"Missing .env file. Copy {example_env_path.name} to .env and fill in credentials.")

    if missing_env_keys:
        warnings.append(f".env is missing keys: {', '.join(missing_env_keys)}")
        if args.require_env:
            errors.append("WorldQuant credentials are not configured.")
    elif env_path.exists():
        print("- .env keys: OK")

    if example_env_path.exists():
        print(f"- Example env: {example_env_path.name}")

    if initial_population_path.exists() and initial_population_path.stat().st_size == 0:
        warnings.append("initial-population.pkl is empty. This is safe to ignore because seeds are optional.")

    if args.check_auth and not missing_env_keys and not missing_modules:
        load_dotenv(env_path)
        username = os.getenv("USERNAME", "")
        password = os.getenv("PASSWORD", "")
        session, response = create_authenticated_session(
            username,
            password,
            context="Doctor authentication check",
        )
        if session is not None:
            print("- Authentication check: OK")
        else:
            status = response.status_code if response is not None else "no-response"
            warnings.append(f"Authentication check failed with status: {status}")
            if response is not None and response.status_code == 401:
                errors.append(
                    "WorldQuant Brain rejected the current credentials. "
                    "Verify the exact Brain login username and password."
                )
            elif response is not None and response.status_code == 429:
                errors.append(
                    "Authentication is being rate-limited by WorldQuant Brain. "
                    "Wait a few minutes before trying again."
                )
            else:
                errors.append(
                    "Authentication check failed. Inspect the response from the login endpoint."
                )
            if response is not None:
                print(f"- Authentication status: {response.status_code}")
                print(f"- Authentication response: {response.text}")

    print("- Next run command:")
    if mode == "windows":
        print("  py main.py")
    else:
        print("  python3 main.py")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")

    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"- {item}")
        return 1

    print("\nEnvironment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
