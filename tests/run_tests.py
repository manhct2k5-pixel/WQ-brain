"""
Test runner script for brain-learn.

This runner supports both:
- unittest.TestCase-based tests discovered from each module
- top-level test_* functions that are not wrapped in TestCase classes
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import unittest


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _test_modules() -> list[str]:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    return sorted(
        filename[:-3]
        for filename in os.listdir(test_dir)
        if filename.startswith("test_") and filename.endswith(".py")
    )


def _load_module_suite(module) -> unittest.TestSuite:
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)

    for name, func in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if func.__module__ != module.__name__:
            continue
        suite.addTest(unittest.FunctionTestCase(func))
    return suite


def _failed_import_case(module_name: str, exc: Exception) -> unittest.TestCase:
    def _raise_import_error() -> None:
        raise exc

    _raise_import_error.__name__ = f"failed_import_{module_name}"
    return unittest.FunctionTestCase(_raise_import_error)


def run_all_tests() -> bool:
    repo_root = _repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    module_names = _test_modules()
    print(f"Found {len(module_names)} test files: {', '.join(module_names)}")

    master_suite = unittest.TestSuite()
    for module_name in module_names:
        print(f"Loading tests from {module_name}...")
        try:
            module = importlib.import_module(f"tests.{module_name}")
        except Exception as exc:
            master_suite.addTest(_failed_import_case(module_name, exc))
            continue
        master_suite.addTests(_load_module_suite(module))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(master_suite)
    print(f"\nTest Summary: {result.testsRun} run, {len(result.failures)} failed, {len(result.errors)} errors")
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
