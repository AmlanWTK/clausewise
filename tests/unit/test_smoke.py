"""Smoke tests: prove the package imports and CI genuinely runs pytest."""

import clausewise


def test_package_imports() -> None:
    assert clausewise.__version__ == "0.1.0"
