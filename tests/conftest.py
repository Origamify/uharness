"""Pytest config: make the repo root importable; tests may only touch the pytest tmp dir."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from uharness import path_permissions


@pytest.fixture(autouse=True)
def _allow_pytest_tmp(tmp_path_factory):
    """Grant the whole pytest tmp base dir as the only allowed root."""
    path_permissions.set_allowed_roots([str(tmp_path_factory.getbasetemp())])
