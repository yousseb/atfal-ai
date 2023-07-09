import os
from pathlib import Path
import pytest


def pytest_sessionstart(session):
    cwd = Path(os.getcwd())
    if (list(cwd.parts)[-1:][0]).lower() == 'tests':
        os.chdir(Path('..').absolute())
