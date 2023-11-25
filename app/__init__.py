# -*- coding: utf-8 -*-

import sys
import pathlib

sys.path.append(str(pathlib.Path(".")))
sys.path.append(str(pathlib.Path("/app")))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "app"))
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "app" / "networks" / "cfrgan"))
