#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the bounded text parts of the full daily CSV export."""

from pathlib import Path
import re
import shutil
import tempfile


PART_NAME = re.compile(r"standard-[0-9]{3}\.csv\Z")
HEADER = b"plan,suite,name,check,cmd\n"


def materialize_full_export(directory, output):
    directory = Path(directory)
    output = Path(output)
    parts = sorted((directory / "parts").glob("standard-*.csv"))
    if not parts or any(not PART_NAME.fullmatch(path.name) for path in parts):
        raise ValueError("Full daily export requires named standard CSV parts")
    if output.exists():
        raise ValueError("Materialized daily export already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for path in directory.iterdir():
            if path.name in ("parts", "standard.csv"):
                continue
            if path.is_symlink():
                raise ValueError("Daily export cannot contain symlinks")
            if path.is_dir():
                shutil.copytree(path, temporary / path.name)
            else:
                shutil.copyfile(path, temporary / path.name)
        with (temporary / "standard.csv").open("wb") as merged:
            for index, path in enumerate(parts):
                content = path.read_bytes()
                if (not content.endswith(b"\n") or (index and content.startswith(HEADER))
                        or (not index and not content.startswith(HEADER))):
                    raise ValueError("Daily CSV part is truncated or repeats the header")
                merged.write(content)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output
