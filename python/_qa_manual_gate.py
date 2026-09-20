#!/usr/bin/env python3
"""_qa_manual_gate.py - the verification gate for MANUAL.md (the shippable
Python manual at the repo root). Same house convention as every other
feature in this repo: nothing ships without a gate (see python/qa_matrix.py,
src/*_test.cpp) - a hand-written manual documenting a Python API is no
exception, and is exactly the kind of doc that silently rots as the API
moves under it if nothing checks it.

Two gates, structured as ONE file with two modes so extraction logic (what
counts as a "code block", what "attribute reference" means) is never
duplicated between a host-side script and a container-side script:

  (a) SYNTAX GATE - always runs, host-side, no `pycamtrt` import required.
      Extracts every ```python fenced block from MANUAL.md and
      `py_compile`s it standalone. A snippet is allowed to be a FRAGMENT
      (referencing names defined earlier in its own example file, e.g.
      `detect`/`DETECT_ENGINE`) - that is a runtime NameError, not a
      SyntaxError, and this gate is syntax-only by design (matching the
      task's own "(a) ... py_compile each (syntax gate)" wording). A block
      that is a Python-console TRANSCRIPT rather than runnable module code
      (`>>> `/`...` prompts) is fenced ```pycon in MANUAL.md specifically
      so it is EXCLUDED from this gate rather than force-fit into invalid
      syntax - see MANUAL.md's recommend() example.

  (b) API GATE (--check-api) - must run where `import pycamtrt` succeeds
      (inside the tensorrt-dev container - see this file's own docstring
      example invocation below). Collects every attribute referenced off
      the `pycamtrt` module in those SAME code blocks - `pycamtrt.X`
      attribute access, and names imported via `from pycamtrt import A, B`
      - and verifies each actually exists on the real, imported module via
      `getattr`. This is the gate that catches a manual documenting an
      attribute that doesn't (or no longer) exists - the exact failure mode
      a hand-written manual is most at risk of as `__init__.py` evolves.

Usage:
    # host-side, no pycamtrt needed - syntax gate only:
    python3 python/_qa_manual_gate.py MANUAL.md

    # inside the tensorrt-dev container - syntax gate + API gate:
    docker run --rm --gpus all --network host \\
      -v $(pwd):/workspace -w /workspace \\
      tensorrt-dev bash -c "
        ln -sf libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so
        PYTHONPATH=/workspace/build:/workspace/python \\
        python3 \\
        python/_qa_manual_gate.py --check-api MANUAL.md
      "

Exit code 0 iff every requested gate passed (non-zero, with FAIL/MISSING
lines printed, otherwise) - fit for a CI/pre-commit check, not just manual
inspection.
"""
from __future__ import annotations

import argparse
import ast
import py_compile
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Set, Tuple

# Deliberately a plain regex, not a markdown parser: the house convention is
# plain ```python fences (see MANUAL.md's STYLE section), and matching the
# manual's own surrounding prose is not a case this needs to handle.
_PY_FENCE_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def extract_python_blocks(manual_text: str) -> List[str]:
    """Every ```python ... ``` fenced block in MANUAL.md, in document
    order (```pycon/```json/etc. blocks are deliberately NOT matched -
    see the module docstring's note on the one ```pycon transcript).
    """
    return [block.strip("\n") for block in _PY_FENCE_RE.findall(manual_text)]


def check_syntax(blocks: List[str]) -> Tuple[int, List[str]]:
    """py_compile every block as a standalone module (a temp .py file, not
    exec() - this gate never RUNS a snippet, only parses/compiles it, so a
    fragment referencing an undefined name is fine here). Returns
    (n_ok, failure_messages).
    """
    failures: List[str] = []
    for i, block in enumerate(blocks):
        fd_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".py", delete=False) as f:
                f.write(block)
                fd_path = f.name
            py_compile.compile(fd_path, doraise=True)
        except py_compile.PyCompileError as e:
            failures.append(f"block {i}: {e}")
        finally:
            if fd_path is not None:
                Path(fd_path).unlink(missing_ok=True)
    return len(blocks) - len(failures), failures


def collect_pycamtrt_attrs(blocks: List[str]) -> Set[str]:
    """Every name referenced as `pycamtrt.<name>` (an ast.Attribute whose
    value is the bare Name `pycamtrt`), plus every name imported via
    `from pycamtrt import A, B, ...`, across every block that parses.
    AST-based rather than regex so a multi-line attribute chain or call
    is handled correctly; a block that fails to parse is silently skipped
    here (check_syntax() above already reports it - the API gate should
    not double-report or crash on a genuine fragment).
    """
    attrs: Set[str] = set()
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pycamtrt":
                for alias in node.names:
                    attrs.add(alias.name)
            elif (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "pycamtrt"):
                attrs.add(node.attr)
    return attrs


def check_api(attrs: Set[str]) -> Tuple[List[str], List[str]]:
    """Verify every collected attribute name exists on the REAL, imported
    `pycamtrt` module via getattr - must run somewhere `import pycamtrt`
    actually succeeds (see this file's module docstring for the exact
    tensorrt-dev container invocation). Returns (found, missing), both
    sorted.
    """
    import pycamtrt  # local import: only this mode needs it to succeed
    found: List[str] = []
    missing: List[str] = []
    for name in sorted(attrs):
        (found if hasattr(pycamtrt, name) else missing).append(name)
    return found, missing


def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manual", help="path to MANUAL.md")
    p.add_argument("--check-api", action="store_true",
                   help="also run the API gate (needs a real `import "
                        "pycamtrt` on the path - run inside the "
                        "tensorrt-dev container, see this file's module "
                        "docstring for the exact invocation)")
    args = p.parse_args(argv)

    text = Path(args.manual).read_text()
    blocks = extract_python_blocks(text)
    print(f"[extract] {len(blocks)} ```python block(s) found in {args.manual}")

    n_ok, failures = check_syntax(blocks)
    print(f"[syntax]  {n_ok}/{len(blocks)} block(s) compile clean")
    for msg in failures:
        print(f"  FAIL {msg}")

    ok = not failures

    if args.check_api:
        attrs = collect_pycamtrt_attrs(blocks)
        print(f"[api]     {len(attrs)} distinct pycamtrt.<attr> reference(s) "
              f"collected: {sorted(attrs)}")
        found, missing = check_api(attrs)
        print(f"[api]     {len(found)}/{len(attrs)} attribute(s) verified "
              f"via real import + getattr")
        for name in missing:
            print(f"  MISSING pycamtrt.{name}")
        ok = ok and not missing
    else:
        print("[api]     skipped (pass --check-api inside the tensorrt-dev "
              "container to run it)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
