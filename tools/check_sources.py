"""Validate recovered Python syntax and clean notebook structure without executing it."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []
python_count = notebook_count = 0
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    try:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf8"), filename=str(path))
            python_count += 1
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf8"))
            assert notebook.get("nbformat") == 4, "unsupported notebook format"
            assert isinstance(notebook.get("cells"), list), "missing cells"
            for cell in notebook["cells"]:
                assert not cell.get("attachments"), "embedded attachment retained"
                if cell.get("cell_type") == "code":
                    assert cell.get("outputs") == [], "saved output retained"
                    assert cell.get("execution_count") is None, "execution count retained"
            notebook_count += 1
    except Exception as exc:
        errors.append(f"{path.relative_to(root)}: {exc}")
for error in errors:
    print(error)
print(f"Python files checked: {python_count}; notebooks checked: {notebook_count}; errors: {len(errors)}")
raise SystemExit(bool(errors))
