import ast
from pathlib import Path
import re


def load_clean_name():
    """Load clean_name function from refresh.py without executing the module."""
    refresh_path = Path(__file__).resolve().parents[1] / "refresh.py"
    source = refresh_path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "clean_name":
            code = ast.get_source_segment(source, node)
            break
    else:
        raise AssertionError("clean_name not found")
    ns = {"re": re}
    exec(code, ns)
    return ns["clean_name"]

clean_name = load_clean_name()

def test_parenthetical():
    assert clean_name("LeBron James (LAL)") == "lebron james"

def test_period_initial():
    assert clean_name("J. Harden") == "j harden"

def test_punctuation():
    assert clean_name("D'Angelo Russell!") == "dangelo russell"
