import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("models", "training", "evaluation", "data_tools", "scripts")


class PortabilityTests(unittest.TestCase):
    def test_code_contains_no_machine_specific_absolute_paths(self):
        violations = []
        for dirname in SCANNED_DIRS:
            for path in (ROOT / dirname).rglob("*"):
                if not path.is_file() or path.suffix.lower() not in {".py", ".ps1", ".md"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if any(
                    marker in text
                    for marker in (
                        "D:" + "\\document\\",
                        "C:" + "\\Users\\",
                        "\u65b9\u6848B",
                        "\u76ee\u524d\u6700\u597d",
                    )
                ):
                    violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
