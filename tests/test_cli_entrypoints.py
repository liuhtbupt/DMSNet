import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliEntrypointTests(unittest.TestCase):
    def test_public_tools_expose_help(self):
        scripts = (
            ROOT / "data_tools" / "convert_sionna_npz_to_h5.py",
            ROOT / "evaluation" / "evaluate_count_runtime.py",
            ROOT / "evaluation" / "evaluate_parameter_cdf.py",
        )
        for script in scripts:
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
