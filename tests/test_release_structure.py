import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseStructureTests(unittest.TestCase):
    def test_required_directories_exist(self):
        for name in (
            "models",
            "training",
            "evaluation",
            "data_tools",
            "scripts",
            "weights",
        ):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_dir(), name)

    def test_release_contains_no_checkpoint_binaries(self):
        checkpoint_files = [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}
        ]
        self.assertEqual(checkpoint_files, [])


if __name__ == "__main__":
    unittest.main()
