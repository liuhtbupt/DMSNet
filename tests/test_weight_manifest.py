import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WeightManifestTests(unittest.TestCase):
    def test_manifest_describes_all_three_modules(self):
        manifest_path = ROOT / "weights" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["checkpoints"]
        self.assertEqual(
            {entry["module"] for entry in entries},
            {"countnet", "paramnet", "refiner"},
        )
        for entry in entries:
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size_bytes"], 0)
            self.assertFalse((ROOT / "weights" / entry["filename"]).exists())


if __name__ == "__main__":
    unittest.main()
