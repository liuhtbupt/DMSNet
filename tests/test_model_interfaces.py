import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ModelInterfaceTests(unittest.TestCase):
    def test_public_model_classes_import(self):
        from models import (
            AxisHeatmapPatchRefinerNet,
            DualBandCrossFusionCountNet,
            JointDetectionParamEstimationNet,
            VelocityEnhancedParamNet,
        )

        for model_class in (
            DualBandCrossFusionCountNet,
            VelocityEnhancedParamNet,
            AxisHeatmapPatchRefinerNet,
            JointDetectionParamEstimationNet,
        ):
            self.assertTrue(callable(model_class))

    def test_joint_model_can_skip_checkpoint_loading(self):
        from models import JointDetectionParamEstimationNet

        model = JointDetectionParamEstimationNet(load_weights=False)
        self.assertEqual(model.Kmax, 5)
        self.assertEqual(tuple(model.refiner.feat_size), (256, 256, 256))


if __name__ == "__main__":
    unittest.main()
