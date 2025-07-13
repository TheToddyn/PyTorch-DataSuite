# test_pipeline.py
"""
Testes automáticos para o pipeline de IA.
"""
import unittest
from model import get_model

class TestModel(unittest.TestCase):
    def test_resnet(self):
        model = get_model("resnet18", 10)
        self.assertIsNotNone(model)
    def test_efficientnet(self):
        model = get_model("efficientnet_b0", 10)
        self.assertIsNotNone(model)
    def test_vit(self):
        model = get_model("vit_b_16", 10)
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()
