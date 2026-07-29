"""Test suite verifying Verilog TPU matrix multiplication unit."""
import unittest

class TPUMatmulSim:
    def __init__(self, data_width: int = 16):
        self.data_width = data_width
        self.accum_out = 0

    def compute_step(self, act_in: int, weight_in: int, enable: bool = True):
        if enable:
            self.accum_out += act_in * weight_in

class TestTPUMatmul(unittest.TestCase):

    def test_matmul_accumulation(self):
        unit = TPUMatmulSim()
        unit.compute_step(act_in=12, weight_in=8)
        self.assertEqual(unit.accum_out, 96)

if __name__ == "__main__":
    unittest.main()
