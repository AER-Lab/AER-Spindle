import unittest
from unittest.mock import patch
import numpy as np
import math
import sys

# Ensure the module path is correct.
sys.path.insert(0, 'C:/Users/geosaad/Desktop/Spindle-Custom')
from read_plot_raw_edf import bandpass_filter_channel

class TestDynamicOrder(unittest.TestCase):
    @patch("builtins.print")
    def test_dynamic_order_fs_values(self, mock_print):
        """
        Verify that the dynamic filter order is computed and printed correctly 
        based on different sampling frequencies (fs). The order is calculated as:
        
            order = max(1, math.ceil(fs / 250))
        """
        # Define a list of test sampling frequencies and their expected orders.
        test_cases = [
            (100, max(1, math.ceil(100 / 250))),   # Expected order: 1
            (250, max(1, math.ceil(250 / 250))),     # Expected order: 1
            (251, max(1, math.ceil(251 / 250))),     # Expected order: 2
            (500, max(1, math.ceil(500 / 250))),     # Expected order: 2
            (1000, max(1, math.ceil(1000 / 250))),   # Expected order: 4
        ]
        
        # Use a fixed lowcut and highcut for all tests.
        lowcut = 5
        highcut = 20

        for fs, expected_order in test_cases:
            with self.subTest(fs=fs):
                # Create dummy data for filtering (e.g., an array of ones).
                data = np.ones(fs)

                # Call the bandpass filter. We are not interested in its return value here.
                bandpass_filter_channel(data, fs, lowcut, highcut)

                # Collect all printed outputs during this call.
                # Combine all arguments passed to print into one string.
                printed_lines = [" ".join(map(str, args)) for args, _ in mock_print.call_args_list]

                # Find the line that contains the dynamic order info.
                order_lines = [
                    line for line in printed_lines 
                    if "Dynamic Order (based on sample rate):" in line
                ]

                self.assertTrue(order_lines, "Dynamic order line was not printed.")
                # Verify that the printed line contains the expected order.
                self.assertIn(
                    str(expected_order),
                    order_lines[0],
                    f"For fs={fs}, expected order {expected_order} but got: {order_lines[0]}"
                )

                # Clear the captured print output before the next iteration.
                mock_print.reset_mock()

if __name__ == "__main__":
    unittest.main()
