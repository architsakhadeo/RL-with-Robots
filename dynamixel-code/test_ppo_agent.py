import unittest
import numpy as np
import torch
from ppo_agent import PPO

"""
python -m unittest discover
"""


class TestPPOAgent(unittest.TestCase):

    def test_return(self):
        # TODO: Run unit tests on your return.
        # How does it handle the end of an episode and the start of the next one?
        # How does it handle the end of the batch?
        # Does it handle gamma and lambda properly?
        
        # Testing Lambda returns for different values of Lambda
        
        value1 = PPO.compute_return([1,1,2,3],torch.tensor([0.9,0.8,1.7,2.3]),0.99, 0.95)
        value2 = [6.42475879855967, 5.72584667514044, 4.935349997639656, 3]
        
        value3 = PPO.compute_return([1,1,2,3],torch.tensor([0.9,0.8,1.7,2.3]),0.99, 0)
        value4 = [1.7920000118017196, 2.6830000472068787, 4.276999952793121, 3]
        
        value5 = PPO.compute_return([1,1,2,3],torch.tensor([0.9,0.8,1.7,2.3]),0.99, 1)
        value6 = [6.861097, 5.9203, 4.97, 3]

        self.assertTrue(value1 == value2)
        self.assertTrue(value3 == value4)
        self.assertTrue(value5 == value6)                


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPPOAgent)
    unittest.TextTestRunner(verbosity=2).run(suite)
