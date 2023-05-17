from pywisard.models import regression
import numpy as np

class TestRegWisard:
    def test_ReW_basecase(self):
        model = regression.RegWisard(7,3)
        model.train([1, 0, 0, 1, 1, 0, 0],7)

        assert model.predict([1, 0, 0, 1, 1, 0, 0]) == 7
        assert model.predict([0, 1, 1, 0, 0, 1, 1]) == 0

    def test_ReW_mean(self):
        model = regression.RegWisard(7,3)
        model.train([1, 0, 0, 1, 1, 0, 0], 7)
        model.train([1, 0, 0, 1, 1, 0, 0], 3)

        assert model.predict([1, 0, 0, 1, 1, 0, 0]) == 5

    def test_ReW_multiinpute(self):
        model = regression.RegWisard(7,3)
        model.train([1, 0, 0, 1, 1, 0, 0], 7)
        model.train([1, 0, 0, 1, 1, 0, 0], 3)
        model.train([0, 1, 1, 0, 0, 1, 1], 3)

        assert model.predict([1, 0, 0, 1, 1, 0, 0]) == 5
        assert model.predict([0, 1, 1, 0, 0, 1, 1]) == 3


    def test_ReW_withforget(self):
        model = regression.RegWisard(7,3, forget_factor=0.9)
        model.train([1, 0, 0, 1, 1, 0, 0], 7)
        model.train([1, 0, 0, 1, 1, 0, 0], 3)

        c = (7*0.9 + 3)/((1 - 0.9**2)/(1 - 0.9))      
        np.testing.assert_almost_equal(model.predict([1, 0, 0, 1, 1, 0, 0]), c,decimal=7)