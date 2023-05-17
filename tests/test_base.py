from pywisard.models import base
import numpy as np

class TestRAM:
    def test_RAM_put(self):
        ram = base.RAM(3)
        input_array, y = [0, 0, 0], 120

        ram.put(input_array, y)

        assert ram.table[0][1] == 120
        assert ram.table[0][0] == 1
        # assert np.sum(ram.table[:, 1]) == 120

        input_array, y = [0, 0, 0], 30

        ram.put(input_array, y)

        assert ram.table[0][1] == 150
        assert ram.table[0][0] == 2
        # assert np.sum(ram.table[:, 1]) == 150
        # assert np.sum(ram.table[:, 0]) == 2
        
        input_array, y = [0, 1, 0], 30
        ram.put(input_array, y)

        assert ram.table[2][1] == 30
        assert ram.table[2][0] == 1
        assert 3 not in ram.table
        # assert np.sum(ram.table[:, 1]) == 180
        # assert np.sum(ram.table[:, 0]) == 3

    def test_RAM_get(self):
        ram = base.RAM(3)

        input_array, y = [0, 0, 0], 120
        ram.put(input_array, y)

        input_array, y = [0, 0, 0], 30
        ram.put(input_array, y)

        input_array, y = [0, 1, 0], 30
        ram.put(input_array, y)

        assert ram.get([0, 0, 0])[1] == 150
        assert ram.get([0, 1, 0])[1] == 30
        assert ram.get([1, 1, 0])[1] == 0

        assert ram.get([0, 0, 0])[0] == 2
        assert ram.get([0, 1, 0])[0] == 1
        assert ram.get([1, 1, 0])[0] == 0

class TestAddressMap:
    def test_RAM_put(self):
        add = base.AddressMapping(7, 3)
        input_array = [1, 0, 1, 0, 0, 0, 1]

        mapped = add.get(input_array)
        mapping = add.mapping

        assert mapped[0][0] == input_array[mapping[0]]
        assert mapped[2][0] == input_array[mapping[6]]
        assert len(mapping) == 7
        assert len(mapped) == 3
        