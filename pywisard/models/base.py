import numpy as np

class RAM:
    def __init__(self, tuple_size, mem_size = 2, forget_factor = None):
        self.tuple_size = tuple_size
        self.mem_size = mem_size
        self.forget_factor = forget_factor

        self.table = {}
        # np.zeros((2**self.tuple_size, self.mem_size), dtype= np.float32)


    def __get_position(self, input_array):
        return int("".join(str(x) for x in input_array), 2)

    def put(self, input_array, y):
        position = self.__get_position(input_array)

        if self.forget_factor is None:
            x = self.table.get(position, [0, 0])
            x[0] += 1
            x[1] += y

            self.table[position] = x
        else:
            x = self.table.get(position, [0, 0])
            x[0] += 1
            x[1] = self.forget_factor * x[1] + y
            
            self.table[position] = x

    def get(self, input_array):
        position = self.__get_position(input_array)
        return self.table.get(position, [0,0])

class AddressMapping:
    def __init__(self, input_size, tuple_size, seed=None):
        self.input_size, self.tuple_size = input_size, tuple_size
        self.n_rams = np.ceil(self.input_size/self.tuple_size)

        np.random.seed(seed=seed)
        self.mapping = np.random.permutation(input_size).astype(np.int32)

    def get(self, input_array):
        output = []
        for i in np.arange(0, self.n_rams, dtype=np.int32):
            x = self.mapping[i*self.tuple_size:(i + 1)*self.tuple_size]
            y = np.array(input_array)[x]

            output.append(y)
        return output