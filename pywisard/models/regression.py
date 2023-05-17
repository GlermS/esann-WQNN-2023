import numpy as np
from pywisard.models import base

## Modela a Wisard como se fosse uma convolução de uma 1d
## O tamanho da tabela das rams = 2^(tuple_size) ** é o número de combinações com os inputs binários 

# exemplo - input_size = 5, tuple_size = 2
# conv1d - stride = tuple_size

# conv1d step 1 - [>1<, >0<,  0 ,  1 ,  1 ]
# conv1d step 2 - [ 1 ,  0 , >0<, >1<,  1 ]
# conv1d step 3 - [ 1 ,  0 ,  0 ,  1 , >1<]





class RegWisard:
    def __init__(self, input_size, tuple_size=32, forget_factor = None):
        self.input_size = input_size
        self.tuple_size = tuple_size
        self.forget_factor = forget_factor
        self.n_rams = int(np.ceil(self.input_size/self.tuple_size))

        if self.n_rams == np.floor(self.input_size/self.tuple_size):
            self.RAMs = [base.RAM(self.tuple_size, forget_factor=self.forget_factor) for i in range(self.n_rams)]
        else:
            self.RAMs = [base.RAM(self.tuple_size, forget_factor=self.forget_factor) for i in range(self.n_rams - 1)]
            self.RAMs.append(base.RAM(self.input_size % self.tuple_size, forget_factor=self.forget_factor))
        self.AddMap = base.AddressMapping(self.input_size, self.tuple_size)

        

    
    def train(self, bin_input, y):
        mapped_input = self.AddMap.get(bin_input)
        for ram, m_input in zip( self.RAMs,mapped_input):
            ram.put(m_input, y)

    def predict(self, bin_input):
        c, s = 0, 0
        mapped_input = self.AddMap.get(bin_input)

        if self.forget_factor is None:
            for ram, m_input in zip( self.RAMs,mapped_input):
                x = ram.get(m_input)
                c+= x[0]
                s+= x[1]
        else:
            for ram, m_input in zip( self.RAMs,mapped_input):
                x = ram.get(m_input)
                c+= (1 - self.forget_factor**x[0])/(1 - self.forget_factor)
                s+= x[1]
        if c > 0:
            return s/c
        else:
            return 0
        



# class Memory(nn.Module):
#     def __init__(self, tuple_size, input_dim, num_outputs, seed=21):
#         nn.Module.__init__(self)

#         # gera uma permutação dos indices - mapping = [5, 3, 1, 4, 2, 0]
#         # número de rams
        
#         self.input_dim = input_dim
#         self.tuple_size = tuple_size

#         self.n_rams = torch.ceil(
#           torch.tensor(self.input_dim/self.tuple_size)
#         ).type(torch.int).item()

#         self.policy_output_dim = torch.Size([num_outputs])

#         self.mapping = torch.randperm(
#             self.input_dim,
#             generator=torch.Generator().manual_seed(seed)
#         )


#         self.key_weights = torch.special.exp2(
#             torch.arange(tuple_size).reshape(1, 1, -1)
#         ).to(torch.long)

#         self.mem_offsets = (2 ** tuple_size) * torch.arange(self.n_rams).reshape((1, -1))

#         self.memory = nn.utils.skip_init(
#             nn.EmbeddingBag,
#             num_embeddings=self.n_rams * 2 ** tuple_size, #  numero todal de slots de memoria
#             embedding_dim=self.policy_output_dim[0],
#             mode='sum'
#         )
#         nn.init.zeros_(self.memory.weight)

#     #pega os indices que serão acessados no memoria
#     def keys(self, x):
#         keys = F.conv1d(
#             x,
#             self.key_weights,
#             stride=self.tuple_size
#         ).squeeze(1)
#         keys += self.mem_offsets
#         return keys
    
#     def forward(self, x):
#         keys = self.keys(x)

#         # RAM neurons are stacked along the first dimension of mem. Therefore,
#         # it's necessary to offset the keys occording to which neuron we are
#         # trying to access
#         return self.memory(keys)