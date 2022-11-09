import  numpy as np
import  torch
import  torch.nn.functional as F


class NTMReadHead(torch.nn.Module):

    def __init__(self, memory, ctrlr_sz):
        """
        :param memory:
        :param ctrlr_sz:
        """
        super(NTMReadHead, self).__init__()
        self.memory = memory
        self.N, self.M = memory.size()
        self.ctrlr_sz = ctrlr_sz

        # Corresponding to k, beta, g, s, gamma sizes from the paper
        # k is of size of M, and s unit of size 3
        self.read_len = [self.M, 1, 1, 3, 1]
        self.fc_read = torch.nn.Linear(ctrlr_sz, sum(self.read_len))
        self.reset_parameters()

    def new_w(self, batchsz):
        # The state holds the previous time step address weightings
        return torch.zeros(batchsz, self.N).to('cuda')

    def reset_parameters(self):
        # Initialize the linear layers
        torch.nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        torch.nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, h, w_prev):
        """
        NTMReadHead forward function.
        :param h: controller hidden variable
        :param w_prev: previous step state
        """
        o = self.fc_read(h)
        # [b, 26] split with [20, 1, 1, 3, 1]
        k, beta, g, s, gamma = torch.split(o, self.read_len, dim=1)

        # obtain address w
        w = self.memory.address(k, beta, g, s, gamma, w_prev)
        # read
        r = self.memory.read(w)

        return r, w

class NTMWriteHead(torch.nn.Module):

    def __init__(self, memory, ctrlrsz):
        """
        :param memory:
        :param ctrlrsz:
        """
        super(NTMWriteHead, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.ctrlrsz = ctrlrsz

        # Corresponding to k, beta, g, s, gamma, e, a sizes from the paper
        self.write_len = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = torch.nn.Linear(ctrlrsz, sum(self.write_len))
        self.reset_parameters()

    def new_w(self, batch_size):
        return torch.zeros(batch_size, self.N).to('cuda')

    def reset_parameters(self):
        # Initialize the linear layers
        torch.nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        torch.nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, h, w_prev):
        """
        NTMWriteHead forward function.
        :param h: controller hidden variable
        :param w_prev: previous step state
        """
        o = self.fc_write(h)
        # [b, len] split with [20, 1, 1, 3, 1, 20, 20]
        k, beta, g, s, gamma, e, a = torch.split(o, self.write_len, dim=1)

        # e should be in [0, 1]
        e = torch.sigmoid(e)

        # retain address w
        w = self.memory.address(k, beta, g, s, gamma, w_prev)
        # write into memory
        self.memory.write(w, e, a)

        return w


