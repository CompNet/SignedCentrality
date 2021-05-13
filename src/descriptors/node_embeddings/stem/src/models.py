import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import util
import numpy as np




def regularize(parameter, p=2):
    zeros = torch.zeros_like(parameter)
    diff = torch.abs(parameter - zeros)
    norm = torch.norm(diff, p)
    return norm


class GraphEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_embedding(self, x):
        raise NotImplementedError

    def get_all_weights(self):
        raise NotImplementedError

    def get_edge_features(self, x, y, operation='hadamard'):
        func = util.FEATURE_FUNCS[operation]
        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)
        return func(x_emb, y_emb)

    def regularize(self, lam=0.0055, p=2):
        regularizer_term = Variable(torch.zeros(1))
        for parameter in self.parameters():
            regularizer_term += regularize(parameter, p)
        regularizer_term *= lam
        return regularizer_term




class EnergyToProbsLayer(nn.Module):
    def __init__(self):
        super(EnergyToProbsLayer, self).__init__()
        self.transform = nn.Sigmoid()
    def forward(self, x):
        ones = torch.ones_like(x)
        positive_prob = self.transform(x)
        negative_prob = ones - positive_prob
        output = torch.cat((negative_prob, positive_prob), dim=1)
        #output = torch.log(output)
        return output


#-----------------------------------------------------------------------------------------------------------------
#
#  My Initial Pseudo Kernel Model
#
#-----------------------------------------------------------------------------------------------------------------

class PseudoKernelEmbedding(GraphEmbeddingModel):
    def __init__(self, num_nodes, dims):
        super().__init__()
        vocab_size = num_nodes + 1
        self.embeddings = nn.Embedding(vocab_size, dims)
        self.initial_layer = nn.Linear(dims, int(dims / 2))
        self.initial_transform = nn.PReLU(init=0.0)#nn.ReLU()
        self.pseudo_kernel = nn.Bilinear(int(dims / 2), int(dims / 2), 1)
        self.transform_layer = EnergyToProbsLayer()

        initrange = (2.0 / (vocab_size + dims)) ** 0.5  # Xavier init
        self.embeddings.weight.data.uniform_(-initrange, initrange)  # init

    def _compute_embedding(self, x):
        emb = self.embeddings(x)
        emb = self.initial_layer(emb)
        emb = self.initial_transform(emb)
        return emb

    def get_all_weights(self):
        data = self.embeddings.weight
        res = self.initial_layer(data)
        res = res.data.numpy()
        return res


    def get_embedding(self, x, tensorfy=True):
        x = int(x)
        if tensorfy:
            x = Variable(torch.LongTensor([x]))
        emb = self.embeddings(x)
        emb = self.initial_layer(emb)
        emb = emb.data.numpy()[0]
        return emb

    def forward(self, u, v):
        u_emb = self._compute_embedding(u)
        v_emb = self._compute_embedding(v)
        prod = self.pseudo_kernel(u_emb, v_emb)
        probs = self.transform_layer(prod)
        return probs






def fit_pseudo_kernel_model(num_nodes, dims, X, y, epochs=100, lr=0.01,
                            lr_decay=0.01, lam=0.000055, p=1, weight_decay=0.0, ratio=0.8,
                            undersample=True, print_loss=True):
    pseudo_kernel_model = PseudoKernelEmbedding(num_nodes, dims)
    optimizer = optim.Adagrad(pseudo_kernel_model.parameters(), lr=lr,
                              lr_decay=lr_decay, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_negative = len(y[y == 0])
    num_positive = int(ratio * num_negative)
    ratio_dict = {0: num_negative, 1: num_positive}
    rus = RandomUnderSampler(return_indices=False, ratio=ratio_dict)
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_resampled, y_resampled = X, y
        if undersample:
            X_resampled, y_resampled = rus.fit_sample(X, y)
        u = Variable(torch.LongTensor(X_resampled[:,0]))
        v = Variable(torch.LongTensor(X_resampled[:,1]))
        probs = pseudo_kernel_model(u, v)
        actual_targets = Variable(torch.LongTensor(y_resampled))
        loss = criterion(probs, actual_targets)
        if epoch >= int(0 * epochs):
            #print('Considering regularization...')
            loss += pseudo_kernel_model.regularize(lam, p)
        loss.backward()
        optimizer.step()
        if print_loss:
            loss_val = loss.data[0]
            print('The loss at epoch ', epoch + 1, ' was ', loss_val)
    return pseudo_kernel_model


#-----------------------------------------------------------------------------------------------------------------
#
#  PseudoKernelRanking Model
#
#-----------------------------------------------------------------------------------------------------------------

class PseudoKernelRankingModel(GraphEmbeddingModel):
    def __init__(self, num_nodes, dims):
        super().__init__()
        vocab = num_nodes + 1
        self.embeddings = nn.Embedding(vocab, dims)
        self.pseudo_kernel = nn.Bilinear(dims, dims, 1)
        self.rank_transform = nn.Sigmoid()

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([int(x)]))
        emb = self.embeddings(x)
        emb = emb.data.numpy()[0]
        return emb

    def forward(self, xi, xj, xk, delta):
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        positive = self.rank_transform(self.pseudo_kernel(i_emb, j_emb))
        negative = self.rank_transform(self.pseudo_kernel(i_emb, k_emb))
        zeros = torch.zeros_like(positive)

        res = torch.max(zeros, negative + delta - positive)
        res = torch.sum(res)
        return res


def fit_ranking_model(num_nodes, dims, triples, triples0, delta, delta0, batch_size, batch_size0, epochs,
                   lr=0.1, lam=0.00055, lr_decay=0.0, p=2, print_loss=True, p0=True):
    ranking_model = PseudoKernelRankingModel(num_nodes, dims)
    optimizer = optim.Adagrad(ranking_model.parameters(), lr=lr, lr_decay=lr_decay)
    for epoch in range(epochs):
        optimizer.zero_grad()
        C = batch_size
        xi, xj, xk = util.get_triples_training_batch(triples, batch_size)
        loss = ranking_model(xi, xj, xk, delta)
        if p0:
            xi, xj, xk = util.get_triples_training_batch(triples0, batch_size0)
            loss += ranking_model(xi, xj, xk, delta0)
            C += batch_size0
        loss /= C
        loss += ranking_model.regularize(lam, p)
        loss.backward()
        optimizer.step()
        if print_loss:
            print('Loss at epoch ', epoch + 1, ' is ', loss.data[0])
    return ranking_model


#-----------------------------------------------------------------------------------------------------------------
#
#  SiNE model of Wang et al.
#
#-----------------------------------------------------------------------------------------------------------------



class SiNESubModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.tanh = nn.Tanh()

        initrange = np.sqrt(6.0/(input_dim + output_dim))
        self.layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, input1):
        x = self.layer(input1)
        x += self.bias
        x = self.tanh(x)
        return x


class SiNECompModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, output_dim, bias=False)
        self.layer2 = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.tanh = nn.Tanh()

        initrange = np.sqrt(6.0 / (input_dim + output_dim))
        self.layer1.weight.data.uniform_(-initrange, initrange)
        self.layer2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input1, input2):
        x = self.layer1(input1) + self.layer2(input2)
        x += self.bias
        x = self.tanh(x)
        return x






class SiNE(GraphEmbeddingModel):
    def __init__(self, num_nodes, dims_arr):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes + 1, dims_arr[0])
        self.embeddings.weight.data.uniform_(-0.0, 0)
        self.layers = []
        self.comp_layer = SiNECompModule(dims_arr[0], dims_arr[1])
        length = len(dims_arr)
        for i in range(1, length - 1):
            layer = SiNESubModule(dims_arr[i], dims_arr[i + 1])
            self.add_module('l{0}'.format(i), layer)
            self.layers.append(layer)
        layer = SiNESubModule(dims_arr[-1], 1)
        self.add_module('l{0}'.format(len(dims_arr)), layer)
        self.layers.append(layer)

    def get_all_weights(self):
        res = self.embeddings.weight.data.numpy()
        return res

    def forward(self, xi, xj, xk, delta):
        i_emb = self.embeddings(xi)
        j_emb = self.embeddings(xj)
        k_emb = self.embeddings(xk)

        zn1 = self.comp_layer(i_emb, j_emb)
        zn2 = self.comp_layer(i_emb, k_emb)

        for layer in self.layers:
            zn1 = layer(zn1)
            zn2 = layer(zn2)

        f_pos = zn1
        f_neg = zn2

        zeros = Variable(torch.zeros(1))

        loss = torch.max(zeros, f_neg + delta - f_pos)
        loss = torch.sum(loss)

        return loss

    def get_embedding(self, x):
        x = Variable(torch.LongTensor([int(x)]))
        emb = self.embeddings(x)
        emb = emb.data.numpy()[0]
        return emb



def fit_sine_model(num_nodes, dims_arr, triples, triples0, delta, delta0, batch_size, batch_size0, epochs,
                   lr=0.01, lam=0.0055, lr_decay=0.0, p=2, print_loss=True, p0=True):
    sine = SiNE(num_nodes, dims_arr)
    optimizer = optim.Adagrad(sine.parameters(), lr=lr, lr_decay=lr_decay)
    for epoch in range(epochs):
        optimizer.zero_grad()
        C = batch_size
        xi, xj, xk = util.get_triples_training_batch(triples, batch_size)
        loss = sine(xi, xj, xk, delta)
        if p0:
            xi, xj, xk = util.get_triples_training_batch(triples0, batch_size0)
            loss += sine(xi, xj, xk, delta0)
            C += batch_size0
        loss /= C
        loss += sine.regularize(lam, p)
        loss.backward()
        optimizer.step()
        if print_loss:
            print('Loss at epoch ', epoch + 1, ' is ', loss.data[0])
    return sine



#-----------------------------------------------------------------------------------------------------------------
#
#  HOPE Model
#
#-----------------------------------------------------------------------------------------------------------------

from gem.embedding.hope import HOPE

class GEMModel(GraphEmbeddingModel):

    def __init__(self):
        super().__init__()
        self.embedding_model = None
        self.embeddings = None

    def fit(self, graph, is_weighted=False):
        self.embedding_model.learn_embedding(graph=graph, is_weighted=is_weighted)
        self.embeddings = self.embedding_model.get_embedding()
        print(self.embeddings)

    def get_all_weights(self):
        return self.embeddings

    def get_embedding(self, x):
        return self.embeddings[x]

class HOPEModel(GEMModel):
    def __init__(self, dims, beta):
        super().__init__()
        self.dims = dims
        self.beta = beta
        self.embedding_model = HOPE(d=dims, beta=beta)

    def get_embedding(self, x):
        return super().get_embedding(x)


def fit_hope(dims,  X_train, num_nodes, beta=0.01, directed=True):
    graph = util.graph_from_numpy_array(X_train, num_nodes, directed)
    embedding_model = HOPEModel(dims, beta)
    embedding_model.fit(graph)
    return embedding_model


#-----------------------------------------------------------------------------------------------------------------
#
#  SDNE Model
#
#-----------------------------------------------------------------------------------------------------------------

from gem.embedding.sdne import SDNE
from keras.layers import Activation, Dense

class SDNEModel(GEMModel):
    def __init__(self, dims, beta=5, alpha=1e-5,nu1=1e-6,
                 nu2=1e-6,K=3, n_units=[64,32],
                 rho=0.3, n_iter=50, xeta=0.01,n_batch=500):
        super().__init__()
        self.dims  = dims
        self.beta = beta
        self.alpha = alpha
        self.nu1 = nu1
        self.nu2 = nu2
        self.K = K
        self.n_units = n_units
        self.rho = rho
        self.n_iter = n_iter
        self.xeta = xeta
        self.n_batch = n_batch

        modelfile = ['./intermediate/enc_model.json', './intermediate/dec_model.json'],
        weightfile = ['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5']

        self.embedding_model = SDNE(d=dims, beta=beta, alpha=alpha, nu1=nu1,nu2=self.nu2, K=K, n_units=n_units,rho=rho,
                                    n_iter=n_iter, xeta=xeta, n_batch=n_batch, modelfile=['./intermediate/enc_model.json', './intermediate/dec_model.json'],
                weightfile=['./intermediate/enc_weights.hdf5', './intermediate/dec_weights.hdf5'])

    def get_embedding(self, x):
        return super().get_embedding(x - 1)



def fit_sdne(dims, num_nodes, X_train, epochs, batch_size, directed=False):
    graph = util.graph_from_numpy_array(X_train, num_nodes, directed )
    embedding_model = SDNEModel(dims, n_iter=epochs, n_batch=batch_size)
    print(graph.number_of_nodes())
    embedding_model.fit(graph=graph, is_weighted=False)
    return embedding_model


