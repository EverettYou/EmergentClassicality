import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.tensorboard import SummaryWriter
from numbers import Number

''' Dipolar1D: one-dimensional dipolar distribution
    Dipolar1D(m) creates a one-dimensional dipolar distribution with
        * m: polarization (the expectation value), should be within [-1,1]
    Samples are binary (+1 or -1).
        p(x) = (1 + x * m)/2 
    The distribution is based on Bernoulli distribution.
        p = (1 - m)/2 # probability of sampling 1 from Bernoulli distribution
        z ~ Bernoulli(p)
        x = 1 - 2 * z # map 0,1 to +1,-1
'''
class Dipolar1D(dist.Distribution):
    arg_constraints = {'m': dist.constraints.interval(-1, 1)}
    def __init__(self, m = None, validate_args=None):
        # sanity checks
        if m is None:
            raise ValueError("Polarization `m` must be specified.")
        if not torch.is_tensor(m):
            m = torch.tensor(m)
        if not torch.is_floating_point(m):
            m = torch.float(m)
        self.m = m
        self._param = self.m
        # determine batch size
        if isinstance(self.m, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.m.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            # sample from Bernouli distribution with p(1) = (1 - m)/2
            m = self.m.expand(shape)
            p = (1 - m)/2
            z = torch.bernoulli(p)
            # result is 0,1 -> need to map to +1,-1
            x = 1 - 2 * z
            x = x.unsqueeze(dim = -1) # each entry to 1-dim vector
            return x
    
    def log_prob(self, x):
        # calculate probability p(x) = (1 + x * m)/2
        p = (1 + x.squeeze(dim = -1) * self.m)/2
        return torch.log(p)
    

''' Dipolar3D: three-dimensional dipolar distribution
    Dipolar3D(m) creates a three-dimensional dipolar distribution with
        * m: polarization (3 times the expectation value) along z directiion, should be within [-1,1]
    Samples are 3D unit vectors x = (x1,x2,x3)
        p(n) = (1 + x3 * m)/(4*pi)
    Sampling scheme:
        z ~ uniform(0,1)
        x3 = (m - 2 + 4 * z)/(1 + sqrt((m-1)**2 + 4 * m * z))
        theta ~ 2*pi* uniform(0,1)
        (x1, x2) = sqrt(1 - x3**2)*(cos(theta), sin(theta))
        x = (x1,x2,x3)
'''
class Dipolar3D(dist.Distribution):
    arg_constraints = {'m': dist.constraints.interval(-1, 1)}
    has_rsample = True
    def __init__(self, m = None, validate_args = None):
        # sanity checks
        if m is None:
            raise ValueError("Polarization `m` must be specified.")
        if not torch.is_tensor(m):
            m = torch.tensor(m)
        if not torch.is_floating_point(m):
            m = torch.float(m)
        self.m = m
        self._param = self.m
        # determine batch size
        if isinstance(self.m, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.m.size()
        super().__init__(batch_shape, validate_args=validate_args)
        
    def sample(self, sample_shape=torch.Size()):
        # sample() simply calls rsample() with no_grad
        with torch.no_grad():
            x = self.rsample(sample_shape)
        return x
    
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # random real in [0,1]
        z = torch.rand(shape)
        # deform z to x3
        x3 = (self.m - 2 + 4 * z)/(1 + torch.sqrt((self.m - 1)**2 + 4 * self.m * z))
        theta = 2 * np.pi * torch.rand(shape) # random anlge in [0,2pi]
        r = torch.sqrt(1 - x3**2) # length of in-plane component
        # generate (x1,x2) components
        x1 = r*torch.cos(theta)
        x2 = r*torch.sin(theta)
        x = torch.stack([x1,x2,x3], dim = -1) # assemble to x
        return x
    
    def log_prob(self, x):
        # x is of the shape [batch_size, sample_size, x_dim]
        # slice in the last dimension to take its last component
        x3 = x.narrow(dim = -1, start = -1, length = 1).squeeze(dim = -1)
        # calculate probability p(x) = (1 + x3 * m)/(4*pi)
        p = (1 + x3 * self.m)/(4*np.pi)
        return torch.log(p)

''' Apparatus: a simulator of weak measurements on ancilla qubits 
    Apparatus(epsilon, sample_size, alpha0 = 0.) creates an apparatus of
        * epsilon: measurement strength
        * sample_size: number of ancilla qubits
        * alpha0: initial alpha value (quantum coherence = sech(alpha))
    Apparatus.sample(sample_size, scheme) generates samples of weak measurement outcomes
        * batch_size: number of samples to draw
        * scheme: 'fixed' (along z) or 'random'
'''
class Apparatus:
    def __init__(self, epsilon, sample_size, alpha0 = 0., scheme = 'fixed'):
        self.epsilon = torch.tensor(epsilon) # measurement strength logit
        self.sample_size = sample_size # number of qubits
        self.alpha0 = torch.tensor(alpha0) # inital alpha
        self.scheme = scheme # measurment scheme
        if self.scheme is 'fixed':
            self.outcome_size = 1
            self.generator = Dipolar1D
            self.backaction = self._backaction1D
        elif self.scheme is 'random':
            self.outcome_size = 3
            self.generator = Dipolar3D
            self.backaction = self._backaction3D
        else:
            raise ValueError("Measurement scheme should be 'fixed' or 'random'.")
            
    def _backaction1D(self, x):
        return self.epsilon * x
    
    def _backaction3D(self, x3):
        y = x3 * torch.tanh(self.epsilon/2)
        return torch.log(1 + y) - torch.log(1 - y)
        
    def sample(self, batch_size):
        # batch_size :: int : number of samples to draw
        # all samples will be drawn in parallel
        with torch.no_grad():
            # prepare an array to host the alpha value of each sample
            alpha = torch.full([batch_size], self.alpha0)
            # empty array to host records of outcomes
            outcome = torch.empty((batch_size, self.sample_size, self.outcome_size))
            # autoregresive sampling bit-by-bit in parallel among samples
            for i in range(self.sample_size):
                # compute polarization
                m = torch.tanh(alpha) * torch.tanh(self.epsilon)
                # sample from dipolar distribution
                outcome[:,i,:] = self.generator(m).sample()
                # update alpha value by backaction
                alpha += self.backaction(outcome[:,i,-1])
            self.alpha = alpha # book keeping alpha
        return outcome

''' VAE Encoder '''
class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, v_dim = None):
        # x_dim :: int : input dimension
        # z_dim :: int : latent dimension
        # v_dim :: int : value dimension (optional)
        super().__init__()
        if v_dim is None:
            v_dim = 2*x_dim**2 + z_dim + 5
        # attention network: FFN + softmax
        # the final Softmax layer convert score to attention
        self.attention = nn.Sequential(
            nn.Linear(x_dim, 4*v_dim),
            nn.ELU(),
            nn.Linear(4*v_dim, 1),
            nn.Softmax(dim = -2))
        # value network: FFN
        self.value = nn.Sequential(
            nn.Linear(x_dim, 4*v_dim),
            nn.ELU(),
            nn.Linear(4*v_dim, v_dim))
        self.mu = nn.Linear(v_dim + x_dim, z_dim)
        self.logstd = nn.Linear(v_dim + x_dim, z_dim)
        
    def forward(self, x):
        # x is of the shape [batch_size, sample_size, x_dim]
        # calculate attention for every input
        att = self.attention(x) # shape [batch_size, sample_size, 1]
        # generate value from input
        val = self.value(x) # shape [batch_size, sample_size, v_dim]
        # provide a shortcut to input by appending input to value
        val = torch.cat([val, x], dim = -1) # shape [batch_size, sample_size, v_dim + x_dim]
        # apply attention to collect values
        attval = torch.bmm(att.transpose(-1,-2), val) # shape [batch_size, 1, v_dim + x_dim]
        attval = attval.squeeze(dim = -2) # shape [batch_size, v_dim + x_dim]
        # from collected values estimate mean and standard deviation
        z_mu = self.mu(attval) # shape [batch_size, z_dim]
        z_std = torch.exp(self.logstd(attval)) # shape [batch_size, z_dim]
        return dist.Normal(z_mu, z_std)
    
''' VAE Decoder '''
class Decoder(nn.Module):
    def __init__(self, z_dim, m_dim = 1, scheme = 'fixed'):
        # z_dim :: integer : latent dimension
        # m_dim :: integer : parameter dimesion
        super().__init__()
        # prediction network: FFN + tanh
        # the final Tanh layer is important to ensure polarization in [-1,1]
        self.predict = nn.Sequential(
            nn.Linear(z_dim, 4*z_dim),
            nn.ELU(),
            nn.Linear(4*z_dim, m_dim),
            nn.Tanh())
        # scheme specifics
        if scheme is 'fixed':
            self.generator = Dipolar1D
        elif scheme is 'random':
            self.generator = Dipolar3D
    
    def forward(self, z):
        # z is of the shape [batch_size, sample_size, z_dim]
        # predict polarizations from latent samples z
        m = self.predict(z).squeeze(-1)         
        # m will be of the shape [batch_dim, sample_size, m_dim] 
        # or [batch_dim, sample_size] (if m_dim=1)
        return self.generator(m) # sample shape [batch_dim, sample_size, x_dim]

''' Variational Autoencoder (VAE) '''
class VAE(nn.Module):
    reg = 1.
    def __init__(self, z_dim, apparatus, plan = 'A'):
        # z_dim :: int : latent space dimension
        # apparatus :: Apparatus : the measurement apparatus to study
        super().__init__()
        self.apparatus = apparatus # hold apparatus
        # scheme specifics
        if self.apparatus.scheme is 'fixed':
            x_dim = 1
        elif self.apparatus.scheme is 'random':
            x_dim = 3
        # construct encoder and decoder
        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(z_dim, scheme=self.apparatus.scheme)
        # set optimizer
        self.optimizer = optim.Adam(self.parameters())
        self.step = 0 # step counter
        # plan specifics
        if plan is 'A': # latent variable sampled once and broadcast over qubits
            self.sample_size = 1
        elif plan is 'B': # labent variable sampled independently for each qubit
            self.sample_size = self.apparatus.sample_size        
    
    def __str__(self):
        return 'VAE_eps'+str(self.apparatus.epsilon.numpy())\
                +'_N'+str(self.apparatus.sample_size)+'_'+str(id(self))[-4:]
    
    def forward(self, x):
        # x is of the shape [batch_size, sample_size, x_dim]
        # get encoding distribution q(z|x)
        q = self.encoder(x) # single sample shape [batch_size, z_dim]
        # sample latent z from q(z|x) by reparametrization
        z = q.rsample([self.sample_size]) # shape [sample_size, batch_size, z_dim]
        z = z.transpose(0,1) # shape [batch_size, sample_size, z_dim]
        # get decoding distribution p(x|z)
        p = self.decoder(z) # single sample shape [batch_size, sample_size, x_dim]
        return p, q
    
    def loss(self, x):
        # x is of the shape [batch_size, sample_size, x_dim]
        # forward pass to construct the distributions p(x|z) and q(z|x)
        p, q = self(x)
        # reconstruction loss
        re_loss = - torch.sum(p.log_prob(x))
        # kl divergence loss
        kl_loss = torch.sum((q.scale**2 + q.loc**2 - 1.)/2 - torch.log(q.scale))
        # total loss
        loss = re_loss + self.reg * kl_loss
        return loss
    
    def learn(self, steps=1, batch_size=1):
        # steps :: int : number of steps to train
        # batch_size :: int : batch size of train data in each step
        super().train() # set the module in training mode
        # initialize training loss
        run_loss = 0.
        # training iteration
        for i in range(steps):
            # clear gradients
            self.optimizer.zero_grad()
            # draw training data
            x = self.apparatus.sample(batch_size)
            # forward pass
            loss = self.loss(x)
            # backward pass
            loss.backward()
            run_loss += loss.item()
            # update model parameters
            self.optimizer.step()
        self.step += steps
        train_loss = run_loss/batch_size/steps
        return train_loss

    def test(self, batch_size=1):
        # test_iterator serves the testing data
        super().eval() # set the module in evaluation mode
        # initialize testing loss
        run_loss = 0.
        # testing iteration
        with torch.no_grad():
            # draw testing data
            x = self.apparatus.sample(batch_size)
            # forward pass
            loss = self.loss(xs)
            run_loss += loss.item()
        test_loss = run_loss/batch_size
        return test_loss