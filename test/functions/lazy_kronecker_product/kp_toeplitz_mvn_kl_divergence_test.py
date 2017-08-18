import torch
from gpytorch.functions.lazy_kronecker_product import KPToeplitzMVNKLDivergence
from gpytorch.functions import MVNKLDivergence
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch import utils
from torch.autograd import Variable
from gpytorch.utils.kronecker_product import kronecker_product


def test_toeplitz_mvn_kl_divergence_forward():
    x = []
    x.append(Variable(torch.linspace(0, 1, 10)))
    x.append(Variable(torch.linspace(0, 1, 10)))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, 3)
    cs = Variable(torch.zeros(2, 5), requires_grad=True)
    for i in range(2):
        covar_x = covar_module.forward(x[i].unsqueeze(1), x[i].unsqueeze(1))
        cs.data[i] = covar_x.c.data

    mu1 = Variable(torch.randn(25), requires_grad=True)
    mu2 = Variable(torch.randn(25), requires_grad=True)

    Ts = []
    for k in range(2):
        Ts.append(Variable(torch.zeros(cs.size()[1], cs.size()[1])))
        for i in range(cs.size()[1]):
            for j in range(cs.size()[1]):
                Ts[k][i, j] = utils.toeplitz.toeplitz_getitem(cs[k], cs[k], i, j)

    K = kronecker_product(Ts)

    U = torch.randn(25, 25).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = MVNKLDivergence()(mu1, U, mu2, K)
    res = KPToeplitzMVNKLDivergence(num_samples=400)(mu1, U, mu2, cs)

    assert all(torch.abs((res.data - actual.data) / actual.data) < 0.15)


def test_toeplitz_mvn_kl_divergence_backward():
    x = []
    x.append(Variable(torch.linspace(0, 1, 10)))
    x.append(Variable(torch.linspace(0, 1, 10)))
    rbf_covar = RBFKernel()
    rbf_covar.initialize(log_lengthscale=-4)
    covar_module = GridInterpolationKernel(rbf_covar, 3)
    cs = Variable(torch.zeros(2, 5), requires_grad=True)
    for i in range(2):
        covar_x = covar_module.forward(x[i].unsqueeze(1), x[i].unsqueeze(1))
        cs.data[i] = covar_x.c.data

    mu1 = Variable(torch.randn(25), requires_grad=True)
    mu2 = Variable(torch.randn(25), requires_grad=True)

    Ts = []
    for k in range(2):
        Ts.append(Variable(torch.zeros(cs.size()[1], cs.size()[1])))
        for i in range(cs.size()[1]):
            for j in range(cs.size()[1]):
                Ts[k][i, j] = utils.toeplitz.toeplitz_getitem(cs[k], cs[k], i, j)

    K = kronecker_product(Ts)

    U = torch.randn(25, 25).triu()
    U = Variable(U.mul(U.diag().sign().unsqueeze(1).expand_as(U).triu()), requires_grad=True)

    actual = MVNKLDivergence()(mu1, U, mu2, K)
    actual.backward()

    actual_cs_grad = cs.grad.data
    actual_mu1_grad = mu1.grad.data
    actual_mu2_grad = mu2.grad.data
    actual_U_grad = U.grad.data

    cs.grad.data.fill_(0)
    mu1.grad.data.fill_(0)
    mu2.grad.data.fill_(0)
    U.grad.data.fill_(0)

    res = KPToeplitzMVNKLDivergence(num_samples=400)(mu1, U, mu2, cs)
    res.backward()

    res_cs_grad = cs.grad.data
    res_mu1_grad = mu1.grad.data
    res_mu2_grad = mu2.grad.data
    res_U_grad = U.grad.data

    assert utils.approx_equal(res_cs_grad, actual_cs_grad)
    assert utils.approx_equal(res_mu1_grad, actual_mu1_grad)
    assert utils.approx_equal(res_mu2_grad, actual_mu2_grad)
    assert utils.approx_equal(res_U_grad, actual_U_grad)
