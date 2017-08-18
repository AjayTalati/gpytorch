from torch.autograd import Function
from gpytorch.utils import LinearCG, StochasticLQ
import torch


class _KPToeplitzTraceLogDetQuadForm(Function):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def forward(self, mu_diff, chol_covar1, covar2_toeplitz_columns):
        def mul_closure(v):
            return kronecker_product_toeplitz_mul(covar2_toeplitz_columns, covar2_toeplitz_columns, v)

        def quad_form_closure(z):
            return z.dot(LinearCG().solve(mul_closure, chol_covar1.t().mv(chol_covar1.mv(z))))

        # log |K2|
        log_det_covar2, = StochasticLQ(num_random_probes=10).evaluate(mul_closure,
                                                                      len(mu_diff),
                                                                      [lambda x: x.log()])

        # Tr(K2^{-1}K1)
        sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))
        trace = 0
        for z in sample_matrix:
            trace = trace + quad_form_closure(z)
        trace = trace / self.num_samples

        # Inverse quad form
        mat_inv_y = LinearCG().solve(mul_closure, mu_diff)
        inv_quad_form = mat_inv_y.dot(mu_diff)

        res = log_det_covar2 + trace + inv_quad_form

        self.save_for_backward(mu_diff, chol_covar1, covar2_toeplitz_columns)
        self.mul_closure = mul_closure
        self.mat_inv_y = mat_inv_y

        return mu_diff.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]

        mu_diff, chol_covar1, covar2_toeplitz_columns = self.saved_tensors
        mat_inv_y = self.mat_inv_y
        mul_closure = self.mul_closure

        def deriv_quad_form_closure(z):
            I_minus_Kinv_M_z = z - LinearCG().solve(mul_closure, chol_covar1.t().mv(chol_covar1.mv(z)))
            Kinv_z = LinearCG().solve(mul_closure, z)
            return kp_sym_toeplitz_derivative_quadratic_form(covar2_toeplitz_columns, Kinv_z, I_minus_Kinv_M_z)

        grad_mu_diff = None
        grad_cholesky_factor = None
        grad_toeplitz_columns = None

        if self.needs_input_grad[0]:
            # Need gradient with respect to mu_diff
            grad_mu_diff = mat_inv_y.mul_(2 * grad_output_value)

        if self.needs_input_grad[1]:
            # Compute gradient with respect to the Cholesky factor L
            grad_cholesky_factor = 2 * LinearCG().solve(mul_closure, chol_covar1)
            grad_cholesky_factor.mul_(grad_output_value)

        if self.needs_input_grad[2]:
            sample_matrix = torch.sign(torch.randn(self.num_samples, len(mu_diff)))

            grad_toeplitz_columns = torch.zeros(covar2_toeplitz_columns.size())
            for z in sample_matrix:
                grad_toeplitz_columns = grad_toeplitz_columns + deriv_quad_form_closure(z)
            grad_toeplitz_columns = grad_toeplitz_columns / self.num_samples

            grad_toeplitz_columns.add_(- kp_sym_toeplitz_derivative_quadratic_form(grad_toeplitz_columns,
                                                                                   mat_inv_y.squeeze(),
                                                                                   mat_inv_y.squeeze()))
            grad_toeplitz_columns.mul_(grad_output_value)

        return grad_mu_diff, grad_cholesky_factor, grad_toeplitz_columns


class KPToeplitzMVNKLDivergence(Function):
    """
    PyTorch function for computing the KL-Divergence between two multivariate
    Normal distributions.
    For this function, the first Gaussian distribution is parameterized by the
    mean vector \mu_1 and the Cholesky decomposition of the covariance matrix U_1:
    N(\mu_1, U_1^{\top}U_1).
    The second Gaussian distribution is parameterized by the mean vector \mu_2
    and the full covariance matrix \Sigma_2: N(\mu_2, \Sigma_2). Furthermore,
    \Sigma_2 is assumed to be a kronecker product of Toeplitz matrices.
    The KL divergence between two multivariate Gaussians is given by:
        KL(N_1||N_2) = 0.5 * (Tr(\Sigma_2^{-1}\Sigma_{1}) + (\mu_2 -
            \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) -
            logdet(\Sigma_{1}) - D)
    Where D is the dimensionality of the distributions.
    """
    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __call__(self, mu1_var, chol_covar1_var, mu2_var, covar2_vars):
        mu_diffs = mu2_var - mu1_var

        # ExactGPMarginalLogLikelihood gives us -0.5 * [\mu_2 -
        # \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) + const]
        # Multiplying that by -2 gives us two of the terms in the KL divergence
        # (plus an unwanted constant that we can subtract out).

        # Get logdet(\Sigma_{1})
        log_det_covar1 = chol_covar1_var.diag().log().sum(0) * 2

        # Get Tr(\Sigma_2^{-1}\Sigma_{1})
        # trace = ToeplitzTraceInvMM(num_samples=self.num_samples)(covar2_var, chol_covar1_var)

        trace_logdet_quadform = _KPToeplitzTraceLogDetQuadForm()(mu_diffs, chol_covar1_var, covar2_vars)

        # get D
        D = len(mu_diffs)

        # Compute the KL Divergence. We subtract out D * log(2 * pi) to get rid
        # of the extra unwanted constant term that ExactGPMarginalLogLikelihood gives us.
        # res = 0.5 * (trace - log_det_covar1 - 2 * K_part - (1 + math.log(2 * math.pi)) * D)
        res = 0.5 * (trace_logdet_quadform - log_det_covar1 - D)

        return res
