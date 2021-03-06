import math
import torch
from gpytorch.utils import LinearCG, StochasticLQ
from gpytorch.utils.toeplitz import interpolated_sym_toeplitz_mul, sym_toeplitz_derivative_quadratic_form
from torch.autograd import Function, Variable


class InterpolatedToeplitzGPMarginalLogLikelihood(Function):
    def __init__(self, W_left, W_right, num_samples=10):
        if isinstance(W_left, Variable):
            self.W_left = W_left.data
        else:
            self.W_left = W_left

        if isinstance(W_right, Variable):
            self.W_right = W_right.data
        else:
            self.W_right = W_right

        self.num_samples = num_samples

    def forward(self, toeplitz_column, labels, noise_diag):
        def mv_closure(v):
            return interpolated_sym_toeplitz_mul(toeplitz_column, v, self.W_left, self.W_right, noise_diag)

        self.save_for_backward(toeplitz_column, labels, noise_diag)

        mat_inv_y = LinearCG().solve(mv_closure, labels)
        # Inverse quad form
        res = mat_inv_y.dot(labels)
        # Log determinant
        ld, tr_inv = StochasticLQ(num_random_probes=10).evaluate(mv_closure,
                                                                 len(labels),
                                                                 [lambda x: x.log(), lambda x: x.pow(-1)])
        res += ld
        res += math.log(2 * math.pi) * len(labels)
        res *= -0.5

        self.mat_inv_y = mat_inv_y
        self.tr_inv = tr_inv
        self.mv_closure = mv_closure
        return labels.new().resize_(1).fill_(res)

    def backward(self, grad_output):
        grad_output_value = grad_output.squeeze()[0]
        toeplitz_column, labels, noise_diag = self.saved_tensors

        mat_inv_y = self.mat_inv_y
        mv_closure = self.mv_closure

        mat_grad = None
        y_grad = None
        noise_grad = None

        if self.needs_input_grad[0]:
            y_mat_inv_W_left = torch.dsmm(self.W_left.t(), mat_inv_y.unsqueeze(1)).t()
            W_right_mat_inv_y = torch.dsmm(self.W_right.t(), mat_inv_y.unsqueeze(1))
            quad_form_part = sym_toeplitz_derivative_quadratic_form(y_mat_inv_W_left.squeeze(),
                                                                    W_right_mat_inv_y.squeeze())
            log_det_part = torch.zeros(len(toeplitz_column))
            sample_matrix = torch.sign(torch.randn(len(labels), self.num_samples))

            left_vectors = torch.dsmm(self.W_left.t(), LinearCG().solve(mv_closure, sample_matrix)).t()
            right_vectors = torch.dsmm(self.W_right.t(), sample_matrix).t()

            for left_vector, right_vector in zip(left_vectors, right_vectors):
                log_det_part += sym_toeplitz_derivative_quadratic_form(left_vector,
                                                                       right_vector)

            log_det_part.div_(self.num_samples)

            mat_grad = quad_form_part - log_det_part
            mat_grad.mul_(0.5 * grad_output_value)

        if self.needs_input_grad[1]:
            # Need gradient with respect to labels
            y_grad = mat_inv_y.mul_(-grad_output_value)

        if self.needs_input_grad[2]:
            quad_form_part = mat_inv_y.dot(mat_inv_y)
            noise_grad = toeplitz_column.new().resize_(1).fill_(quad_form_part - self.tr_inv)
            noise_grad.mul_(0.5 * grad_output_value)

        return mat_grad, y_grad, noise_grad
