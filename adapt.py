# Domain adaptation approaches
import torch
from torch import nn
from torch.autograd import Function
import torch.distributions
import torch.nn.functional as F
import numpy as np
import ot
import geomloss
import copy

import utils

# Wasserstein Marginal Distance regularized source risk minimization using model outputs (as opposed to DeepJDOT which is more complicated)
class WRR:
    def __init__(self, model, loss_fun, learning_rate, device, p=2, reg=0.05):
        self.model = model.copy(device)
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
        # For debugging
        self.name = "WRR"
        self.device = device
        self.p = p
        self.reg = reg

    def calc_ot(self, f_source, f_target):
        num_source = f_source.shape[0]
        num_target = f_target.shape[0]

        ### Python crashes regularly with POT so switching to GeomLoss
        ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=self.p, blur=self.reg)
        total_cost = ot_loss(f_source, f_target)

        """
        cost_mat = ot.utils.euclidean_distances(f_source, f_target, squared=self.use_squared_dist)
        scale = torch.max(cost_mat)
        cost_mat = cost_mat / scale
        # Weights of the points
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float)
        total_cost = torch.sum(prob_mat * cost_mat)
        """

        return total_cost

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        pred_source = self.model(X_source)
        f_source = torch.clone(pred_source)
        f_target = torch.clone(self.model(X_target))
        ot_cost = self.calc_ot(f_source, f_target)
        loss = self.loss_fun(pred_source, y_source) + ot_cost
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()


# Used to optimize DANN (features are optimized to maximize domain classifier error).
# TODO: Is this necessary? Replace with 2 optimizers: one min. and one max.
# Alternatively, pytorch guide suggests using 'hooks'
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN:
    def __init__(
        self, model, loss_fun, discriminator, layer_to_apply_disc, device, learning_rate, num_epochs, num_batches
    ):
        self.name = "DANN"
        self.device = device
        self.model = model.copy(device)
        self.model.track_features(layer_to_apply_disc)
        self.discriminator = discriminator.copy(device)
        self.opt_model = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.loss_class = copy.deepcopy(loss_fun).to(device)
        self.loss_domain = copy.deepcopy(loss_fun).to(device)
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.idx = 0
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.discriminator.parameters():
            p.requires_grad = True

    def forward_adversarial(self, X_data):
        epoch_idx = self.idx // self.num_batches
        p = (self.idx + epoch_idx * self.num_batches) / (self.num_epochs * self.num_batches)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        input_data = torch.as_tensor(X_data, dtype=torch.float32)
        class_output = self.model(input_data)
        # TODO: This gradient reversal seems very unnecessary if we maintain two optimizers
        reverse_feature = ReverseLayerF.apply(self.model.features, alpha)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        source_batch_size = X_source.shape[0]

        # Feeding in source inputs
        domain_label = utils.one_hot(torch.zeros(source_batch_size, device=self.device, dtype=torch.long), 2)
        class_output, domain_output = self.forward_adversarial(X_source)
        err_s_label = self.loss_class(class_output, y_source)
        err_s_domain = self.loss_domain(domain_output, domain_label)

        # Feeding in target labels
        target_batch_size = X_target.shape[0]
        domain_label = utils.one_hot(torch.ones(target_batch_size, device=self.device, dtype=torch.long), 2)
        _, domain_output = self.forward_adversarial(X_target)

        err_t_domain = self.loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        self.opt_model.step()
        self.opt_model.zero_grad()
        self.opt_disc.step()
        self.opt_disc.zero_grad()
        self.idx += 1


class ReverseKL(torch.nn.Module):
    def __init__(self, model, device, learning_rate, alpha_reverse, alpha_forward, augment_softmax):
        super(ReverseKL, self).__init__()
        self.model = model.copy(device)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
        self.name = "KL"
        self.device = device
        self.alpha_reverse = alpha_reverse  # as the reverse-KL-regularizer scale
        self.alpha_forward = alpha_forward
        self.augment_softmax = augment_softmax

    def compute_kl(self, mean_s, std_s, sample_s, distr_s, mean_t, std_t, sample_t, distr_t):
        mix_coeff_source = torch.distributions.categorical.Categorical(torch.ones(mean_s.shape[0], device=self.device))
        mixture_source = torch.distributions.mixture_same_family.MixtureSameFamily(mix_coeff_source, distr_s)
        mix_coeff_target = torch.distributions.categorical.Categorical(torch.ones(mean_t.shape[0], device=self.device))
        mixture_target = torch.distributions.mixture_same_family.MixtureSameFamily(mix_coeff_target, distr_t)
        kl_reg = self.alpha_reverse * (mixture_target.log_prob(sample_t) - mixture_source.log_prob(sample_t)).mean()
        if self.alpha_forward != 0.0:
            kl_reg += (
                self.alpha_forward * (mixture_source.log_prob(sample_s) - mixture_target.log_prob(sample_s)).mean()
            )
        return kl_reg

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        mean_s, std_s, sample_s, out_s, distr_s = self.model.forward_distr(X_source)
        mean_t, std_t, sample_t, out_t, distr_t = self.model.forward_distr(X_target)

        out_s = torch.softmax(out_s, 1)
        if self.augment_softmax != 0.0:
            scale_down = 1 - self.augment_softmax * out_s.shape[1]
            out_s = out_s * scale_down + self.augment_softmax
        err = F.nll_loss(torch.log(out_s), torch.argmax(y_source, dim=1))
        err += self.compute_kl(mean_s, std_s, sample_s, distr_s, mean_t, std_t, sample_t, distr_t)

        err.backward()
        self.opt.step()
        self.opt.zero_grad()


# TODO: For now we only use a Gaussian kernel
class MMD:
    def __init__(self, model, loss_fun, device, learning_rate, alpha):
        self.model = model.copy(device)
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0)
        self.name = "MMD"
        self.device = device
        self.alpha = alpha

    def calc_kernel(self, X_source, X_target):
        mat_dist = torch.cdist(X_source, X_target)
        kernel_mat = torch.zeros_like(mat_dist)
        gammas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for gamma in gammas:
            kernel_mat += torch.exp(-gamma * mat_dist)
        return kernel_mat

    def calc_mmd(self, pred_source, pred_target):
        K_source = self.calc_kernel(pred_source, pred_source).mean()
        K_target = self.calc_kernel(pred_target, pred_target).mean()
        K_source_target = self.calc_kernel(pred_source, pred_target).mean()
        return K_source + K_target - 2 * K_source_target

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        pred_source = self.model(X_source)
        pred_target = self.model(X_target)
        err = self.loss_fun(pred_source, y_source)
        err += self.alpha * self.calc_mmd(pred_source, pred_target)
        err.backward()
        self.opt.step()
        self.opt.zero_grad()


class Oracle:
    def __init__(self, model, loss_fun, device, learning_rate, mode, num_classes):
        self.mode = mode
        if mode == "ERM":
            self.name = "ERM"
        elif mode == "CCA":
            self.name = "CCA"
        else:
            self.name = "LJE"  # low-joint-error
        print(f"Initializing Oracle as {self.name}")
        self.num_classes = num_classes
        self.device = device
        self.model = model.copy(device)
        self.loss = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        for p in self.model.parameters():
            p.requires_grad = True

    def adapt(self, X_source, y_source, X_target, y_target=[]):
        pred_source = self.model(X_source)
        err = self.loss(pred_source, y_source)
        if len(y_target) > 0 and self.mode == "LJE":
            err += self.loss(self.model(X_target), y_target)
        elif len(y_target) > 0 and self.mode == "CCA":
            pred_target = self.model(X_target)
            w_dist = torch.zeros(self.num_classes)
            num_min_per_class = 5
            for i in range(self.num_classes):
                x_class_source = pred_source[y_source.argmax(dim=1) == i]
                x_class_target = pred_target[y_target.argmax(dim=1) == i]
                if x_class_source.shape[0] > num_min_per_class and x_class_target.shape[0] > num_min_per_class:
                    w_dist[i] = self.calc_w_distance(x_class_source, x_class_target)
            err += torch.max(w_dist)
        err.backward()
        self.opt.step()
        self.opt.zero_grad()

    def calc_w_distance(self, x_source, x_target):
        num_source = x_source.shape[0]
        num_target = x_target.shape[0]
        ot_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05)
        w_dist = ot_loss(x_source, x_target)

        # Weights of the points
        # w_source = torch.ones(num_source) / num_source
        # w_target = torch.ones(num_target) / num_target
        # cost_mat = ot.utils.euclidean_distances(x_source, x_target, squared=True)
        # scale = torch.max(cost_mat)
        # cost_mat = cost_mat / scale
        # prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float)
        # w_dist = torch.sqrt(torch.sum(prob_mat * cost_mat * scale))

        return w_dist


# Can we / should we do OT in the interior / last layers? Or all of them?
# Feature activations can be stored to speed up this implementation
class JDOT:
    def __init__(
        self,
        model,
        loss_fun,
        device,
        alpha,
        lamb,
        learning_rate,
        num_iter=1,
        use_layer=-2,
        add_source_loss=True,
        use_squared_dist=False,
    ):
        self.model = model.copy(device)
        self.loss_fun = copy.deepcopy(loss_fun)
        self.opt = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        )
        self.name = "JDOT"
        self.debug_loss = []
        self.debug_acc = []
        self.alpha = alpha
        self.lamb = lamb
        self.num_iter = num_iter
        self.model.track_features(use_layer)
        self.add_source_loss = add_source_loss
        self.use_squared_dist = use_squared_dist
        self.device = device

    def adapt(self, X_train, y_train, X_shift, y_shift=[]):
        num_target = X_shift.shape[0]
        results = {
            "acc": torch.zeros(self.num_iter),
            "loss": torch.zeros(self.num_iter),
            "w_dist": torch.zeros(self.num_iter),
        }
        for k in range(self.num_iter):
            # print(f"JDOT Iter: {k}")
            prob, w_dist = self.transport(X_train, X_shift, y_train)
            # Debugging
            if len(y_shift) != 0:
                y_pred = self.model(X_shift)
                results["loss"][k] = self.loss_fun(y_pred, y_shift)
                results["acc"][k] = torch.mean((y_pred.argmax(1) == y_shift.argmax(1)).type(torch.float))
                results["w_dist"][k] = w_dist
            loss = self.loss(X_train, y_train, X_shift, prob)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return results

    def transport(self, X_source, X_target, y_source):
        num_source = X_source.shape[0]
        num_target = X_target.shape[0]
        # Weights of the points
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target

        self.model(X_source)
        source_activations = torch.clone(self.model.features)
        pred_target = self.model(X_target)
        target_activations = torch.clone(self.model.features)
        cost_mat = ot.utils.euclidean_distances(source_activations, target_activations, squared=self.use_squared_dist)
        cost_mat = self.alpha * cost_mat + self.lamb * self.calc_loss_mat(y_source, pred_target)
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float).to(self.device)
        if self.use_squared_dist is True:
            return prob_mat, torch.sqrt(torch.sum(prob_mat * cost_mat))
        return prob_mat, torch.sum(prob_mat * cost_mat)

    def calc_loss_mat(self, y_source, y_pred):
        num_source = y_source.shape[0]
        num_target = y_pred.shape[0]
        ys = torch.repeat_interleave(y_source, num_target, dim=0)
        yt = y_pred.repeat(num_source, 1)
        # FIXME: This won't work if loss_fun is different from cross-entropy!!!
        loss_fun = nn.CrossEntropyLoss(reduction="none")
        return loss_fun(yt, ys).reshape(num_source, num_target)

    def loss(self, X_train, y_train, X_shift, prob_mat):
        source_loss = 0.0
        val = self.loss_fun(self.model(X_train), y_train)
        if self.add_source_loss is True:
            source_loss += val
        source_activations = torch.clone(self.model.features)

        idx_source, idx_target = torch.where(prob_mat)
        probs = prob_mat[torch.where(prob_mat)]
        y_pred = self.model(X_shift)
        layer_loss = torch.sum(probs * torch.dist(self.model.features[idx_target], source_activations[idx_source]))
        self.loss_fun.reduction = "none"
        losses = self.loss_fun(y_pred[idx_target], y_train[idx_source])
        self.loss_fun.reduction = "mean"
        if len(losses.shape) == 2:
            losses = torch.mean(losses, dim=1)
        target_loss = torch.sum(probs * losses)
        return source_loss + self.alpha * layer_loss + self.lamb * target_loss
