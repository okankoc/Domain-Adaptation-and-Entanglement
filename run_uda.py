"""Script for testing unsupervised domain adaptation algorithms in several different distribution shift scenarios."""

import types
import numpy as np
import torch
import torchvision.models
import ot
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

import utils
import shifts
from adapt import JDOT, DANN, Oracle, WRR, MMD, ReverseKL
from models.conv import ConvNet, ConvNet2, LeNet, SmallCNN, ConvDomainClassifier
from models.prob_conv import ProbConvNet, ProbConvNet2, ProbLeNet, ProbSmallCNN
from models.mlp import MultiLayerPerceptron as MLP
from models.mlp import ProbMultiLayerPerceptron as PMLP

# Necessary in mac osx to be able close figures in emacs
matplotlib.use(backend="QtAgg", force=True)

# POT library gives errors!
DISABLE_ENTANGLE = True


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=1)))


@torch.no_grad()
def calc_entanglement(method, scenario, device, loss_fun, blur, num_batches_to_est=30):
    # import geomloss
    # ot_loss = geomloss.SamplesLoss(loss="sinkhorn", p=1, blur=blur)
    num_batches = 0
    entanglement = 0.0
    w_marginal = 0.0
    print("Calculating average entanglement and WRR risk values")
    for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_test_dataloader, scenario.target_test_dataloader):
        X_train, y_train = X_train.to(device), utils.one_hot(y_train.to(device), scenario.num_classes)
        X_shift, y_shift = X_shift.to(device), utils.one_hot(y_shift.to(device), scenario.num_classes)
        num_batches += 1

        pred_train = method.model.to(device)(X_train)
        pred_shift = method.model.to(device)(X_shift)

        # USE GEOMLOSS
        # WRR += loss_fun.to(device)(pred_train, y_train).item()
        # w_marginal = ot_loss(pred_train, pred_shift)

        # USE POT
        num_source = X_train.shape[0]
        num_target = X_shift.shape[0]
        w_source = torch.ones(num_source) / num_source
        w_target = torch.ones(num_target) / num_target
        cost_mat = ot.utils.euclidean_distances(pred_train.to("cpu"), pred_shift.to("cpu"), squared=False)
        scale = torch.max(cost_mat)
        cost_mat = cost_mat / scale
        prob_mat = ot.emd(a=w_source, b=w_target, M=cost_mat).type(torch.float)
        w_marginal += torch.sum(prob_mat * cost_mat * scale)

        # With GEOMLOSS we cannot get the optimal map
        # pred_and_label_train = torch.concatenate((pred_train, y_train), axis=1)
        # pred_and_label_shift = torch.concatenate((pred_shift, y_shift), axis=1)
        # w_joint = ot_loss(pred_and_label_train, pred_and_label_shift)
        # entanglement += w_joint - w_marginal

        costs_y = ot.utils.euclidean_distances(y_train.to("cpu"), y_shift.to("cpu"), squared=False)
        entanglement += torch.sum(prob_mat * costs_y)

        if num_batches % 10 == 0:
            print(
                f"Batch index: {num_batches}, avg W_marginal_dist: {w_marginal / num_batches}, avg entanglement: {entanglement / num_batches}"
            )
        if num_batches >= num_batches_to_est:
            break
    avg_entanglement = entanglement / num_batches
    avg_w_marginal = w_marginal / num_batches
    print(f"Average W marginal: {avg_w_marginal}")
    print(f"Entanglement est.: {avg_entanglement}")
    return avg_entanglement, avg_w_marginal


def load_model(model, scenario):
    save_path = "save_files/" + scenario.name + "/" + model.name + ".pth"
    model.load_state_dict(torch.load(save_path))
    return model


def save_model(model, scenario):
    save_path = "save_files/" + scenario.name + "/" + model.name + ".pth"
    torch.save(model.state_dict(), save_path)


def calc_results(scenario, methods, loss_fun, device):
    results = torch.zeros(6, len(methods))
    for i, method in enumerate(methods):
        print("===============================")
        print(f"Method {method.name}")
        loss_s, acc_s, loss_t, acc_t = utils.report_acc(scenario, method.model, loss_fun, device)
        if DISABLE_ENTANGLE is True:
            ent, w_marginal = 0.0, 0.0
        else:
            ent, w_marginal = calc_entanglement(method, scenario, device, loss_fun, blur=1e-6)
        results[:, i] = torch.tensor([acc_s, acc_t, loss_s, loss_t, w_marginal, ent])
    return results


def get_methods(model, loss_fun, device, num_epochs, num_batches, num_classes):
    if "mlp" in model.name:
        use_layer = -2
        layer_to_apply_disc = -2
    else:
        use_layer = "last_features"
        layer_to_apply_disc = "flatten"
    # Prepare adaptation methods
    methods = []
    # methods.append(Oracle(model, loss_fun, device, learning_rate=1e-4, mode='LJE', num_classes=num_classes))
    # methods.append(Oracle(model, loss_fun, device, learning_rate=1e-4, mode='ERM', num_classes=num_classes))
    # methods.append(Oracle(model, loss_fun, device, learning_rate=1e-4, mode='CCA', num_classes=num_classes))
    methods.append(MMD(model, loss_fun, device, learning_rate=1e-4, alpha=1.0))
    # methods.append(
    #     ReverseKL(
    #         model,
    #         device,
    #         learning_rate=1e-3,
    #         alpha_reverse=0.1,
    #         alpha_forward=0.1,
    #         augment_softmax=0.0).to(device))
    # methods.append(
    #     JDOT(
    #         model,
    #         loss_fun,
    #         device,
    #         alpha=0.001,
    #         lamb=0.001,
    #         learning_rate=1e-4,
    #         use_layer=use_layer,
    #         add_source_loss=True,
    #         use_squared_dist=True,
    #     )
    # )
    # discriminator = MLP([50, 10, 2], nn.ReLU())
    # discriminator = ConvDomainClassifier()
    # Track features: 9 for ConvNet2 used optionally in MNIST-M, 15 for ConvNet, 1 for the 256x100x50x10 MLP used in bidirectional USPS shifts
    # methods.append(
    #     DANN(
    #         model,
    #         loss_fun,
    #         discriminator=discriminator.to(device),
    #         layer_to_apply_disc=layer_to_apply_disc,
    #         device=device,
    #         learning_rate=1e-4,
    #         num_epochs=num_epochs,
    #         num_batches=num_batches,
    #     )
    # )
    # methods.append(
    #     WRR(
    #         model,
    #         loss_fun,
    #         learning_rate=1e-4,
    #         device=device,
    #         p=2,
    #         reg=0.05)
    # )
    return methods


def run_shift(methods, loss_fun, scenario, device, num_epochs, results):
    batch_idx = 0
    # Initial prediction (useful for pre-trained model)
    # calc_results(scenario, methods, loss_fun, device)
    # Run adaptation
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        for (X_train, y_train), (X_shift, y_shift) in zip(scenario.source_dataloader, scenario.target_dataloader):
            X_train, X_shift, y_train, y_shift = (
                X_train.to(device),
                X_shift.to(device),
                utils.one_hot(y_train.to(device), scenario.num_classes),
                utils.one_hot(y_shift.to(device), scenario.num_classes),
            )
            for method in methods:
                method.adapt(X_train, y_train, X_shift, y_shift)

            if batch_idx % 10 == 0:
                print(f"Batch id: {batch_idx}")
            batch_idx += 1

        results[:, :, epoch] = calc_results(scenario, methods, loss_fun, device)

    # In case we only want to compute entanglement at the very end
    # save_model(methods[0].model, scenario)
    # ent, wrr = calc_entanglement(methods, scenario, device, loss_fun, blur=1e-6)

    return results


def plot_loss_table2(results, method, save_name):
    num_runs, _, num_methods, num_epochs = results.shape
    results = torch.tensor(results)

    # results are stored as: [acc_s, acc_t, loss_s, loss_t, w_marginal, ent]
    fig, ax = plt.subplots()
    stds, means = torch.std_mean(results[:, 3, 0, :], dim=0)
    ax.errorbar(x=np.arange(num_epochs), y=means, yerr=stds, label="Target loss")
    std_e, mean_e = torch.std_mean(results[:, 5, 0, :], dim=0)
    ax.errorbar(x=np.arange(num_epochs), y=mean_e, yerr=std_e, label="Entanglement")

    loss_s = results[:, 2, 0, :]
    w_marginal = results[:, 4, 0, :]
    wrr_est = w_marginal + loss_s
    std_wrr, mean_wrr = torch.std_mean(wrr_est, dim=0)
    ax.errorbar(x=np.arange(num_epochs), y=mean_wrr, yerr=std_wrr, label="WRR value")

    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(save_name + ".pdf", format="pdf")


def plot_acc_table1(results, methods, save_name):
    num_runs, _, num_methods, num_epochs = results.shape
    results = torch.tensor(results)

    # TODO: Figure out the scale to use for WRR!
    fig, ax = plt.subplots()
    for i in range(num_methods):
        stds, means = torch.std_mean(results[:, 1, i, :], dim=0)
        ax.errorbar(x=np.arange(num_epochs), y=means, yerr=stds, label=methods[i].name)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(save_name + ".pdf", format="pdf")

    # Plot entanglement
    fig, ax = plt.subplots()
    for i in range(num_methods):
        std_e, mean_e = torch.std_mean(results[:, -1, i, :], dim=0)
        plt.errorbar(x=np.arange(num_epochs), y=mean_e, yerr=std_e, label=methods[i].name)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.savefig(save_name + "_entanglement.pdf", format="pdf")


def run_uda(scenario, loss_fun, num_epochs, num_runs, device):
    # Optionally train model on only source data before running domain adaptation
    # utils.train_model_on_source(model, loss_fun, scenario, device, num_epochs=5)
    # Optionally train model on both source and target data to inspect the optimized model
    # model2 = model.copy(device)
    # utils.train_model_on_source_and_target(model2, loss_fun, scenario, device, num_epochs=5)

    # calc_entanglement(methods, scenario, device, loss_fun, blur=1e-6)
    # return

    # source acc, target acc, source_loss, target_loss, marginal_dist and entanglement estimates are stored in results
    # for each run and epoch and method
    num_batches = min(len(scenario.source_dataloader), len(scenario.target_dataloader))
    model = init_model(scenario, device)
    methods = get_methods(model, loss_fun, device, num_epochs, num_batches, scenario.num_classes)
    results = torch.zeros(num_runs, 6, len(methods), num_epochs)
    for i in range(num_runs):
        torch.manual_seed(seed=i)
        print(f"Experiment num {i}")
        results[i] = run_shift(methods, loss_fun, scenario, device, num_epochs, results[i])
        model = init_model(scenario, device)
        methods = get_methods(model, loss_fun, device, num_epochs, num_batches, scenario.num_classes)
    return results, methods


def init_scenario(dataloader_options):
    # scenario = shifts.MNIST_to_USPS(dataloader_options, use_sampler=True, class_balanced=False)
    # scenario = shifts.USPS_to_MNIST(dataloader_options, use_sampler=True)
    # scenario = shifts.MNIST_to_MNIST_M(dataloader_options, preprocess=False)
    # scenario = shifts.SVHN_to_MNIST(dataloader_options, class_balanced=False)
    scenario = shifts.CIFAR_CORRUPT(dataloader_options, corruptions=["fog", "frost", "snow"])
    # scenario = shifts.PORTRAITS(dataloader_options, size=(32,32), train_ratio=0.8)
    # scenario = shifts.OFFICEHOME(dataloader_options, size=(224,224))
    return scenario


def init_resnet(device, num_classes):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.num_classes = num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # for OfficeHomeDataset
    model.name = "RESNET18"
    model.features = []

    def track_features(model, layer_id):
        # Ignores the layer_id!
        # Register hooks for the layers you're interested in
        def fun(module, inputs, outputs):
            model.features = inputs[0]

        hook = model.fc.register_forward_hook(fun)

    def copy(model, device):
        new_model = torchvision.models.resnet18().to(device)
        new_model.fc = nn.Linear(model.fc.in_features, model.fc.out_features).to(device)
        new_model.load_state_dict(model.state_dict())
        new_model.num_classes = model.num_classes
        new_model.name = "RESNET18"
        new_model.copy = types.MethodType(copy, new_model)
        new_model.track_features = types.MethodType(track_features, new_model)
        new_model.features = []
        return new_model

    model.copy = types.MethodType(copy, model)
    model.track_features = types.MethodType(track_features, model)
    model.to(device)
    return model


def init_model(scenario, device):
    # model = MLP(layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes], f_nonlinear=nn.ReLU()).to(device)
    model = ConvNet(num_classes=scenario.num_classes).to(device)
    # model = ConvNet2(num_classes=scenario.num_classes).to(device)
    # model = LeNet(num_classes=scenario.num_classes).to(device)
    # model = SmallCNN(num_classes=scenario.num_classes).to(device)
    # load_model(model, scenario)
    # model = init_resnet(device, scenario.num_classes)

    # Probabilistic Representation Networks to test with reverse-KL algorithm
    # model = PMLP(layer_sizes=[scenario.input_size, 200, 200, scenario.num_classes], f_nonlinear=nn.ReLU()).to(device)
    # model = ProbConvNet(num_classes=scenario.num_classes).to(device)
    # model = ProbConvNet2(num_classes=scenario.num_classes).to(device)
    # model = ProbLeNet(num_classes=scenario.num_classes).to(device)
    # model = ProbSmallCNN(num_classes=scenario.num_classes).to(device)

    return model


def run_single_case_table1():
    loss_fun = nn.CrossEntropyLoss()

    device = "mps"
    # device = 'cpu'
    dataloader_options = {"batch_size": 64, "shuffle": False, "drop_last": True}

    scenario = init_scenario(dataloader_options)
    model = init_model(scenario, device)
    num_epochs = 10
    save_name = "results/table1/" + scenario.name + "_" + model.name
    results, methods = run_uda(scenario, loss_fun, num_epochs=num_epochs, num_runs=3, device=device)
    np.save(save_name, results)
    # results = np.load(save_name + ".npy")
    # plot_loss_table2(results, methods[0], save_name)
    plot_acc_table1(results, methods, save_name)


def run_single_case_table2():
    loss_fun = EuclideanLoss()

    device = "mps"
    # device = 'cpu'
    dataloader_options = {"batch_size": 64, "shuffle": True, "drop_last": True}

    scenario = init_scenario(dataloader_options)
    model = init_model(scenario, device)
    num_epochs = 5
    save_name = "results/table2/" + scenario.name + "_" + model.name
    results, methods = run_uda(scenario, loss_fun, num_epochs=num_epochs, num_runs=1, device=device)
    np.save(save_name, results)
    # results = np.load(save_name + ".npy")
    plot_loss_table2(results, methods[0], save_name)


def generate_results():
    num_epochs = 10
    # Different loss functions to test here
    loss_fun = nn.CrossEntropyLoss()
    # loss_fun = nn.MSELoss()
    # loss_fun = EuclideanLoss()

    device = "cpu"  # 'mps'
    dataloader_options = {"batch_size": 64, "shuffle": True, "drop_last": True}

    num_scenarios = 6
    scenarios = [None for i in range(num_scenarios)]
    scenarios[0] = shifts.MNIST_to_USPS(dataloader_options, gen_acc_curve=False)
    scenarios[1] = shifts.USPS_to_MNIST(dataloader_options, gen_acc_curve=False)
    scenarios[2] = shifts.MNIST_to_MNIST_M(dataloader_options, gen_acc_curve=False, preprocess=False)
    scenarios[3] = shifts.SVHN_to_MNIST(dataloader_options, gen_acc_curve=False)
    scenarios[4] = shifts.CIFAR_CORRUPT(dataloader_options, gen_acc_curve=False, corruptions=["fog", "frost", "snow"])
    scenarios[5] = shifts.PORTRAITS(dataloader_options, gen_acc_curve=False, size=(32, 32), train_ratio=0.8)

    for scenario in scenarios:
        num_models = 5
        models = [None for i in range(num_models)]
        models[0] = MLP(layer_sizes=[scenario.input_size, 200, 100, scenario.num_classes], f_nonlinear=nn.ReLU()).to(
            device
        )
        models[1] = ConvNet(num_classes=scenario.num_classes).to(device)
        models[2] = ConvNet2(num_classes=scenario.num_classes).to(device)
        models[3] = LeNet(num_classes=scenario.num_classes).to(device)
        models[4] = SmallCNN(num_classes=scenario.num_classes).to(device)
        # load_model(model, scenario)
        for model in models:
            save_name = "results/table1" + scenario.name + "_" + model.name
            results, methods = run_uda(scenario, model, loss_fun, num_epochs=num_epochs, num_runs=3, device=device)
            np.save(save_name, results)
            # results = np.load(save_name + ".npy")

            plot_acc_table1(results, methods, save_name)
            # break
        # break


if __name__ == "__main__":
    # import faulthandler
    # faulthandler.enable()
    torch.set_printoptions(precision=4, sci_mode=False)

    run_single_case_table1()
    # run_single_case_table2()
    # generate_results()
