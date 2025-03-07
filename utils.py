"""Utility functions used in the codebase of 'Domain Adaptation and Entanglement: An Optimal Transport Perspective."""

import time
import torch
import numpy as np
import glog as log
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.func import functional_call, hessian

# Necessary in mac osx to be able close figures in emacs
matplotlib.use(backend="QtAgg", force=True)


class GenericDataset(Dataset):
    def __init__(self, X_train, y_train, transform=None):
        self.data = X_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, targets

    def __len__(self):
        return self.data.shape[0]


# Checks that the same network architecture used for both source *and* target can achieve high accuracy
def train_model_on_source_and_target(model, loss_fun, scenario, device, num_epochs):
    save_path = "save_files/" + scenario.name + "/train_on_both/" + model.name + ".pth"
    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt = {
        "name": "ADAM",
        "method": adam,
        "num_epochs": num_epochs,
        "batch_size": scenario.dataloader_options["batch_size"],
    }
    combined_data = torch.utils.data.ConcatDataset([scenario.source_data, scenario.target_data])
    dataloader = DataLoader(combined_data, **scenario.dataloader_options)

    try:
        log.info(f"Saved model found! Loading parameters from file: {save_path}")
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path))
    except:
        log.info(f"Save file {save_path} not found. Training from scratch...")
        train(
            dataloader,
            model,
            loss_fun,
            opt,
            num_epochs=num_epochs,
            device=device,
            params=None,
            report_every=10,
            report_acc=False,
        )
        log.info(f"Saving parameters to file: {save_path}")
        torch.save(model.state_dict(), save_path)

    report_acc(scenario, model, loss_fun, device)


def train_model_on_source(model, loss_fun, scenario, device, num_epochs):
    save_path = "save_files/" + scenario.name + "/" + model.name + ".pth"
    try:
        # Load parameters from a file
        model.load_state_dict(torch.load(save_path))
        log.info(f"Saved model found! Loading parameters from file: {save_path}")
    except:
        log.info(f"Either model has changed or save file {save_path} not found. Training from scratch...")
        adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt = {
            "name": "ADAM",
            "method": adam,
            "num_epochs": num_epochs,
            "batch_size": scenario.dataloader_options["batch_size"],
        }
        params = dict(model.named_parameters())  # We actually don't need it for SGD
        train(
            scenario.source_dataloader,
            model,
            loss_fun,
            opt,
            num_epochs,
            device=device,
            params=params,
            report_every=10,
        )
        # Report accuracy/loss on whole training dataset
        test(scenario.source_dataloader, model, loss_fun, device)
        log.info(f"Saving parameters to file: {save_path}")
        torch.save(model.state_dict(), save_path)
    return model


def report_acc(scenario, model, loss_fun, device, report_train=False):
    if report_train:
        # These are very slow
        print(f"Reporting accuracy/loss on source {scenario.source_name} training dataset...")
        test(scenario.source_dataloader, model, loss_fun, device)

        print(f"Reporting accuracy/loss on target {scenario.target_name} training dataset...")
        test(scenario.target_dataloader, model, loss_fun, device)

    print(f"Reporting accuracy/loss on {scenario.source_name} test dataset...")
    loss_source, acc_source = test(scenario.source_test_dataloader, model, loss_fun, device)

    print(f"Reporting accuracy/loss on {scenario.target_name} test dataset...")
    loss_target, acc_target = test(scenario.target_test_dataloader, model, loss_fun, device)
    return loss_source, acc_source, loss_target, acc_target


# Terminate when gradient norm falls below a threshold
# Does not fix the number of epochs, hence the name 'flexible'
# Uses only SGD for now
def flex_train(opt, model, loss_fun, X, y, thresh_norm=1e-3, max_epochs=1000, debug=False):
    for i in range(max_epochs):
        loss = loss_fun(model(X), y)
        loss.backward()
        opt.step()
        grad_norm = 0
        for w in model.parameters():
            grad_norm += torch.sum(w.grad**2)
        grad_norm = torch.sqrt(grad_norm)
        if debug:
            print(f"Iter: {i}, grad norm: {grad_norm}, thresh: {thresh_norm}")
        opt.zero_grad()
        if grad_norm < thresh_norm:
            break


def train(dataloader, model, loss_fun, optimizer, num_epochs, device, params=None, report_every=1, report_acc=True):
    print(f"==== {optimizer['name']} ====")
    size = len(dataloader.dataset)
    t0 = time.perf_counter()
    model.train()
    for epoch in range(num_epochs):
        # print(f"Epoch {epoch+1}\n-------------------------------")
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), one_hot(y, model.num_classes).to(device)
            # TODO: Is this necessary?
            X = X.contiguous()
            loss, params = step(optimizer, model, params, loss_fun, X, y)
            if batch % report_every == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} epoch:{epoch+1} [{current:>5d}/{size:>5d}]")
        if report_acc is True:
            print("Train dataset metrics:")
            test(dataloader, model, loss_fun, device)
    print(f"Method took {time.perf_counter() - t0} sec")


def test(dataloader, model, loss_fun, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    num_points = 0
    # We can't use this when we use a sampler to up/downsample the dataset!!!
    # num_inputs = len(dataloader.dataset)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fun(pred, one_hot(y, model.num_classes)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_points += y.shape[0]
    test_loss /= num_batches
    correct /= num_points
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


# Convenient wrapper for stepping with first order AND custom second order methods
def step(optimizer, model, params, loss_fun, X, y):
    opt = optimizer["method"]
    # TODO: Check if optimizer is custom!
    if optimizer["name"] == "SGD" or optimizer["name"] == "ADAM":
        loss = loss_fun(model(X), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss, None
    else:
        params, loss = opt.step(model, params, loss_fun, X, y)
        return loss, params


# Convenience loss wrapper for 'functional' calls (where weight is accessed from outside the model class)
def compute_loss(params, model, loss_fun, inputs, targets):
    prediction = functional_call(model, params, (inputs,))
    return loss_fun(prediction, targets)


# Hessian of loss with respect to parameters
def compute_hessian(model, loss_fun, inputs, targets, params):
    f = lambda params: compute_loss(params, model, loss_fun, inputs, targets)
    hess = hessian(f)(params)
    return model.flatten_hessian(hess)


def one_hot(x, k):
    return torch.nn.functional.one_hot(x, k).float()


def report(num_classes, y_test, y_pred):
    num_test = len(y_test)
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_test):
        conf_mat[int(y_test[i]), int(y_pred[i])] += 1
    # Correct classification rate a.k.a. accuracy
    accuracy = np.sum(np.diag(conf_mat)) / num_test

    log.info(f"Conf mat: \n {conf_mat}")
    log.info(f"Accuracy: {accuracy}")


def viz_data(inputs, labels, model=None):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    if model is not None:
        preds = model(inputs).to("cpu")
    for i in range(cols * rows):
        if i < len(inputs):
            img, label = inputs[i].to("cpu"), labels[i].to("cpu")
            figure.add_subplot(rows, cols, i + 1)
            if model is not None:
                sorted_pred, sorted_indices = torch.sort(preds[i], descending=True)
                print(f"Predictions for sample {i}: {sorted_pred} at indices {sorted_indices}")
                plt.title(f"Label: {label}, prediction: {sorted_indices[0].item()}")
            else:
                plt.title(f"Label: {label}")
            plt.axis("off")
            if len(img.shape) == 3:
                # Permute first dimension to last in order to visualize color images with plt
                if img.shape[0] == 3:
                    plt.imshow(torch.permute(img, (1, 2, 0)))
                else:
                    plt.imshow(img.squeeze(), cmap="gray")
            else:
                plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def viz_transports(X_train, X_transport, img_shape):
    num_batch = X_train.shape[0]
    figure = plt.figure(figsize=(8, 8))
    num_viz = 3
    rows, cols = num_viz, 2
    idx = torch.randint(num_batch, size=(num_viz,))
    for i in range(num_viz):
        img = X_train[idx[i]]
        figure.add_subplot(rows, cols, 2 * i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title("MNIST")
        img_transport = nn.Unflatten(-1, img_shape)(X_transport[idx[i]])
        figure.add_subplot(rows, cols, 2 * i + 2)
        plt.imshow(img_transport, cmap="gray")
        plt.title("MNIST -> USPS")
    plt.show()


# Plots the contours of the model predictions (in terms of probability)
def plot_prob_2d(model, X, ax, color_map, h=0.02, alpha=0.8):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h), indexing="xy")

    model.soft_max = True
    X_input = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    probs = model(X_input)[:, 1]
    model.soft_max = False

    # Put the result into a color plot
    probs = probs.reshape(xx.shape)
    ax.contourf(xx, yy, probs.detach().numpy(), cmap=color_map, alpha=alpha)


def plot_points_2d(
    X_test=None, y_test=None, X_train=None, y_train=None, title="", ax=None, color_map=plt.cm.RdBu, draw_idx=True
):
    if ax is None:
        ax = plt.subplot(111)
    ax.set_title(title)

    if X_train is not None:
        ax.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c=y_train,
            cmap=color_map,
            alpha=1.0,
            marker="+",
        )
        if draw_idx:
            for i in range(X_train.shape[0]):
                ax.text(X_train[i, 0], X_train[i, 1], str(i))
    # and testing points
    if X_test is not None:
        ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_test,
            cmap=color_map,
            alpha=1.0,
            edgecolors="black",
            s=25,
        )
        if draw_idx:
            for i in range(X_test.shape[0]):
                ax.text(X_test[i, 0], X_test[i, 1], str(i))


def gen_gauss_covariates(mean, var, num_samples):
    if np.isscalar(var):
        return mean + np.sqrt(var) * np.random.randn(num_samples)


def draw_gaussians():
    return mean + np.linalg.cholesky(var) @ np.random.randn(num_samples)
    num_tr = 150
    mean_train = 1.0
    var_train = 1 / 4.0
    x_train = gen_gauss_covariates(mean=mean_train, var=var_train, num_samples=num_tr)
    num_test = 150
    mean_test = 2.0
    var_test = 1 / 2.0
    x_test = gen_gauss_covariates(mean=mean_test, var=var_test, num_samples=num_tr)
    p_train = np.zeros(num_tr)
    p_test = np.zeros(num_tr)
    for i in range(num_tr):
        p_train[i] = p_gauss(x_train[i], mean_train, var_train)
        p_test[i] = p_gauss(x_test[i], mean_test, var_test)

    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.yaxis.grid(False)
    ax.scatter(x_train, p_train, c="b", label="p_source")
    ax.scatter(x_test, p_test, c="r", label="p_target")
    ax.legend()
    plt.show()
