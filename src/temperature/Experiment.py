import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from tqdm import trange

import src.temperature.losses as losses

class Experiment():

    def __init__(self, name, trainset, valset, testset, model, test_loss, optimizer, outdir, device, regloss=False, coeffs=None, windloss=False, numvars=2):
        self.name = name
        self.train_loader = trainset
        self.val_loader = valset
        self.test_loader = testset
        self.model = model
        self.test_loss = test_loss
        self.optimizer = optimizer
        
        # Set up a directory for the experiment
        self.dir_setup(outdir)
        self.device = device
        
        self.loss_fn = losses.squared_error_loss 
        
        self.regloss = regloss
        self.coeffs = coeffs
        self.windloss = windloss

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        # Check how many items the DataLoaders contain
        self.num_vars = numvars #len(next(iter(trainset)))

    def dir_setup(self, parent):
        self.outdir = parent + "/" + self.name
        os.mkdir(self.outdir)
        print("Created new directory to save model states and results: " + self.outdir)

        os.mkdir(self.outdir + "/" + "images")
        os.mkdir(self.outdir + "/" + "losses")

    # TODO: print loss components, eg. regloss for each term and windloss in addition to total loss
    def train_loop(self):
        size = len(self.train_loader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.train()

        losses = []
        
        # Case 1: SST data, no wind
        if self.num_vars == 2:

        #if not self.wind_in_data:
            for batch, (X, y) in enumerate(self.train_loader):
                # Compute prediction and loss
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)

                w_pred = outputs[0]
                y_pred = outputs[1]

                loss = self.loss_fn(y_pred, y, reg=self.regloss, w_pred=w_pred, coeffs=self.coeffs)
                losses.append(loss.item())

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            for batch, (X, y, W) in enumerate(self.train_loader):
                # Compute prediction and loss
                X = X.to(self.device)
                y = y.to(self.device)
                W = W.to(self.device)

                outputs = self.model(X)

                w_pred = outputs[0]
                y_pred = outputs[1]

                loss = self.loss_fn(y_pred, y, reg=self.regloss, w_pred=w_pred, coeffs=self.coeffs, wdl=self.windloss, w=W)
                losses.append(loss.item())

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return losses

    def val_loop(self):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        losses = []
        num_loops = 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also reduces unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            if self.num_vars == 2: #not self.wind_in_data:
                for X, y in self.val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(X)
                    y_pred = outputs[1]

                    step_loss = self.loss_fn(y_pred, y).item()
                    losses.append(step_loss)
                    num_loops += 1
                return np.mean(losses)
            else:
                for X, y, W in self.val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    W = W.to(self.device)
                    outputs = self.model(X)
                    w_pred = outputs[0]
                    y_pred = outputs[1]

                    step_loss = self.loss_fn(y_pred, y, wdl=self.windloss, w_pred=w_pred, w=W).item()
                    losses.append(step_loss)
                    num_loops += 1
                return np.mean(losses)

    def test(self):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        self.model.eval()
        losses = []
        num_loops = 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also reduces unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            if self.num_vars == 2: #not self.wind_in_data:
                for X, y in self.test_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(X)
                    y_pred = outputs[1]

                    step_loss = self.test_loss(y_pred, y).item()
                    print("Item:", num_loops, "Loss:", step_loss)
                    losses.append(step_loss)
                    num_loops += 1
            else:
                for X, y, W in self.test_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    outputs = self.model(X)
                    y_pred = outputs[1]

                    step_loss = self.test_loss(y_pred, y).item()
                    print("Item:", num_loops, "Loss:", step_loss)
                    losses.append(step_loss)
                    num_loops += 1

        return losses

    def save_results(self, losses, fname):
        np.save(fname + ".npy", losses)

    def save_model_state(self, epoch):

        fname = self.outdir + "/" + "modelCheckpoints/" + self.name + "_epoch_" + str(epoch) + ".pt"

        try:
            os.mkdir(self.outdir + "/" + "modelCheckpoints")
        except FileExistsError as e:
            pass
        finally:
            torch.save(self.model, fname)

        print("Saved checkpoint at epoch", epoch)

    def run(self, epochs):
        print("Running experiment: " + self.name + "...")

        # -------------------- Training -------------------------

        print("Training over " + str(epochs) + " epochs...")

        for epoch in trange(epochs, desc="Training", unit="Epoch"): 

            epoch_losses = self.train_loop()
            epoch_mean = np.round(np.mean(epoch_losses), 5)

            fname = self.outdir + "/losses/train_epoch_" + str(epoch)  
            self.save_results(epoch_losses, fname)

            val_loss = np.round(self.val_loop(), 5)
            fname = self.outdir + "/losses/val_epoch_" + str(epoch)
            self.save_results(val_loss, fname)

            print("Mean Training Loss:", epoch_mean)
            print("Validation Loss: ", val_loss)

            self.train_losses.append(epoch_mean)
            self.val_losses.append(val_loss)

            if epoch % 100 == 0:  
                self.save_model_state(epoch)

        self.save_results(self.train_losses, self.outdir + "/losses/mean_train_losses")
        self.save_results(self.val_losses, self.outdir + "/losses/mean_val_losses")

        # -------------------- Testing ---------------------------

    def visualize_examples(self, examples, fname):
        self.model.eval()

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        with torch.no_grad():
            if self.num_vars == 2: #not self.wind_in_data:
                for idx, (X, y) in enumerate(examples):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    if torch.cuda.is_available():
                        self.model.cuda()

                    outputs = self.model(X)
                    W_pred = outputs[0]
                    y_pred = outputs[1]

                    #step_loss = self.test_loss(y_pred, y).item()

                    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 12))
                    ax1 = axes[0][0]
                    ax2 = axes[0][1]
                    ax3 = axes[1][0]
                    ax4 = axes[1][1]
                    ax5 = axes[2][0]
                    ax6 = axes[2][1]
                    
                    # X[2] and X[3] (second-to-last and last input days)
                    x2_plot = ax1.imshow(X[0][2].cpu())
                    ax1.set_title("Penultimate Input Image")
                    fig.colorbar(x2_plot, ax=ax1)
                    ax1.set(yticks=[0,20,40,60])
                    
                    x3_plot = ax2.imshow(X[0][3].cpu())
                    ax2.set_title("Last Input Image")
                    fig.colorbar(x3_plot, ax=ax2)
                    ax2.set(yticks=[0,20,40,60])
                    
                    yplot = ax3.imshow(y[0].cpu())
                    ax3.set_title("Ground truth")
                    fig.colorbar(yplot, ax=ax3)
                    ax3.set(yticks=[0,20,40,60])
                    
                    ypred_plot = ax4.imshow(y_pred[0].cpu())
                    ax4.set_title("Prediction")
                    fig.colorbar(ypred_plot, ax=ax4)
                    ax4.set(yticks=[0,20,40,60])
                    
                    wupred_plot = ax5.imshow(W_pred[0][0].cpu(), cmap="magma")
                    ax5.set_title("W(u) Prediction")
                    fig.colorbar(wupred_plot, ax=ax5)
                    ax5.set(yticks=[0,20,40,60])
                    
                    wvpred_plot = ax6.imshow(W_pred[0][1].cpu(), cmap="magma")
                    ax6.set_title("W(v) Prediction")
                    fig.colorbar(wvpred_plot, ax=ax6)
                    ax6.set(yticks=[0,20,40,60])

                    fig.savefig(self.outdir + "/images/" + fname + "_" + str(idx) + ".png")
                    plt.close()
            else:
                for idx, (X, y, W) in enumerate(examples):
                    X = X.to(self.device)
                    W = W.to(self.device)
                    y = y.to(self.device)

                    if torch.cuda.is_available():
                        self.model.cuda()

                    outputs = self.model(X)  # outputs W, y_pred

                    W_pred = outputs[0]        
                    y_pred = outputs[1]

                    #step_loss = self.test_loss(y_pred, y).item()

                    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 16))
                    ax1 = axes[0][0]
                    ax2 = axes[0][1]
                    ax3 = axes[1][0]
                    ax4 = axes[1][1]
                    ax5 = axes[2][0]
                    ax6 = axes[2][1]
                    ax7 = axes[3][0]
                    ax8 = axes[3][1]
                    
                    # X[2] and X[3] (second-to-last and last input days)
                    x2_plot = ax1.imshow(X[0][2].cpu())
                    ax1.set_title("Penultimate Input Image")
                    fig.colorbar(x2_plot, ax=ax1)
                    ax1.set(yticks=[0,20,40,60])
                    
                    x3_plot = ax2.imshow(X[0][3].cpu())
                    ax2.set_title("Last Input Image")
                    fig.colorbar(x3_plot, ax=ax2)
                    ax2.set(yticks=[0,20,40,60])
                    
                    # y vs y_pred
                    y_plot = ax3.imshow(y[0].cpu())  
                    ax3.set_title("Temperature Ground truth")
                    fig.colorbar(y_plot, ax=ax3)
                    ax3.set(yticks=[0,20,40,60])
                    
                    ypred_plot = ax4.imshow(y_pred[0].cpu())  
                    ax4.set_title("Temperature Prediction")
                    fig.colorbar(ypred_plot, ax=ax4)
                    ax4.set(yticks=[0,20,40,60])
                    
                    # w_u vs w_u_pred
                    wu_plot = ax5.imshow(W[0][0].cpu(), cmap="magma")  
                    ax5.set_title("W(u) Ground truth")
                    fig.colorbar(wu_plot, ax=ax5)
                    ax5.set(yticks=[0,20,40,60])
                    
                    wupred_plot = ax6.imshow(W_pred[0][0].cpu(), cmap="magma")  
                    ax6.set_title("W(u) Prediction")
                    fig.colorbar(wupred_plot, ax=ax6)
                    ax6.set(yticks=[0,20,40,60])
                    
                    # w_v vs w_v_pred
                    wv_plot = ax7.imshow(W[0][1].cpu(), cmap="magma")  
                    ax7.set_title("W(v) Ground truth")
                    fig.colorbar(wv_plot, ax=ax7)
                    ax7.set(yticks=[0,20,40,60])
                    
                    wvpred_plot = ax8.imshow(W_pred[0][1].cpu(), cmap="magma")  
                    ax8.set_title("W(v) Prediction")
                    fig.colorbar(wvpred_plot, ax=ax8)
                    ax8.set(yticks=[0,20,40,60])

                    fig.savefig(self.outdir + "/images/" + fname + "_" + str(idx) + ".png")
                    plt.close()

    def plot_loss(self, title, xlab, legend_items, fname):
        losses_to_plot = [self.train_losses, self.val_losses]
        plt.plot(losses_to_plot[0])
        plt.plot(losses_to_plot[1])
        plt.title(title)

        plt.ylabel("Loss")
        plt.xlabel(xlab)
        plt.legend(legend_items, loc="upper left")

        plt.savefig(self.outdir + "/images/" + fname + ".png")
        plt.close()
        