import datetime
import os
import torch
import torch.nn as nn

from src.models.BezConv import BezConv as BC
from src.dataloader import DataLoader
from src.temperature.Experiment import Experiment
import utils

device = utils.check_device()
outdir = "/projectnb/labci/Lucia/rainfall-pde-ml/experiments/"
os.chdir("/projectnb/labci/Lucia/rainfall-pde-ml/")

# ------------------------------ Set up directory for experiments -----------------------
try:
    os.mkdir(outdir + str(datetime.date.today()))
except FileExistsError as e:
    print("Already ran experiments today.")
finally:
    path = outdir + str(datetime.date.today())

# ------------------------------ Data Loaders -----------------------
DL = DataLoader.DataLoader(path, [0.8, 0.1, 0.1], 32, 4)
DL.create_training()
DL.create_validation()
DL.create_test()

EL = utils.create_example_loader()

# ~----- Experiment 1: no regularization for 100 epochs -----~
net1 = BC(device=device, coeffs=[0,0,0], hist=4).to(device)
optim1 = torch.optim.Adam(net1.parameters(), lr=1e-3)
exp1 = Experiment.Experiment(name="BezenacSqErr_100",  # de Bezenac model, trained on err^2 Loss
                             trainset=DL.TrainLoader,  # data loaders: 1. training
                             valset=DL.ValLoader,      #               2. validation
                             testset=DL.TestLoader,    #               3. testing
                             model=net1,               # model
                             regloss=False,            # whether to regularize the training loss
                             test_loss=nn.MSELoss(reduction="mean"),  # loss function for testing
                             optimizer=optim1,
                             outdir=path)
exp1.run(epochs=100)

# ~----- Experiment 2: magnitude regularization = -0.1 for 100 epochs -----~
net2 = BC(device=device, coeffs=[-0.1,0,0], hist=4).to(device)
optim2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
exp2 = Experiment.Experiment(name="BezenacRegSqErr_100",  # de Bezenac model, trained on err^2 Loss
                             trainset=DL.TrainLoader,  # data loaders: 1. training
                             valset=DL.ValLoader,      #               2. validation
                             testset=DL.TestLoader,    #               3. testing
                             model=net1,               # model
                             regloss=True,             # whether to regularize the training loss
                             test_loss=nn.MSELoss(reduction="mean"),  # loss function for testing
                             optimizer=optim2,
                             outdir=path)
exp2.run(epochs=100)

# ~----- Experiment 3: a range of regularization candidates, each for 50 epochs (lr=1e-3) ----~
coeff_candidates = [-0.1, 0, 0.1, 1]

for coeff in coeff_candidates:
    # Magnitude
    net = BC(device, [coeff, 0, 0], hist=4).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    exp = Experiment(name="BezConv_mag=" + str(coeff) + "_50",
                     trainset=DL.TrainLoader,
                     valset=DL.ValLoader,
                     testset=DL.TestLoader,
                     model=net,
                     regloss=True,
                     test_loss=nn.MSELoss(reduction="mean"),
                     optimizer=optim,
                     outdir=path)

    print("Magnitude coefficient value:", coeff)
    exp.run(epochs=50)
    exp.visualize_examples(EL, "exampleImage")
    exp.visualize_examples(DL.ValLoader, "validationImage")
    exp.plot_loss("Performance in Training", "Epoch", ["train", "val"], "lossPlot")

    # Divergence
    net = BC(device, [0, coeff, 0], hist=4).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    exp = Experiment(name="BezConv_div=" + str(coeff) + "_50",
                     trainset=DL.TrainLoader,
                     valset=DL.ValLoader,
                     testset=DL.TestLoader,
                     model=net,
                     regloss=True,
                     test_loss=nn.MSELoss(reduction="mean"),
                     optimizer=optim,
                     outdir=path)

    print("Divergence coefficient value:", coeff)
    exp.run(epochs=50)
    exp.visualize_examples(EL, "exampleImage")
    exp.visualize_examples(DL.ValLoader, "validationImage")
    exp.plot_loss("Performance in Training", "Epoch", ["train", "val"], "lossPlot")

    net = BC(device, [coeff, 0, 0], hist=4).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Smoothness
    exp = Experiment(name="BezConv_smo=" + str(coeff) + "_50",
                     trainset=DL.TrainLoader,
                     valset=DL.ValLoader,
                     testset=DL.TestLoader,
                     model=net,
                     regloss=True,
                     test_loss=nn.MSELoss(reduction="mean"),
                     optimizer=optim,
                     outdir=path)

    print("Smoothness coefficient value:", coeff)
    exp.run(epochs=50)
    exp.visualize_examples(EL, "exampleImage")
    exp.visualize_examples(DL.ValLoader, "validationImage")
    exp.plot_loss("Performance in Training", "Epoch", ["train", "val"], "lossPlot")
