import datetime
import os
import torch
import torch.nn as nn

from src.models import BezConv as BC
from src.dataloader import DataLoader
import utils
import Experiment

device = utils.check_device()

# ------------------------------ Set up directory for experiments -----------------------
try:
    os.mkdir("/projectnb/labci/Lucia/rainfall-pde-ml/experiments/" + str(datetime.date.today()))
except FileExistsError as e:
    print("Already ran experiments today.")
finally:
    path = "/projectnb/labci/Lucia/rainfall-pde-ml/experiments/" + str(datetime.date.today())

# ------------------------------ Data Loaders -----------------------
DL = DataLoader.DataLoader(path, [0.8, 0.1, 0.1], 32, 4)
DL.create_training()
DL.create_validation()
DL.create_test()

net1 = BC(hist=4).to(device)
# net1.to(device)
optim = torch.optim.Adam(net1.parameters(), lr=1e-3)

exp1 = Experiment.Experiment(name="BezenacSqErr_100",  # de Bezenac model, trained on err^2 Loss
                             trainset=DL.TrainLoader,  # data loaders: 1. training
                             valset=DL.ValLoader,      #               2. validation
                             testset=DL.TestLoader,    #               3. testing
                             model=net1,               # model
                             regloss=False,            # whether to regularize the training loss
                             test_loss=nn.MSELoss(reduction="mean"),  # loss function for testing
                             optimizer=optim,
                             outdir=path)

exp1.run(epochs=100)
