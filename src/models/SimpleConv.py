import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce
from src.models.components import _EncoderBlock
from src.models.components import _DecoderBlock
from src.models.components import _CenterBlock


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                # He initialization, from He, K. et al, 2015
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class SimpleConv(nn.Module):
    """Class to implement a simpler Convolution-Deconvolution model to take as input the past k historical weather
    images and output a single future prediction for the same variable.
    """

    def __init__(self, device, hist=1):
        """Function to construct the model.

        Parameters
        ----------
        device : torch.device : device on which to perform computations
        hist : int : the number of days of "history" to use for prediction, default = 1.
        """

        super(SimpleConv, self).__init__()

        self.enc1 = _EncoderBlock(hist, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512)
        self.dec4 = _CenterBlock(512, 386)
        self.dec3 = _DecoderBlock(386 + 256, 256, 194)
        self.dec2 = _DecoderBlock(194 + 128, 128, 98)
        self.dec1 = _DecoderBlock(98 + 64, 64, 1)

        self.final = nn.Sequential(nn.Conv2d(2, 2, kernel_size=3), )
        initialize_weights(self)

        self.device = device
        self.hist = hist

    def forward(self, x):
        # TODO: test
        """Function to execute the forward pass of the model.

        Parameters
        ----------
        x : torch.FloatTensor : the historical input images

        Returns
        -------
        pred : torch.FloatTensor : the predicted next image
        """

        enc1 = self.enc1(x)
        # print("enc1",enc1.shape)
        enc2 = self.enc2(enc1)
        # print("enc2",enc2.shape)
        enc3 = self.enc3(enc2)
        # print("enc3",enc3.shape)
        enc4 = self.enc4(enc3)
        # print("enc4",enc4.shape)
        dec4 = self.dec4(enc4)
        # print("dec4",dec4.shape)

        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        print("final", final.shape)

        pred = F.interpolate(final, x.size()[2:], mode='bilinear')
        print("pred", pred.shape)

        return pred
