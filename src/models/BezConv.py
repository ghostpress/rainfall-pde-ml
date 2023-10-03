import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce


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


# TODO: move these classes and the method above to a separate file, eg. components.py

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=1)

    def forward(self, x):
        residual = x
        # print("residual",residual.shape)
        out = self.cv1(x)
        # print("cv1",out.shape)
        out = self.bn1(out)
        # print("bn1",out.shape)
        out = self.lr1(out)
        # print("lr1",out.shape)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.maxp(out)
        return out


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, middle_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.tcv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        # print("residual",residual.shape)
        out = self.cv1(x)
        # print("cv1",out.shape)
        out = self.bn1(out)
        # print("bn1",out.shape)
        out = self.lr1(out)
        # print("lr1",out.shape)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.tcv(out)
        return out


class _CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_CenterBlock, self).__init__()
        self.cv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.lr1 = nn.LeakyReLU(0.1)
        self.cv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lr2 = nn.LeakyReLU(0.1)
        self.tcv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3)

    def forward(self, x):
        residual = x
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.lr1(out)
        out += residual
        out = self.cv2(out)
        out = self.bn2(out)
        out = self.lr2(out)
        out = self.tcv(out)
        return out


class BezConv(nn.Module):
    """Class to implement a physics-driven Convolution-Deconvolution Neural Network (CDNN) as described in
    (de Bezenac et. al., 2019). This model takes as input historical 2D image(s) of a weather variable, X, and uses a
    convolutional neural network (CNN) to estimate the wind vector field W that drives the motion of X. From there, the
    next image is predicted using a "warping" of the most recent input image and W, as if to see how the variable
    evolves with the wind. The "warping" in this paper is a radial basis function kernel, or a Gaussian centered in X-W.
    """

    def __init__(self, device, hist=1):
        """Function to construct the model.

        Parameters
        ----------
        device : torch.device : device on which to perform computations
        hist : int : the number of days of "history" to use for prediction, default = 1.
        """

        super(BezConv, self).__init__()

        self.enc1 = _EncoderBlock(hist, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512)
        self.dec4 = _CenterBlock(512, 386)
        self.dec3 = _DecoderBlock(386 + 256, 256, 194)
        self.dec2 = _DecoderBlock(194 + 128, 128, 98)
        self.dec1 = _DecoderBlock(98 + 64, 64, 2)

        self.final = nn.Sequential(nn.Conv2d(2, 2, kernel_size=3), )
        initialize_weights(self)

        self.hist = hist
        self.device = device

    def wind(self, x):
        """Function to estimate the wind vector field from historical input images.

        Parameters
        ----------
        x : torch.FloatTensor : the historical input images

        Returns
        -------
        wind : torch.FloatTensor : the estimated wind vector field
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

        wind = F.interpolate(final, x.size()[2:], mode='bilinear')

        return wind

    @staticmethod
    def kernel(distsq, D, dt):
        """Method to implement the k() function or radial basis function kernel described in (de Bezenac et. al., 2019).

        Parameters
        ----------
        distsq : float or torch.FloatTensor : the value of the squared norm of the distance
        D : float : the diffusion coefficient
        dt : float : the timestep

        Returns
        -------
        res : float or torch.FloatTensor: the result of the function
        """

        res = torch.exp(-distsq / (4 * D * dt)) / (4 * np.pi * D * dt)
        return res

    def warp(self, image, W, hist):
        """Function to compute the warping of the input data and an estimated wind vector field, in order to produce an
        output predicted image.

        Parameters
        ----------
        image : torch.FloatTensor : the most recent input image to warp
        W : torch.FloatTensor : the estimated wind vector field
        hist : int : how many prior images to use for estimation

        Returns
        -------
        warped : torch.FloatTensor : the warped prediction image
        """

        D = 0.45
        dt = 1

        interval = torch.arange(image.size()[-1]).type(torch.FloatTensor)

        x1 = interval[None, :, None, None, None]  # .to(self.device)
        x2 = interval[None, None, :, None, None]  # .to(self.device)
        y1 = interval[None, None, None, :, None]  # .to(device)
        y2 = interval[None, None, None, None, :]  # .to(device)

        # x - wind - y
        distsq = (x1 - y1 - W[:, 0, :, :, None, None]) ** 2 + (x2 - y2 - W[:, 1, :, :, None, None]) ** 2
        mult = image[:, hist - 1, None, None, :, :] * self.kernel(distsq, D, dt)
        warped = mult.sum(4).sum(3)

        return warped

    def forward(self, x):
        """Function to execute the forward pass of the model. All the computations are done in the methods above,
        so this function simply returns their outputs.

        Note: the wind vector field W is returned for use in the regularized loss function later. For simple difference
        loss functions, returning y_pred is sufficient.

        Parameters
        ----------
        x : torch.FloatTensor : the historical input images

        Returns
        -------
        W : torch.FloatTensor : the estimated wind vector field
        y_pred : torch.FloatTensor : the predicted next image
        """

        W = self.wind(x)
        y_pred = self.warp(x, W, self.hist)

        return W, y_pred

    @staticmethod
    def compute_regloss(f, dtype=torch.FloatTensor):
        gradient = reduce(torch.add, torch.gradient(f))

        magnitude = torch.Tensor([torch.mean(torch.linalg.norm(f, axis=0, ord=2) ** 2)]).to(BezConv.device)
        divergence = torch.Tensor([torch.mean(gradient.sum(0)**2)]).to(BezConv.device)
        smoothness = torch.Tensor([torch.mean(torch.linalg.norm(gradient, axis=0, ord=2)**2)]).to(BezConv.device)

        # TODO: check dimensions of divergence and smoothness
        return magnitude.type(dtype)[0], divergence.type(dtype), smoothness.type(dtype)

    @staticmethod
    def loss(y_pred, y, reg_coeffs, w=None, reg=False):
        assert(len(reg_coeffs) == 3)
        error = torch.sum((y_pred - y)**2, axis=1)
        loss = torch.mean(error)

        if reg:
            for c in reg_coeffs:
                reg_term = BezConv.compute_regloss(w[0, :, :, :])
                loss += c * reg_term

        return loss
