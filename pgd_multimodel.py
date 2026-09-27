import torch
import torch.nn as nn
import numpy as np
from scipy import stats as st
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.transforms as transforms




class PGN(nn.Module):
    #https://github.com/Trustworthy-AI-Group/PGN/blob/main/Incv3_PGN_Attack.py NeurIPs 2023
    def __init__(self,model, eps=0.05,
                  steps=10,momentum = 1.0,zeta=3.0,delta=0.5,N=20, returnGrad=False,targeted=False):
        super(PGN,self).__init__()
        self.model = model
        self.eps = eps
        self.num_iter = steps
        self.alpha = self.eps / self.num_iter
        self.momentum =momentum
        self.zeta = zeta
        self.delta = delta
        self.N = N
        self.returnGrad = returnGrad
        self.targeted = targeted
        
    def forward(self,images,labels):
        lower = torch.clamp(images-self.eps,0,1).cuda()
        upper = torch.clamp(images+self.eps,0,1).cuda()
        
        x = images.clone().detach().cuda()
        grad = torch.zeros_like(x).detach().cuda()
        for i in range(self.num_iter):
            avg_grad = torch.zeros_like(x).detach().cuda()
            for _ in range(self.N):
                x_near = x + torch.rand_like(x).uniform_(-self.eps*self.zeta, self.eps*self.zeta)
                x_near = V(x_near, requires_grad = True)
                loss = 0
                for modeli in range(len(self.model)):
                    output_v3 = self.model[modeli](x_near)
                    if self.targeted:
                        loss -= F.cross_entropy(output_v3, labels)
                    else:
                        loss += F.cross_entropy(output_v3, labels)
                    
                g1 = torch.autograd.grad(loss, x_near,
                                            retain_graph=False, create_graph=False)[0]
                x_star = x_near.detach() + self.alpha * (-g1)/torch.abs(g1).mean([1, 2, 3], keepdim=True)

                nes_x = x_star.detach()
                nes_x = V(nes_x, requires_grad = True)
                loss= 0
                for modeli in range(len(self.model)):

                    output_v3 = self.model[modeli](nes_x)
                    if self.targeted:
                        loss -= F.cross_entropy(output_v3, labels)
                    else:
                        loss += F.cross_entropy(output_v3, labels)
                g2 = torch.autograd.grad(loss, nes_x,
                                            retain_graph=False, create_graph=False)[0]

                avg_grad += (1-self.delta)*g1 + self.delta*g2
            noise = (avg_grad) / torch.abs(avg_grad).mean([1, 2, 3], keepdim=True)
            noise = self.momentum * grad + noise
            grad = noise
            
            x = x + self.alpha * torch.sign(noise)
            x = self.clip_by_tensor(x, lower, upper)
        if self.returnGrad:
            return grad
        return x.detach()
    def clip_by_tensor(self,t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
class PGD(nn.Module):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, modelList, eps=0.3,
                 alpha=2/255, steps=40, random_start=True,targeted=False,returnGrad=False):
        super(PGD,self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.modelList = modelList
        self.targeted = targeted
        self.returnGrad= returnGrad

    def forward(self, images, labels,modelList = None):
        r"""
        Overridden.
        """
        if modelList is not None:
            self.modelList = modelList
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()


        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            cost = 0
            for modeli in range(len(self.modelList)):
                outputs = self.modelList[modeli](adv_images)
                if self.targeted:
                    cost -= loss(outputs, labels)
                else:
                    cost += loss(outputs, labels)

            # # Calculate loss
            # if self._targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if self.returnGrad==True:
                return grad
        return adv_images
