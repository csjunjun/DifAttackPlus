import torch
import torch.nn as nn
import numpy as np
from scipy import stats as st
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.nn.functional as F
import torchvision.transforms as transforms
class TIMIFGSM(nn.Module):
    r"""
    TIFGSM in the paper 'Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks'
    [https://arxiv.org/abs/1904.02884]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=0.0,
        kernel_name="gaussian",
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=False,
        targeted = False,
        change=True,
        returnGrad=False
    ):
        super(TIMIFGSM,self).__init__()
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.returnGrad = returnGrad
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.targeted = targeted
        self.model = model
        self.change = change

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

       

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().cuda()
        stacked_kernel = self.stacked_kernel.cuda()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            

            # Calculate loss
            # if self.targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)

            cost = 0
            if not self.change:
                adv_input = self.input_diversity(adv_images)
            for modeli in range(len(self.model)):
                if not self.change:
                    outputs = self.model[modeli](adv_input)
                else:
                    outputs = self.model[modeli](self.input_diversity(adv_images))
                
                if self.targeted:
                    cost -= loss(outputs, labels)
                else:
                    cost += loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            # depth wise conv2d
            grad = torch.nn.functional.conv2d(grad, stacked_kernel, stride=1, padding="same", groups=3)
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if self.returnGrad:
                return grad
        return adv_images

    def kernel_generation(self):
        if self.kernel_name == "gaussian":
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == "linear":
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == "uniform":
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(
            np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
            / (kernlen + 1)
            * 2
        )
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = torch.nn.functional.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x




class VNIFGSM(nn.Module):
    r"""
    VNI-FGSM in the paper 'Enhancing the Transferability of Adversarial Attacks through Variance Tuning
    [https://arxiv.org/abs/2103.15571], Published as a conference paper at CVPR 2021
    Modified from "https://github.com/JHL-HUST/VT"
    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=1.0,
        kernel_name="gaussian",
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
        random_start=False,
        targeted = False,
        change=True,
        N=5,
        beta=3/2,
        tidi=False,
        returnGrad=False
    ):
        super(VNIFGSM,self).__init__()
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())
        self.targeted = targeted
        self.model = model
        self.change = change
        self.N = N
        self.beta = beta
        self.tidi = tidi
        self.returnGrad = returnGrad

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

       

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().cuda()
        v = torch.zeros_like(images).detach().cuda()
        if self.tidi:
            stacked_kernel = self.stacked_kernel.cuda()

        adv_images = images.clone().detach()

        # if self.random_start:
        #     # Starting at a uniformly random point
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(
        #         -self.eps, self.eps
        #     )
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        
        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum

        

            cost = 0
            # if not self.change:
            #     adv_input = self.input_diversity(adv_images)
            for modeli in range(len(self.model)):
                #if not self.change:
                
                # else:
                if self.tidi:
                    outputs = self.model[modeli](self.input_diversity(nes_images))
                else:
                    outputs = self.model[modeli](nes_images)
                
                if self.targeted:
                    cost -= loss(outputs, labels)
                else:
                    cost += loss(outputs, labels)

            adv_grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            if self.tidi:
                adv_grad = torch.nn.functional.conv2d(adv_grad, stacked_kernel, stride=1, padding="same", groups=3)

            grad = (adv_grad + v) / torch.mean(
                torch.abs(adv_grad + v), dim=(1, 2, 3), keepdim=True
            )
            grad = grad + momentum * self.decay
            momentum = grad
            
            # Calculate Gradient Variance
            GV_grad = torch.zeros_like(images).detach().cuda()
            for _ in range(self.N):
                neighbor_images = adv_images.detach() + torch.randn_like(
                    images
                ).uniform_(-self.eps * self.beta, self.eps * self.beta)
                neighbor_images.requires_grad = True
                cost = 0
                for modeli in range(len(self.model)):
                    #if not self.change:
                    outputs = self.model[modeli](neighbor_images)
                    # else:
                    #     outputs = self.model[modeli](self.input_diversity(adv_images))
                    
                    if self.targeted:
                        cost -= loss(outputs, labels)
                    else:
                        cost += loss(outputs, labels)
                GV_grad += torch.autograd.grad(
                    cost, neighbor_images, retain_graph=False, create_graph=False
                )[0]
            # obtaining the gradient variance
            v = GV_grad / self.N - adv_grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if self.returnGrad:
                return grad
        return adv_images

    def kernel_generation(self):
        if self.kernel_name == "gaussian":
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == "linear":
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == "uniform":
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(
            np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
            / (kernlen + 1)
            * 2
        )
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = torch.nn.functional.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x


class SINIFGSM(nn.Module):
    r"""
     SI-NI-FGSM in the paper 'NESTEROV ACCELERATED GRADIENT AND SCALEINVARIANCE FOR ADVERSARIAL ATTACKS'
    [https://arxiv.org/abs/1908.06281], Published as a conference paper at ICLR 2020
    Modified from "https://githuba.com/JHL-HUST/SI-NI-FGSM"

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 0.0)
        kernel_name (str): kernel name. (Default: gaussian)
        len_kernel (int): kernel length.  (Default: 15, which is the best according to the paper)
        nsig (int): radius of gaussian kernel. (Default: 3; see Section 3.2.2 in the paper for explanation)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TIFGSM(model, eps=8/255, alpha=2/255, steps=10, decay=1.0, resize_rate=0.9, diversity_prob=0.7, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        decay=1.0,
        kernel_name="gaussian",
        len_kernel=15,
        nsig=3,
        resize_rate=0.9,
        diversity_prob=0.5,
        m=5,
        targeted = False,
        tidi=False,
        returnGrad=False
    ):
        super(SINIFGSM,self).__init__()
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.m=m
        self.targeted = targeted
        self.model = model
        self.tidi = tidi
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel

        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.nsig = nsig
        self.returnGrad = returnGrad
        self.stacked_kernel = torch.from_numpy(self.kernel_generation())


    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

       

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().cuda()
        stacked_kernel = self.stacked_kernel.cuda()

        adv_images = images.clone().detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay * self.alpha * momentum            
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().cuda()
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                cost=0
                for modeli in range(len(self.model)):
                    
                    outputs = self.model[modeli](self.input_diversity(nes_images))
                    
                    
                    if self.targeted:
                        cost -= loss(outputs, labels)
                    else:
                        cost += loss(outputs, labels)
                        
                
                adv_grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]            
            adv_grad = adv_grad / self.m
            

            # Update adversarial images
            #grad = torch.nn.functional.conv2d(adv_grad, stacked_kernel, stride=1, padding="same", groups=3)#这一行是sinitifgsm的错误代码，named as sinidifgsm
            adv_grad = torch.nn.functional.conv2d(adv_grad, stacked_kernel, stride=1, padding="same", groups=3)
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            if self.returnGrad==True:
                return grad
        return adv_images

    def diy(self, cleanimages,images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

       

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().cuda()
        stacked_kernel = self.stacked_kernel.cuda()

        adv_images = images.clone().detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_image = adv_images + self.decay * self.alpha * momentum            
            # Calculate sum the gradients over the scale copies of the input image
            adv_grad = torch.zeros_like(images).detach().cuda()
            for i in torch.arange(self.m):
                nes_images = nes_image / torch.pow(2, i)
                cost=0
                for modeli in range(len(self.model)):
                    
                    outputs = self.model[modeli](self.input_diversity(nes_images))
                    
                    
                    if self.targeted:
                        cost -= loss(outputs, labels)
                    else:
                        cost += loss(outputs, labels)
                        
                
                adv_grad += torch.autograd.grad(
                    cost, adv_images, retain_graph=False, create_graph=False
                )[0]            
            adv_grad = adv_grad / self.m
            

            # Update adversarial images
            #grad = torch.nn.functional.conv2d(adv_grad, stacked_kernel, stride=1, padding="same", groups=3)
            adv_grad = torch.nn.functional.conv2d(adv_grad, stacked_kernel, stride=1, padding="same", groups=3)
            grad = self.decay * momentum + adv_grad / torch.mean(
                torch.abs(adv_grad), dim=(1, 2, 3), keepdim=True
            )
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - cleanimages, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(cleanimages + delta, min=0, max=1).detach()
            if self.returnGrad==True:
                return grad
        # if self.returnGrad==True:
        #     #return grad
        #     return adv_images-cleanimages
        return adv_images

    def kernel_generation(self):
        if self.kernel_name == "gaussian":
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == "linear":
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == "uniform":
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen, kernlen)) * 1.0 / (kernlen * kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1 - np.abs(
            np.linspace((-kernlen + 1) / 2, (kernlen - 1) / 2, kernlen)
            / (kernlen + 1)
            * 2
        )
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = torch.nn.functional.interpolate(
            x, size=[rnd, rnd], mode="bilinear", align_corners=False
        )
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

#https://github.com/ZhengyuZhao/Targeted-Transfer/blob/main/eval_ensemble.py
class TTA(nn.Module):
    def __init__(self,models,lr,max_iterations,epsilon,return_grad,useDI, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_iterations = max_iterations
        self.models = models
        self.epsilon = epsilon
        self.channels=3
        self.kernel_size=5
        self.kernel = self.gkern(self.kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([self.kernel, self.kernel, self.kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        self.gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        self.lr = lr
        self.return_grad = return_grad
        self.imgsize = 224
        self.useDI = useDI

    def DI(self,X_in):
        rnd = np.random.randint(self.imgsize, self.imgsize+30,size=1)[0]
        h_rem = self.imgsize+30 - rnd
        w_rem = self.imgsize+30 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c[0] <= 0.7:
            X_out = F.pad(torch.nn.functional.interpolate(X_in, size=(int(rnd),int(rnd))),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
            return  X_out 
        else:
            return  X_in  

    def gkern(self,kernlen=15, nsig=3):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

  
    def forward(self,image,label,target_label,models=None):
        if models is not None:
            self.models = models
        delta = torch.zeros_like(image,requires_grad=True).cuda()
        grad_pre = 0
        for t in range(self.max_iterations):
            if self.useDI:
                x_aug = self.DI(image+delta)
            else:
                x_aug = image+delta
            logits = 0 
            for modeli in self.models:
                logits += modeli(x_aug)
            loss = -logits.gather(1, target_label.unsqueeze(1)).squeeze(1)
            loss = loss.sum()
            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3) 
            grad_a = grad_c + 1 * grad_pre
            grad_pre = grad_a            
            delta.grad.zero_()
            delta.data = delta.data - self.lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon / 255,self.epsilon / 255) 
            delta.data = ((image + delta.data).clamp(0,1)) - image 
        if self.return_grad :
            return delta.detach() 
        return image+delta.data



#https://github.com/SignedQiu/MEFAttack/tree/main
class MEF(nn.Module):
    def __init__(self, model,sample_num=20,iteration_num=20,eps=0.05,gamma=2,\
                 kesai=0.15,inner_mu=0.9,outer_mu=0.5,returnGrad = False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_num = sample_num 
        self.iteration_num = iteration_num
        self.epsilon = eps
        self.step_size = self.epsilon / self.iteration_num
        self.gamma = gamma
        self.inner_mu = inner_mu
        self.outer_mu = outer_mu
        self.kesai = kesai
        self.source_model = model
        self.returnGrad = returnGrad    

    def forward(self, images, labels): 
        img = images.detach().clone()
        grad_pre = torch.zeros_like(images)
        grad_t = torch.zeros_like(images)

        b, c, h, w = images.shape

        grad_list = torch.zeros([self.sample_num, b, c, h, w]).cuda()
        grad_pgia = torch.zeros([self.sample_num, b, c, h, w]).cuda()

        for j in range(self.iteration_num):

            img_x = img.clone().detach()
            for k in range(self.sample_num):

                img_near = img_x + torch.rand_like(img_x).uniform_(-self.gamma*self.epsilon, self.gamma*self.epsilon)
                img_min = img_near + self.kesai*self.epsilon*(grad_pgia[k])
                img_min.requires_grad_(True)
                loss_total=0
                for modeli in range(len(self.source_model)):
                    logits = self.source_model[modeli](img_min)
                    loss = nn.CrossEntropyLoss(reduction='mean')(logits,labels)
                    if modeli==0:
                        loss_total=loss
                    else:
                        loss_total+=loss
                loss_total.backward()
                grad_list[k] = img_min.grad.detach().clone()
                img_min.grad.zero_()

            grad = (1/self.sample_num)*grad_list
            grad_pgia = ((grad / torch.mean(torch.abs(grad), (2, 3, 4), keepdim=True)) - self.inner_mu * grad_pgia)
            grad_t = grad.sum(0)
            grad_t = grad_t / torch.mean(torch.abs(grad_t), (1, 2, 3), keepdim=True)
            input_grad = grad_t + self.outer_mu * grad_pre
            grad_pre = input_grad
            input_grad = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3), keepdim=True)
            img = img.data + self.step_size * torch.sign(input_grad)

            img = torch.where(img > images + self.epsilon, images + self.epsilon, img)
            img = torch.where(img < images - self.epsilon, images - self.epsilon, img)
            img = torch.clamp(img, min=0, max=1)
        if self.returnGrad:
            return input_grad
        return img
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
