
import os
from WhiteBoxAttack import PGD as PGDMultiModel
from WhiteBoxAttack import TIMIFGSM as timMultiModel
from WhiteBoxAttack import SINIFGSM as siniMultiModel
from WhiteBoxAttack import VNIFGSM as vniMultiModel
from WhiteBoxAttack import PGN
from config import refs,popdict

from PIL import Image
import gc
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import cv2
import torch.nn.functional as F

from torchvision import transforms

from torchvision.datasets import ImageFolder

from advertorch.attacks import LinfPGDAttack,CarliniWagnerL2Attack,LinfMomentumIterativeAttack
from advertorch.context import ctx_noparamgrad_and_eval
import utils
import random


class MyRobustModel(nn.Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model
    def forward(self,x):
        return self.model(x)[0]
    
def setSeed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)  
   
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False       
    torch.backends.cudnn.deterministic = True

#Model & Functions

#image values are clamped in range of 0 & 1 to get rid of negative values 
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 224, 224)
    return x




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        #Encoder
        self.conv_1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)

        self.conv_2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)

        self.conv_3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

       

        self.conv_5 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(512)

        self.conv_6 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(512)

        #Decoder
        self.deconv_0 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=0)
        self.batchNorm0_d = nn.BatchNorm2d(256)

        self.deconv_1 = nn.ConvTranspose2d(512+256, 256, 5, stride=4, padding=1, output_padding=1)
        self.batchNorm1_d = nn.BatchNorm2d(256)

        
        self.deconv_2 = nn.ConvTranspose2d(256+128, 128, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm2_d = nn.BatchNorm2d(128)
        
        self.deconv_3 = nn.ConvTranspose2d(128+64, 64, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm3_d = nn.BatchNorm2d(64)
        
        self.deconv_4 = nn.ConvTranspose2d(64+32, 32, 3, stride=2, padding=1, output_padding=1)
        self.batchNorm4_d = nn.BatchNorm2d(32)
        
        self.deconv_5 = nn.ConvTranspose2d(32, 3, 3,  padding=1)

        self.weight0_vis = nn.Conv2d(512,512,1,1)
        self.weight1_vis = nn.Conv2d(512,512,1,1)
        self.weight2_vis = nn.Conv2d(128,128,1,1)
        self.weight3_vis = nn.Conv2d(64,64,1,1)
        self.weight4_vis = nn.Conv2d(32,32,1,1)

        self.weight0_sem = nn.Conv2d(512,512,1,1)
        self.weight1_sem = nn.Conv2d(512,512,1,1)
        self.weight2_sem = nn.Conv2d(128,128,1,1)
        self.weight3_sem = nn.Conv2d(64,64,1,1)
        self.weight4_sem = nn.Conv2d(32,32,1,1)

        self.weight0_vis_vis = nn.Conv2d(512,512,1,1)
        self.weight1_vis_vis = nn.Conv2d(512,512,1,1)
        self.weight2_vis_vis = nn.Conv2d(128,128,1,1)
        self.weight3_vis_vis = nn.Conv2d(64,64,1,1)
        self.weight4_vis_vis = nn.Conv2d(32,32,1,1)

        self.weight0_vis_adv = nn.Conv2d(512,512,1,1)
        self.weight1_vis_adv = nn.Conv2d(512,512,1,1)
        self.weight2_vis_adv = nn.Conv2d(128,128,1,1)
        self.weight3_vis_adv = nn.Conv2d(64,64,1,1)
        self.weight4_vis_adv = nn.Conv2d(32,32,1,1)

        self.weight0_sem_vis = nn.Conv2d(512,512,1,1)
        self.weight1_sem_vis = nn.Conv2d(512,512,1,1)
        self.weight2_sem_vis = nn.Conv2d(128,128,1,1)
        self.weight3_sem_vis = nn.Conv2d(64,64,1,1)
        self.weight4_sem_vis = nn.Conv2d(32,32,1,1)

        self.weight0_sem_adv = nn.Conv2d(512,512,1,1)
        self.weight1_sem_adv = nn.Conv2d(512,512,1,1)
        self.weight2_sem_adv = nn.Conv2d(128,128,1,1)
        self.weight3_sem_adv = nn.Conv2d(64,64,1,1)
        self.weight4_sem_adv = nn.Conv2d(32,32,1,1)


        self.combine0_adv = nn.Conv2d(512*2,512,1,1)
        self.combine1_adv = nn.Conv2d(512*2,512,1,1)
        self.combine2_adv = nn.Conv2d(128*2,128,1,1)
        self.combine3_adv = nn.Conv2d(64*2,64,1,1)
        self.combine4_adv = nn.Conv2d(32*2,32,1,1)

        self.combine0_vis = nn.Conv2d(512*2,512,1,1)
        self.combine1_vis = nn.Conv2d(512*2,512,1,1)
        self.combine2_vis = nn.Conv2d(128*2,128,1,1)
        self.combine3_vis = nn.Conv2d(64*2,64,1,1)
        self.combine4_vis = nn.Conv2d(32*2,32,1,1)

        self.combine0 = nn.Conv2d(512*2,512,1,1)
        self.combine1 = nn.Conv2d(512*2,512,1,1)
        self.combine2 = nn.Conv2d(128*2,128,1,1)
        self.combine3 = nn.Conv2d(64*2,64,1,1)
        self.combine4 = nn.Conv2d(32*2,32,1,1)



    def forward(self, x):
       # Encoder
        conv_b1 = F.relu(self.batchNorm1(self.conv_1(x)))
        conv_b2 = F.relu(self.batchNorm2(self.conv_2(conv_b1)))
        conv_b3 = F.relu(self.batchNorm3(self.conv_3(conv_b2)))
        conv_b4 = F.relu(self.batchNorm4(self.conv_4(conv_b3)))
        conv_b5 = F.relu(self.batchNorm5(self.conv_5(conv_b4)))
        conv_b6 = F.relu(self.batchNorm6(self.conv_6(conv_b5)))

        #Decoupling 
        conv_b6_vis = self.weight0_vis(conv_b6) #512,4,4
        conv_b5_vis = self.weight1_vis(conv_b5) #512,7,7
        conv_b3_vis = self.weight2_vis(conv_b3) #128,28,28
        conv_b2_vis = self.weight3_vis(conv_b2) #64,56,56
        conv_b1_vis = self.weight4_vis(conv_b1) #32,112,112

        conv_b6_sem = self.weight0_sem(conv_b6)
        conv_b5_sem = self.weight1_sem(conv_b5)
        conv_b3_sem = self.weight2_sem(conv_b3)
        conv_b2_sem = self.weight3_sem(conv_b2)
        conv_b1_sem = self.weight4_sem(conv_b1)

        conv_b6_vis_vis = self.weight0_vis_vis(conv_b6_vis)
        conv_b5_vis_vis = self.weight1_vis_vis(conv_b5_vis)
        conv_b3_vis_vis = self.weight2_vis_vis(conv_b3_vis)
        conv_b2_vis_vis = self.weight3_vis_vis(conv_b2_vis)
        conv_b1_vis_vis = self.weight4_vis_vis(conv_b1_vis)

        conv_b6_vis_adv = self.weight0_vis_adv(conv_b6_vis)
        conv_b5_vis_adv = self.weight1_vis_adv(conv_b5_vis)
        conv_b3_vis_adv = self.weight2_vis_adv(conv_b3_vis)
        conv_b2_vis_adv = self.weight3_vis_adv(conv_b2_vis)
        conv_b1_vis_adv = self.weight4_vis_adv(conv_b1_vis)

        conv_b6_sem_vis = self.weight0_sem_vis(conv_b6_sem)
        conv_b5_sem_vis = self.weight1_sem_vis(conv_b5_sem)
        conv_b3_sem_vis = self.weight2_sem_vis(conv_b3_sem)
        conv_b2_sem_vis = self.weight3_sem_vis(conv_b2_sem)
        conv_b1_sem_vis = self.weight4_sem_vis(conv_b1_sem)

        conv_b6_sem_adv = self.weight0_sem_adv(conv_b6_sem)
        conv_b5_sem_adv = self.weight1_sem_adv(conv_b5_sem)
        conv_b3_sem_adv = self.weight2_sem_adv(conv_b3_sem)
        conv_b2_sem_adv = self.weight3_sem_adv(conv_b2_sem)
        conv_b1_sem_adv = self.weight4_sem_adv(conv_b1_sem)



       #Combine
        
        conv_b6_vis = self.combine0_vis(torch.cat((conv_b6_vis_vis,conv_b6_sem_vis),dim=1))
        conv_b5_vis = self.combine1_vis(torch.cat((conv_b5_vis_vis,conv_b5_sem_vis),dim=1))
        conv_b3_vis = self.combine2_vis(torch.cat((conv_b3_vis_vis,conv_b3_sem_vis),dim=1))
        conv_b2_vis = self.combine3_vis(torch.cat((conv_b2_vis_vis,conv_b2_sem_vis),dim=1))
        conv_b1_vis = self.combine4_vis(torch.cat((conv_b1_vis_vis,conv_b1_sem_vis),dim=1))

        conv_b6_sem = self.combine0_adv(torch.cat((conv_b6_vis_adv,conv_b6_sem_adv),dim=1))
        conv_b5_sem = self.combine1_adv(torch.cat((conv_b5_vis_adv,conv_b5_sem_adv),dim=1))
        conv_b3_sem = self.combine2_adv(torch.cat((conv_b3_vis_adv,conv_b3_sem_adv),dim=1))
        conv_b2_sem = self.combine3_adv(torch.cat((conv_b2_vis_adv,conv_b2_sem_adv),dim=1))
        conv_b1_sem = self.combine4_adv(torch.cat((conv_b1_vis_adv,conv_b1_sem_adv),dim=1))

        conv_b6 =  self.combine0(torch.cat((conv_b6_vis,conv_b6_sem),dim=1))
        conv_b5 =  self.combine1(torch.cat((conv_b5_vis,conv_b5_sem),dim=1))
        conv_b3 =  self.combine2(torch.cat((conv_b3_vis,conv_b3_sem),dim=1))
        conv_b2 =  self.combine3(torch.cat((conv_b2_vis,conv_b2_sem),dim=1))
        conv_b1 =  self.combine4(torch.cat((conv_b1_vis,conv_b1_sem),dim=1))

        #Decode
        deconv_b0 = F.relu(self.batchNorm0_d(self.deconv_0(conv_b6)))
        concat_0 = torch.cat((deconv_b0, conv_b5),1)

        deconv_b1 = F.relu(self.batchNorm1_d(self.deconv_1(concat_0)))
        concat_1 = torch.cat((deconv_b1, conv_b3),1)

        deconv_b2 = F.relu(self.batchNorm2_d(self.deconv_2(concat_1)))
        concat_2 = torch.cat((deconv_b2, conv_b2),1)

        deconv_b3 = F.relu(self.batchNorm3_d(self.deconv_3(concat_2)))
        concat_3 = torch.cat((deconv_b3, conv_b1),1)

        deconv_b4 = F.relu(self.batchNorm4_d(self.deconv_4(concat_3)))

        deconv_b5 = F.tanh(  self.deconv_5(deconv_b4))



        return deconv_b5,conv_b6_vis,conv_b5_vis,conv_b3_vis,conv_b2_vis,conv_b1_vis,conv_b6_sem,conv_b5_sem,conv_b3_sem,conv_b2_sem,conv_b1_sem\
        

    def decode(self,conv_b6_vis,conv_b5_vis,conv_b3_vis,conv_b2_vis,conv_b1_vis,conv_b6_sem,conv_b5_sem,conv_b3_sem,conv_b2_sem,conv_b1_sem):
       z0 =  self.combine0(torch.cat((conv_b6_vis,conv_b6_sem),dim=1))
       z =  self.combine1(torch.cat((conv_b5_vis,conv_b5_sem),dim=1))
       z2 =  self.combine2(torch.cat((conv_b3_vis,conv_b3_sem),dim=1))
       z3 =  self.combine3(torch.cat((conv_b2_vis,conv_b2_sem),dim=1))
       z4 =  self.combine4(torch.cat((conv_b1_vis,conv_b1_sem),dim=1))

       deconv_b0 = F.relu(self.batchNorm0_d(self.deconv_0(z0)))
       concat_0 = torch.cat((deconv_b0, z),1)

       deconv_b1 = F.relu(self.batchNorm1_d(self.deconv_1(concat_0)))
       concat_1 = torch.cat((deconv_b1, z2),1)
       
       deconv_b2 = F.relu(self.batchNorm2_d(self.deconv_2(concat_1)))
       concat_2 = torch.cat((deconv_b2, z3),1)

       deconv_b3 = F.relu(self.batchNorm3_d(self.deconv_3(concat_2)))
       concat_3 = torch.cat((deconv_b3, z4),1)

       deconv_b4 = F.relu(self.batchNorm4_d(self.deconv_4(concat_3)))

       deconv_b5 = F.tanh(  self.deconv_5(deconv_b4))   
       return deconv_b5     
    

def mysortkey(filename:str):
    return int(filename.split("_")[0]) 


def testSensitivity(modelpathp):
    model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
    model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
    #pretrainedmodelPath='/data/liujun/junliu/weights/deCouplingAttack/PretrainModels/AE_model_nochange_noloss2_40.pth'
    print(modelpathp)
    model_clean.eval()
    model_adv.eval()  

    imgbase = '/data/liujun/junliu/ImageNet_val_mini/randomCropped224'      
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)
    from PIL import Image
    trans = transforms.Compose([
            transforms.ToTensor()
            ])
    linf_con = 0.05 
    target_label = -1
    logits_decrease_list=[]
    #for imgpath,labels in testfiles:
    imgtotal = 0
    for filename in imagelist:
        imgpath = "{}/{}".format(imgbase,filename)
        label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

        batch = Image.open(imgpath)
        batch = batch.convert("RGB")
        batch = trans(batch).unsqueeze(0)
        
        
        lower = torch.clamp(batch-linf_con,0,1).cuda()
        upper = torch.clamp(batch+linf_con,0,1).cuda()
        batch = (batch-0.5)/0.5
        
        #batch, label = data       # Get a batch,-1,1
        batch = batch.cuda()
        label = label.cuda()
        with torch.no_grad():
            logits_clean = net(batch*0.5+0.5)
        pre=torch.argmax(logits_clean,dim=1)
        if target_label>=0 and pre==target_label:
            continue
        elif pre != label:
            continue
        # ===================forward=====================
        with torch.no_grad():
            output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_clean(batch) 
        print("rec l2:{}".format(float(torch.norm(output*0.5-batch*0.5))))    

        batch_perturb = batch*0.5+0.5+0.1*torch.randn_like(batch).cuda()
        batch_perturb = torch.clamp(batch_perturb,lower,upper)
        batch_perturb = (batch_perturb-0.5)/0.5

        with torch.no_grad():
            output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,\
                z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem\
                    = model_adv(batch_perturb) 

        
        with torch.no_grad():
            output_inter1 = model_adv.decode(z_vis0,z_vis,\
                                            z2_vis,z3_vis,\
                                            z4_vis,z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem)
        
        with torch.no_grad():
            logits_perturb = net(output_inter1*0.5+0.5)
        
        logits_decrease = nn.functional.softmax(logits_clean,dim=1)[0][label]-nn.functional.softmax(logits_perturb,dim=1)[0][label]
        logits_decrease_list.append(float(logits_decrease))
        imgtotal +=1
        if imgtotal==10:
            break
    print("avg logits_decrease_list:{}".format(np.mean(np.asarray(logits_decrease_list))))
    return np.mean(np.asarray(logits_decrease_list))


def upsample(scale=2,npop=5):
    up = torch.nn.Upsample((224,224),mode='bilinear')
    tmp = torch.randn((npop,3,224//scale,224//scale)).cuda()
    uptmp = up(tmp)
    return tmp,uptmp


#testinitcheckrandn
def test(npop,target_label,modelchoice,outputn="",\
         saveres=False,savemax=True,trainmode="PGD",usedownsample=1,scale = 4,device=None,testRandomInit=False):
    if not os.path.exists(outputn) and saveres:
        os.mkdir(outputn)
        print(f"mkdir {outputn}")
    with torch.no_grad():
        

        MSE = nn.MSELoss()  #define mean square error loss

        sigma = 0.1
        sigma_f=0.1 #default 
        lr=0.01 #default
        i = 0
        linf_con = 0.05 
        modelpathp='{}/{}/Weight.pth.tar'.format(outputname,yourweightname)
        model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
        model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
        print(modelpathp)
        model_clean.eval()
        model_adv.eval()
        succ_list=[]
        query_list,l2_list,linf_list,fail_list,all_query_list,all_l2_list,all_linf_list = [],[],[],[],[],[],[]

        imgbase = '../ImageNet_val'      
        imagelist = [f for f in os.listdir(imgbase)]
        imagelist.sort(key=mysortkey)
        trans = transforms.Compose([
                transforms.ToTensor()
                ])
        
        for filename in imagelist:
            imgpath = "{}/{}".format(imgbase,filename)
            label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

            batch = Image.open(imgpath)
            batch = batch.convert("RGB")
            batch = trans(batch).unsqueeze(0)
            
            
            lower = torch.clamp(batch-linf_con,0,1).cuda()
            upper = torch.clamp(batch+linf_con,0,1).cuda()
            batch = (batch-0.5)/0.5
            
            #batch, label = data       # Get a batch,-1,1
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = net(batch*0.5+0.5)
            pre=torch.argmax(output,dim=1)
            if target_label>=0 and pre==target_label:
                continue
            elif pre != label:
                continue
            mu = sigma*torch.randn_like(batch).detach().cuda()
    
            # ===================forward=====================
            with torch.no_grad():
                if modelchoice == "clean":
                    output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                    z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_clean(batch+mu) 
                    
                elif modelchoice=="adv":
                    output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                    z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch+mu) 
                elif modelchoice=="None":
                    output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                    _,_,_,_,_= model_clean(batch) 
                        
                    output ,_,_,_,_,_,\
                    z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch+mu) 

                    output = model_adv.decode(z_vis0,z_vis,\
                                                z2_vis,z3_vis,\
                                                    z4_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem)

                

            if linf_con>0:
                output = (torch.clamp(output*0.5+0.5,lower,upper)-0.5)/0.5
            with torch.no_grad():
                adv_logits = net(output*0.5+0.5)
            adv_pre=torch.argmax(adv_logits,dim=1)
            del adv_logits
            if target_label>=0:
                succ = adv_pre==target_label
            else:
                succ = adv_pre!=label
            advl2 = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1)
            advlinf = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)

            succ_res = torch.where(succ==True)
            if len(succ_res[0])>0:
                succ_l2 = advl2[succ_res]
                succ_linf = advlinf[succ_res]
                minidx = torch.argmin(succ_l2)
                succ_list.append(i)
                query_list.append(1)
                l2_list.append(float(succ_l2[minidx]))
                linf_list.append(float(succ_linf[minidx]))
                all_query_list.append(1)
                all_l2_list.append(float(succ_l2[minidx]))
                all_linf_list.append(float(succ_linf[minidx]))

                if saveres:
                    if savemax:
                        maxidx = torch.argmax(succ_l2)
                    else:
                        maxidx = minidx
                    Image.fromarray(np.array(np.round((output[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                    np.save(f"{outputn}/Resnet18_{i}_adv_target-1_label{int(label)}.npy",(output[minidx]*0.5+0.5).detach().cpu().numpy())

                i+= len(batch)
                print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} \n'\
                    .format(str(i),len(succ_res[0])>0,1, float(torch.mean(advl2)),float(torch.mean(advlinf))))
                #del advl2,advlinf
                if i>=200:
                    break
                continue

            query=1
            succ = False
            while query<10000:
                if usedownsample==1:
                    mu_z,upmuze = upsample(scale=scale,npop=npop)
                    modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                    batch_perturb = batch.repeat(npop,1,1,1)+ modify
                else:
                    upmuze = torch.randn((npop,3,224,224)).cuda()
                    mu_z = upmuze
                    modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                    batch_perturb = batch.repeat(npop,1,1,1)+ modify
                with torch.no_grad():
                    output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,\
                        z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem\
                            = model_adv(batch_perturb) 

                
                    output_inter1 = model_adv.decode(z_vis0.repeat(npop,1,1,1),z_vis.repeat(npop,1,1,1),\
                                                 z2_vis.repeat(npop,1,1,1),z3_vis.repeat(npop,1,1,1),\
                                                    z4_vis.repeat(npop,1,1,1),z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem)
                loss1= MSE(output,batch)
                loss3 = MSE(output_inter1,batch.repeat(npop,1,1,1))
                
                if linf_con>0:
                    output_inter1 = (torch.clamp(output_inter1*0.5+0.5,lower,upper)-0.5)/0.5
                with torch.no_grad():
                    adv_logits = net(output_inter1*0.5+0.5)
                adv_pre=torch.argmax(adv_logits,dim=1)
                del adv_logits
                if target_label>=0:
                    succ = adv_pre==target_label
                else:
                    succ = adv_pre!=label
                query+=npop

                
                loss_black = None
                for jj in range(npop):
                    if loss_black is None:
                        loss_black=adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)
                    else:
                        loss_black = torch.cat((loss_black,adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)),dim=0)


                advl2 = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1)
                advlinf = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)
               

                # print('Img:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                #     .format(str(i+1),query, torch.mean(advl2).data,torch.mean(advlinf).data,loss1.data,loss3.data,\
                #             torch.min(loss_black).data))
                
                succ_res = torch.where(succ==True)
                if len(succ_res[0])>0:
                    succ_l2 = advl2[succ_res]
                    succ_linf = advlinf[succ_res]
                    minidx = torch.argmin(succ_l2)
                    succ_list.append(i)
                    query_list.append(query)
                    l2_list.append(float(succ_l2[minidx]))
                    linf_list.append(float(succ_linf[minidx]))

                    if saveres:
                        if savemax:
                            maxidx = torch.argmax(succ_l2)
                        else:
                            maxidx = minidx
                        Image.fromarray(np.array(np.round((output_inter1[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")

                    break
                else:
                    Reward = -loss_black
                    A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10)
                    
                    if usedownsample>0:
                        downmu = (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224//scale,224//scale)

                        up = torch.nn.Upsample((224,224),mode='bilinear')
                        mu    += up(downmu)
                    else:
                    

                        
                        mu += (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224,224)
                        

                    del A
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache() 
                
            # if not len(succ_res[0])>0:  
            #     fail_list.append(i)  
            #     query_fail_list.append(query)
                #l2_fail_list.append(advl2)
                #linf_fail_list.append(advlinf)
            i+= len(batch)
            print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                .format(str(i),len(succ_res[0])>0,query, float(torch.mean(advl2)),float(torch.mean(advlinf)),loss1.data,loss3.data,\
                        torch.min(loss_black).data))

            # if (i+1)%10==0:
            #     print("Succ rate:{:.4f}".format(len(succ_list)/i))
            #     print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
            #     print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
            #     print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))    

            #     print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
            #     print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
            #     print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))  
            all_query_list.append(query)
            all_l2_list.append(float(torch.mean(advl2)))
            all_linf_list.append(float(torch.mean(advlinf)))
    
            if i>=200:
                break
        print("Succ rate:{:.4f}".format(len(succ_list)/i))
        print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
        print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
        print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))

        print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
        print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
        print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))

        print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_query_list))))
        print("All Avg.l2_list:{:.4f}".format(np.mean(np.asarray(all_l2_list))))
        print("All Avg.linf_list:{:.4f}".format(np.mean(np.asarray(all_linf_list))))

        print("All Median .query:{:.4f}".format(np.median(np.asarray(all_query_list))))
        print("All Median.l2_list:{:.4f}".format(np.median(np.asarray(all_l2_list))))
        print("All Median.linf_list:{:.4f}".format(np.median(np.asarray(all_linf_list))))


        return len(succ_list)/i,np.mean(np.asarray(query_list)),np.median(np.asarray(query_list)),np.mean(np.asarray(all_query_list)),np.median(np.asarray(all_query_list))



@torch.no_grad()
def testBiasedMu(npop,target_label,modelchoice,outputn="",saveres=False,savemax=True,\
                attackType="PGN",usedownsample=1,scale=4):
    if not os.path.exists(outputn) and saveres:
        os.mkdir(outputn)
        print(f"mkdir {outputn}")
    with torch.no_grad():
        

        MSE = nn.MSELoss()  #define mean square error loss
        sigma = 0.1
        sigma_f=0.1 #default 
        lr=0.01 #default
        i = 0
        linf_con = 0.05 # 
        print("target_label:{} sigma:{} sigma_f:{} lr:{} npop:{}".format(float(target_label),sigma,sigma_f,lr,npop))
        modelpathp='{}/{}/AE_model_imagerecWeight_{}_{}.pth.tar'.format(outputname,yourweightname)
                
        model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
        model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
        print(modelpathp)
        model_clean.eval()
        model_adv.eval()
        succ_list=[]
        query_list,l2_list,linf_list,fail_list,all_query_list,all_l2_list,all_linf_list = [],[],[],[],[],[],[]

        imgbase = '../ImageNet_val'      
        imagelist = [f for f in os.listdir(imgbase)]
        imagelist.sort(key=mysortkey)
        trans = transforms.Compose([
                transforms.ToTensor()
                ])
        if  attackType == "FTM":
            from feature_tuning_mixup import ftmAttack
            alpha = 2
            p=1 #prob for DI
            ftsetting = {
                'ftm_beta':0.01,
                'mixup_layer':'conv_linear_include_last',
    'mix_prob':0.1,
    'channelwise':True,
    'mix_upper_bound_feature':0.75,
    'mix_lower_bound_feature':0.,
    'shuffle_image_feature':'SelfShuffle',
    'blending_mode_feature':'M',
    'mixed_image_type_feature':'C',
    'divisor':4

            }
            print(f"FTM alpha:{alpha},p={p},ftsetting:{ftsetting}")
            adversary = ftmAttack(source_models=adv_models, p=p,alpha=alpha,ftsetting=ftsetting,\
                targeted=True,attack_type='RTMF',num_iter=300,max_epsilon=0.05*255,mu=1.0,returnGrad=True)
            
        
        elif attackType=="PGN":
            alpha = 2/255
            step=5
            print(f"step={step},alpha=None")  
            adversary = PGN(adv_models,eps=0.05,steps=step,returnGrad=True,targeted=(target_label>=0))
        
        print(f"alpha:{alpha}")   


        for filename in imagelist:
            imgpath = "{}/{}".format(imgbase,filename)
            label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

            batch = Image.open(imgpath)
            batch = batch.convert("RGB")
            batch = trans(batch).unsqueeze(0)
            
            
            lower = torch.clamp(batch-linf_con,0,1).cuda()
            upper = torch.clamp(batch+linf_con,0,1).cuda()
            batch = (batch-0.5)/0.5
            
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = net(batch*0.5+0.5)
            pre=torch.argmax(output,dim=1)
            if target_label>=0 and pre==target_label:
                continue
            elif pre != label:
                continue
            
            with torch.enable_grad():
                if target_label>=0:
                    if attackType=="FTM":
                        grad = adversary.forward(batch.cuda()*0.5+0.5,label,target_label)
                    else:
                        grad = adversary.forward(batch.cuda()*0.5+0.5,target_label)
                else:                                
                    grad = adversary.forward(batch.cuda()*0.5+0.5,label.cuda())
                grad = (grad-0.5)/0.5#notnorm
            mu = sigma*grad.detach().cuda()


            # ===================forward=====================
            with torch.no_grad():
                if  modelchoice=="adv":
                    output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                    z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch+mu)                             

                elif modelchoice=="None":
                    output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                    _,_,_,_,_= model_clean(batch) 
                        
                    output ,_,_,_,_,_,\
                    z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch+mu) 

                    output = model_adv.decode(z_vis0,z_vis,\
                                                z2_vis,z3_vis,\
                                                    z4_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem)

                
            query=1
            if linf_con>0:
                output = (torch.clamp(output*0.5+0.5,lower,upper)-0.5)/0.5
            with torch.no_grad():
                adv_logits = net(output*0.5+0.5)
            adv_pre=torch.argmax(adv_logits,dim=1)
            del adv_logits
            if target_label>=0:
                succ = adv_pre==target_label
            else:
                succ = adv_pre!=label
            advl2 = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1)
            advlinf = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)

            succ_res = torch.where(succ==True)
            if len(succ_res[0])>0:
                succ_l2 = advl2[succ_res]
                succ_linf = advlinf[succ_res]
                minidx = torch.argmin(succ_l2)
                succ_list.append(i)
                query_list.append(1)
                l2_list.append(float(succ_l2[minidx]))
                linf_list.append(float(succ_linf[minidx]))
                all_query_list.append(1)
                all_l2_list.append(float(succ_l2[minidx]))
                all_linf_list.append(float(succ_linf[minidx]))

                if saveres:
                    if savemax:
                        maxidx = torch.argmax(succ_l2)
                    else:
                        maxidx = torch.argmin(succ_l2)
                    Image.fromarray(np.array(np.round((output[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                
                # print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} \n'\
                #     .format(str(i),len(succ_res[0])>0,1, float(torch.mean(advl2)),float(torch.mean(advlinf))))
                    np.save(f"{outputn}/{i}_{int(label)}.npy",(output[maxidx]*0.5+0.5).detach().cpu().numpy())
                i+= len(batch)
                #del advl2,advlinf
                if i>=200:
                    break

                continue

            
            succ = False
            while query<10000:
                if usedownsample==1:
                    mu_z,upmuze = upsample(scale=scale,npop=npop)
                    modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                    batch_perturb = batch.repeat(npop,1,1,1)+ modify


                else:
                    upmuze = torch.randn((npop,3,224,224)).cuda()
                    mu_z = upmuze
                    modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                    batch_perturb = batch.repeat(npop,1,1,1)+ modify
                with torch.no_grad():
                    output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,\
                        z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem\
                            = model_adv(batch_perturb) 

                
                with torch.no_grad():
                    output_inter1 = model_adv.decode(z_vis0.repeat(npop,1,1,1),z_vis.repeat(npop,1,1,1),\
                                                 z2_vis.repeat(npop,1,1,1),z3_vis.repeat(npop,1,1,1),\
                                                    z4_vis.repeat(npop,1,1,1),z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem)
                loss1= MSE(output,batch)
                loss3 = MSE(output_inter1,batch.repeat(npop,1,1,1))
                
                if linf_con>0:
                    output_inter1 = (torch.clamp(output_inter1*0.5+0.5,lower,upper)-0.5)/0.5
                with torch.no_grad():
                    adv_logits = net(output_inter1*0.5+0.5)
                # if clip:
                #     adv_pre = torch.clamp
                adv_pre=torch.argmax(adv_logits,dim=1)
                del adv_logits
                if target_label>=0:
                    succ = adv_pre==target_label
                else:
                    succ = adv_pre!=label
                query+=npop

                
                loss_black = None
                for jj in range(npop):
                    if loss_black is None:
                        loss_black=adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)
                    else:
                        loss_black = torch.cat((loss_black,adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)),dim=0)


                advl2 = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1)
                advlinf = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)
               

                # print('Img:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                #     .format(str(i+1),query, torch.mean(advl2).data,torch.mean(advlinf).data,loss1.data,loss3.data,\
                #             torch.min(loss_black).data))
                
                succ_res = torch.where(succ==True)
                if len(succ_res[0])>0:
                    succ_l2 = advl2[succ_res]
                    succ_linf = advlinf[succ_res]
                    minidx = torch.argmin(succ_l2)
                    succ_list.append(i)
                    query_list.append(query)
                    l2_list.append(float(succ_l2[minidx]))
                    linf_list.append(float(succ_linf[minidx]))

                    if saveres:
                        if savemax:
                            maxidx = torch.argmax(succ_l2)
                        else:
                            maxidx = torch.argmin(succ_l2)
                        Image.fromarray(np.array(np.round((output_inter1[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                        np.save(f"{outputn}/{i}_{int(label)}.npy",(output_inter1[maxidx]*0.5+0.5).detach().cpu().numpy())

                    #del advl2,advlinf
                    break
                else:
                    Reward = -loss_black
                    A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10)
                    
                    if usedownsample>0:
                        downmu = (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224//scale,224//scale)
                        up = torch.nn.Upsample((224,224),mode='bilinear')
                        mu    += up(downmu)
                        
                    else:
                        mu += (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224,224)

                    del A
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache() 
                # if i%50==0:
                #     print_mem(step=query,device=device) 

            # if not len(succ_res[0])>0:  
            #     fail_list.append(i)  
            #     query_fail_list.append(query)
                #l2_fail_list.append(advl2)
                #linf_fail_list.append(advlinf)
            i+= len(batch)
            print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                .format(str(i),len(succ_res[0])>0,query, float(torch.mean(advl2)),float(torch.mean(advlinf)),loss1.data,loss3.data,\
                        torch.min(loss_black).data))

            # if (i+1)%10==0:
            #     print("Succ rate:{:.4f}".format(len(succ_list)/i))
            #     print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
            #     print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
            #     print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))    

            #     print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
            #     print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
            #     print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))  
            torch.cuda.empty_cache()   
            all_l2_list.append(float(torch.mean(advl2)))
            all_linf_list.append(float(torch.mean(advlinf)))     
            all_query_list.append(query)
            if i>=200:
                break
        print("Succ rate:{:.4f}".format(len(succ_list)/i))
        print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
        print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
        print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))

        print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
        print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
        print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))
        print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_query_list))))
        print("All Avg.l2_list:{:.4f}".format(np.mean(np.asarray(all_l2_list))))
        print("All Avg.linf_list:{:.4f}".format(np.mean(np.asarray(all_linf_list))))

        print("All Median .query:{:.4f}".format(np.median(np.asarray(all_query_list))))
        print("All Median.l2_list:{:.4f}".format(np.median(np.asarray(all_l2_list))))
        print("All Median.linf_list:{:.4f}".format(np.median(np.asarray(all_linf_list))))


        return len(succ_list)/i,np.mean(np.asarray(query_list)),np.median(np.asarray(query_list)),np.mean(np.asarray(all_query_list)),np.median(np.asarray(all_query_list))



def testTranSensitivy():
    target_label=-1
    saveppppp=f"../difplus_{targetmodename}_featuresensi"
    if not os.path.exists(saveppppp):
        os.mkdir(saveppppp)
        print(f"mkdir {saveppppp}")
    MSE = nn.MSELoss()  #define mean square error loss

    sample_idx = 0      # samples to store or show
    total_mse = 0       # total mse error
    total_ssim = 0      # total structure similarity error
    total_psnr = 0      # total PSNR error
    sigma = 0.1
    sigma_f=0.1 
    lr=0.01 
    i = 0
    linf_con = 0.05 
    modelpathp='{}/{}/Weight.pth.tar'.format(outputname)
    model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
    model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
    print(modelpathp)
    model_clean.eval()
    model_adv.eval()
    succ_list=[]
    query_list,l2_list,linf_list,fail_list,query_fail_list,l2_fail_list,linf_fail_list = [],[],[],[],[],[],[]

    imgbase = '../ImageNet_val'      
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)
    from PIL import Image
    trans = transforms.Compose([
            transforms.ToTensor()
            ])
    
    vis_change_list,adv_change_list=[],[]
    xilist = [0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.5,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,4.0]
    #xilist=[0.001]
    print(f"xi list:{xilist}")
    for abc in xilist:
        a,b,c,d,e = abc,abc,abc,abc,abc
        print("abc={}===============================================================".format(a))
        itotal,vis_change,adv_change = 0,0,0
        dif_viss,dif_advs=[],[]
        
        for filename in imagelist:
            imgpath = "{}/{}".format(imgbase,filename)
            label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

            batch = Image.open(imgpath)
            batch = batch.convert("RGB")
            batch = trans(batch).unsqueeze(0)
            
            
            lower = torch.clamp(batch-linf_con,0,1).cuda()
            upper = torch.clamp(batch+linf_con,0,1).cuda()
            batch_surro_adv = (batch.cuda()-0.5)/0.5
            batch = (batch-0.5)/0.5
            
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = net(batch*0.5+0.5)
            pre=torch.argmax(output,dim=1)
            if target_label>=0 and pre==target_label:
                continue
            elif pre != label:
                continue
            # ===================forward=====================
            with torch.no_grad():
                output ,z_vis0_ori_adv,z_vis_ori_adv,z2_vis_ori_adv,z3_vis_ori_adv,z4_vis_ori_adv,\
                z_sem0_ori_adv,z_sem_ori_adv,z2_sem_ori_adv,z3_sem_ori_adv,z4_sem_ori_adv= model_adv(batch_surro_adv) 
                    
                output ,z_vis0_ori_clean,z_vis,z2_vis_ori_clean,z3_vis_ori_clean,z4_vis_ori_clean,\
                z_sem0_ori_clean,z_sem_ori_clean,z2_sem_ori_clean,z3_sem_ori_clean,z4_sem_ori_clean= model_clean(batch_surro_adv) 


                p1 = torch.randn_like(z_vis0_ori_adv)
                p2 = torch.randn_like(z_vis_ori_adv)
                p3 = torch.randn_like(z2_vis_ori_adv)
                p4 = torch.randn_like(z3_vis_ori_adv)
                p5 = torch.randn_like(z4_vis_ori_adv)
                
                perturb_vis0 = z_vis0_ori_adv + a*p1
                perturb_vis1 = z_vis_ori_adv + b*p2
                perturb_vis2 = z2_vis_ori_adv + c*p3
                perturb_vis3 = z3_vis_ori_adv + d*p4
                perturb_vis4 = z4_vis_ori_adv + e*p5

                perturb_adv0 = z_sem0_ori_adv + a*p1
                perturb_adv1 = z_sem_ori_adv +  b*p2
                perturb_adv2 = z2_sem_ori_adv + c*p3
                perturb_adv3 = z3_sem_ori_adv + d*p4
                perturb_adv4 = z4_sem_ori_adv + e*p5


                output_inter1 = model_adv.decode(perturb_vis0,perturb_vis1,\
                                            perturb_vis2,perturb_vis3,\
                                                perturb_vis4,z_sem0_ori_clean,z_sem_ori_adv,z2_sem_ori_adv,z3_sem_ori_adv,z4_sem_ori_adv)
                output_inter2 = model_adv.decode(z_vis0_ori_clean,z_vis,z2_vis_ori_clean,z3_vis_ori_clean,z4_vis_ori_clean,perturb_adv0,perturb_adv1\
                                            ,perturb_adv2,perturb_adv3,perturb_adv4)  


            diff_vis = torch.norm(output_inter1*0.5-batch*0.5) 
            diff_adv = torch.norm(output_inter2*0.5-batch*0.5) 
            dif_viss.append(float(diff_vis))
            dif_advs.append(float(diff_adv))
            with torch.no_grad():
                output_vis = net(output_inter1*0.5+0.5)
                output_adv = net(output_inter2*0.5+0.5)
            pre_vis=torch.argmax(output_vis,dim=1)
            pre_adv=torch.argmax(output_adv,dim=1)
            if pre_vis != label:
                vis_change += 1
            if pre_adv !=label:
                adv_change+=1 
            itotal +=1
            
            if itotal>=200:
                vis_change_list.append(vis_change/itotal)
                adv_change_list.append(adv_change/itotal)
                print("vis_change:{}/{}={},adv_change:{}/{}={}".format(vis_change,itotal,vis_change/itotal,adv_change,itotal,adv_change/itotal))
                print("diff viss:avg:{},std:{};         diff advs:adv:{},std:{}".format(np.mean(np.asarray(dif_viss)),\
                                                                                        np.std(np.asarray(dif_viss)),\
                                                                                            np.mean(np.asarray(dif_advs)),\
                                                                                                np.std(np.asarray(dif_advs))))
                np.save("{}/diff_vis_a{}_clip.npy".format(saveppppp,a),np.asarray(dif_viss))
                np.save("{}/diff_adv_a{}_clip.npy".format(saveppppp,a),np.asarray(dif_advs))

                break
    print(vis_change_list)
    print(adv_change_list)
        
   
@torch.no_grad()
def testTran(npop,target_label,modelchoice,attackType="PGN",\
             outputn="",saveres=False,savemax=True,adv_models=None,usedownsample=0,scale=4):
    if not os.path.exists(outputn) and saveres:
        os.mkdir(outputn)
        print(f"mkdir {outputn}")

    
    MSE = nn.MSELoss()  #define mean square error loss

    sigma = 0.1
    sigma_f=0.1 #default 
    lr=0.01 #default
    i = 0
    linf_con = 0.05 # the same as CGATTACK
    print("target_label:{} sigma:{} sigma_f:{} lr:{} npop:{}".format(float(target_label),sigma,sigma_f,lr,npop))
    modelpathp = '{}/{}/Weight_{}_{}.pth.tar'.format(outputname,yourweightname)
    model_clean.load_state_dict(torch.load(modelpathp)['state_dict_clean'])
    model_adv.load_state_dict(torch.load(modelpathp)['state_dict_adv'])
    print(modelpathp)
    model_clean.eval()
    model_adv.eval()
    succ_list=[]
    query_list,l2_list,linf_list,fail_list,query_fail_list,l2_fail_list,linf_fail_list = [],[],[],[],[],[],[]
    all_query_list,all_l2_list,all_linf_list = [],[],[]

    imgbase = '../ImageNet_val'      
    imagelist = [f for f in os.listdir(imgbase)]
    imagelist.sort(key=mysortkey)
    from PIL import Image
    trans = transforms.Compose([
            transforms.ToTensor()
            ])
    targeted = not (target_label==-1)
    print(f"targeted:{targeted}")
    if attackType=="FTM":
        advt = augmentationTestEns(targeted=targeted,attackType=attackType,adv_models=adv_models,change=True)
    for filename in imagelist:
        imgpath = "{}/{}".format(imgbase,filename)
        label = torch.tensor([int(filename.split("_")[-1].split(".")[0])])

        batch = Image.open(imgpath)
        batch = batch.convert("RGB")
        batch = trans(batch).unsqueeze(0)
        
        
        lower = torch.clamp(batch-linf_con,0,1).cuda()
        upper = torch.clamp(batch+linf_con,0,1).cuda()
        batch = (batch-0.5)/0.5
        
        #batch, label = data       # Get a batch,-1,1
        batch = batch.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = net(batch*0.5+0.5)
        pre=torch.argmax(output,dim=1)
        if target_label>=0 and pre==target_label:
            continue
        elif pre != label:
            continue
        with torch.enable_grad():
            if attackType=="FTM":
                batch_surro_adv,_ = advt(batch.cuda()*0.5+0.5,label.cuda(),target_label)
            else:
                batch_surro_adv,_ = augmentationTest(batch.cuda()*0.5+0.5,label.cuda(),target_label,\
                                                    targeted=targeted,attackType=attackType,adv_models=adv_models)
        batch_surro_adv = (batch_surro_adv-0.5)/0.5

        # ===================forward=====================
        with torch.no_grad():
            if modelchoice=="adv":
                
                output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch_surro_adv) 
           
            elif modelchoice=="None":
                output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                _,_,_,_,_= model_clean(batch) 
                    
                output ,_,_,_,_,_,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_adv(batch_surro_adv) 

                with torch.no_grad():
                    output = model_adv.decode(z_vis0,z_vis,\
                                                z2_vis,z3_vis,\
                                                    z4_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem)
            if linf_con>0:
                output = (torch.clamp(output*0.5+0.5,lower,upper)-0.5)/0.5
            with torch.no_grad():
                adv_logits = net(output*0.5+0.5)
            # if clip:
            #     adv_pre = torch.clamp
            adv_pre=torch.argmax(adv_logits,dim=1)
            del adv_logits
            if target_label>=0:
                succ = adv_pre==target_label
            else:
                succ = adv_pre!=label
            advl2 = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1)
            advlinf = torch.norm((output*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)

            succ_res = torch.where(succ==True)
            if len(succ_res[0])>0:
                succ_l2 = advl2[succ_res]
                succ_linf = advlinf[succ_res]
                minidx = torch.argmin(succ_l2)
                succ_list.append(i)
                query_list.append(1)
                l2_list.append(float(succ_l2[minidx]))
                linf_list.append(float(succ_linf[minidx]))

                all_query_list.append(1)
                all_l2_list.append(float(succ_l2[minidx]))
                all_linf_list.append(float(succ_linf[minidx]))

                if saveres:
                    maxidx = torch.argmax(succ_l2)
                    Image.fromarray(np.array(np.round((output_inter1[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}.png")
                i+= len(batch)
                print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} \n'\
                    .format(str(i),len(succ_res[0])>0,1, float(torch.mean(advl2)),float(torch.mean(advlinf))))
                #del advl2,advlinf
                if i>=200:
                    break
                continue


        mu = sigma*torch.randn_like(batch).detach().cuda()
        query=1
        succ = False


        while query<10000:
            if usedownsample==1:
                mu_z,upmuze = upsample(scale=scale,npop=npop)
                modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                batch_perturb = batch_surro_adv.repeat(npop,1,1,1)+ modify
           
            else:
                upmuze = torch.randn((npop,3,224,224)).cuda()
                mu_z = upmuze
                modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                batch_perturb = batch_surro_adv.repeat(npop,1,1,1)+ modify        
            with torch.no_grad():
                output_p ,z_p_vis0,z_p_vis,z2_p_vis,z3_p_vis,z4_p_vis,\
                    z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem\
                        = model_adv(batch_perturb) 

            
            with torch.no_grad():
                output_inter1 = model_adv.decode(z_vis0.repeat(npop,1,1,1),z_vis.repeat(npop,1,1,1),\
                                                z2_vis.repeat(npop,1,1,1),z3_vis.repeat(npop,1,1,1),\
                                                z4_vis.repeat(npop,1,1,1),z_p_sem0,z_p_sem,z2_p_sem,z3_p_sem,z4_p_sem)
            
            if linf_con>0:
                output_inter1 = (torch.clamp(output_inter1*0.5+0.5,lower,upper)-0.5)/0.5
            with torch.no_grad():
                adv_logits = net(output_inter1*0.5+0.5)
            # if clip:
            #     adv_pre = torch.clamp
            adv_pre=torch.argmax(adv_logits,dim=1)
            del adv_logits
            if target_label>=0:
                succ = adv_pre==target_label
            else:
                succ = adv_pre!=label
            query+=npop

            
            loss_black = None
            for jj in range(npop):
                if loss_black is None:
                    loss_black=adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)
                else:
                    loss_black = torch.cat((loss_black,adv_loss(output_inter1[jj].unsqueeze(0)*0.5+0.5,label,target=target_label,models=[net]).unsqueeze(0)),dim=0)


            advl2 = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1)
            advlinf = torch.norm((output_inter1*0.5-batch*0.5).flatten(start_dim=1),dim=1,p=np.inf)
            

            # print('Img:{} query:{} advL2:{:.6f} advLinf:{:.6f} , minlossblack:{:.6f}\n'\
            #     .format(str(i+1),query, torch.mean(advl2).data,torch.mean(advlinf).data,\
            #             torch.min(loss_black).data))
            
            succ_res = torch.where(succ==True)
            if len(succ_res[0])>0:
                succ_l2 = advl2[succ_res]
                succ_linf = advlinf[succ_res]
                minidx = torch.argmin(succ_l2)
                succ_list.append(i)
                query_list.append(query)
                l2_list.append(float(succ_l2[minidx]))
                linf_list.append(float(succ_linf[minidx]))

                if saveres:
                    maxidx = torch.argmax(succ_l2)
                    if savemax:
                        Image.fromarray(np.array(np.round((output_inter1[maxidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                        np.save(f"{outputn}/Resnet18_{i}_adv_target-1_label{int(label)}.npy",(output_inter1[maxidx]*0.5+0.5).detach().cpu().numpy())
                    else:
                        Image.fromarray(np.array(np.round((output_inter1[minidx]*0.5+0.5).permute(1,2,0).detach().cpu().numpy()*255),dtype=np.uint8)).save(f"{outputn}/{i}_{int(label)}.png")
                        np.save(f"{outputn}/Resnet18_{i}_adv_target-1_label{int(label)}.npy",(output_inter1[minidx]*0.5+0.5).detach().cpu().numpy())

                break
            else:
                Reward = -loss_black
                A      = (Reward - torch.mean(Reward))/(torch.std(Reward) + 1e-10)
                if usedownsample==1:
                    downmu = (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224//scale,224//scale)

                    up = torch.nn.Upsample((224,224),mode='bilinear')
                    mu    += up(downmu)
                else:
                    mu += (lr/ (npop * sigma_f))*(torch.matmul(mu_z.flatten(start_dim=1).t(), A.view(-1, 1))).view(1, -1).reshape(-1,3,224,224)
                del A

        i+= len(batch)
        print('Img:{} succ:{} query:{} \n'.format(str(i),len(succ_res[0])>0,query))

        # if (i+1)%10==0:
        #     print("Succ rate:{:.4f}".format(len(succ_list)/i))
        #     print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
        #     print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
        #     print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))    

        #     print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
        #     print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
        #     print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))  
        torch.cuda.empty_cache()  
        all_l2_list.append(float(torch.mean(advl2)))
        all_linf_list.append(float(torch.mean(advlinf)))     
        all_query_list.append(query)
        
        if i>=200:
            break

        if i% 10==0:
            print("Succ rate:{:.4f}".format(len(succ_list)/i))
            print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
            print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
            print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))

            print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
            print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
            print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))
            print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_query_list))))
            print("All Avg.l2_list:{:.4f}".format(np.mean(np.asarray(all_l2_list))))
            print("All Avg.linf_list:{:.4f}".format(np.mean(np.asarray(all_linf_list))))

            print("All Median .query:{:.4f}".format(np.median(np.asarray(all_query_list))))
            print("All Median.l2_list:{:.4f}".format(np.median(np.asarray(all_l2_list))))
            print("All Median.linf_list:{:.4f}".format(np.median(np.asarray(all_linf_list))))

    print("Succ rate:{:.4f}".format(len(succ_list)/i))
    print("Succ Avg.query:{:.4f}".format(np.mean(np.asarray(query_list))))
    print("Succ Avg.l2_list:{:.4f}".format(np.mean(np.asarray(l2_list))))
    print("Succ Avg.linf_list:{:.4f}".format(np.mean(np.asarray(linf_list))))

    print("Succ Median .query:{:.4f}".format(np.median(np.asarray(query_list))))
    print("Succ Median.l2_list:{:.4f}".format(np.median(np.asarray(l2_list))))
    print("Succ Median.linf_list:{:.4f}".format(np.median(np.asarray(linf_list))))
    print("All Avg.query:{:.4f}".format(np.mean(np.asarray(all_query_list))))
    print("All Avg.l2_list:{:.4f}".format(np.mean(np.asarray(all_l2_list))))
    print("All Avg.linf_list:{:.4f}".format(np.mean(np.asarray(all_linf_list))))

    print("All Median .query:{:.4f}".format(np.median(np.asarray(all_query_list))))
    print("All Median.l2_list:{:.4f}".format(np.median(np.asarray(all_l2_list))))
    print("All Median.linf_list:{:.4f}".format(np.median(np.asarray(all_linf_list))))


    return len(succ_list)/i,np.mean(np.asarray(query_list)),np.median(np.asarray(query_list)),np.mean(np.asarray(all_query_list)),np.median(np.asarray(all_query_list))


def augmentation(x, true_lab, targeted=False,attackType="pgd"):

    model_idx = np.random.randint(0, len(adv_models))
    model_chosen = adv_models[model_idx]
    if   attackType!="pgdMulti":                
        with ctx_noparamgrad_and_eval(model_chosen):
            xadv = adversary.perturb(x, true_lab if targeted else None)
            adv_label = adversary._get_predicted_label(xadv)
            #print("xadv norm:{}".format(torch.mean(torch.norm(xadv.flatten(start_dim=1)-x.flatten(start_dim=1),dim=0))))
            #save_image(xadv,'test.png')
            if targeted:
                attack_acc= len(torch.where(adv_label==true_lab)[0])
            else:
                attack_acc= len(torch.where(adv_label!=true_lab)[0])
            #print("augmentation attack acc:{:.4f}".format(attack_acc/len(adv_label)))
            return xadv,adv_label
    elif attackType=="pgdMulti":
        xadv = adversary(x,true_lab)
        return xadv,None
   
class augmentationTestEns(nn.Module):
    def __init__(self,targeted=False,attackType="pgd",adv_models=None,change=True):
        super().__init__()
        self.attackType = attackType
        if attackType=="pgdMulti":

            self.adversary = PGDMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=30,random_start=True,targeted=targeted
            )  
        elif attackType=="PGN":
            surro_model_idx="all"
            step=5
            print(f"step:{step},surro_model_idx:{surro_model_idx}")
            #adversary = PGN(adv_models[surro_model_idx],eps=0.05)
            self.adversary = PGN(adv_models,eps=0.05,steps=step)
        elif attackType=="timifgsm":
            self.adversary = timMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=30,random_start=True,targeted=targeted,
                change=change
            )  
        elif attackType=="sinifgsm":
            self.adversary = siniMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
            )  
        elif attackType=="sinitifgsm":
            # if targeted:
            #     adversary = siniMultiModel(
            #         adv_models,eps=0.05,alpha=2/255,steps=30,targeted=targeted,tidi=True
            #     ) 
            # else:
            self.adversary = siniMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted,tidi=True
            ) 
        elif attackType=="vnifgsm":
            self.adversary = vniMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
            )
        elif attackType=="vnitifgsm":
            step=10
            #print(f"step={step}")
            self.adversary = vniMultiModel(
                adv_models,eps=0.05,alpha=2/255,steps=step,targeted=targeted,tidi=True
            )         
        elif attackType=="mtimifgsm" or attackType=="mtimifgsmti" :
            net_update_list = []
            for modelname in ref.split(","):
                net_update = utils.load_adv_imagenet([modelname])[0] 
                net_update_list.append(net_update)
            self.adversary = MMIFGSM(net_update_list,eps=0.05,alpha=2/255,steps=5,decay=1,modelnum=len(net_update_list),targeted=targeted)


        elif  attackType == "FTM":
            from feature_tuning_mixup import ftmAttack
            alpha = 2
            p=1 #prob for DI
            ftsetting = {
                'ftm_beta':0.01,
                'mixup_layer':'conv_linear_include_last',
    'mix_prob':0.1,
    'channelwise':True,
    'mix_upper_bound_feature':0.75,
    'mix_lower_bound_feature':0.,
    'shuffle_image_feature':'SelfShuffle',
    'blending_mode_feature':'M',
    'mixed_image_type_feature':'C',
    'divisor':4

            }
            print(f"FTM alpha:{alpha},p={p},ftsetting:{ftsetting}")
            self.adversary = ftmAttack(source_models=adv_models, p=p,alpha=alpha,ftsetting=ftsetting,\
                targeted=True,attack_type='RTMF',num_iter=300,max_epsilon=0.05*255,mu=1.0,returnGrad=False)
    def forward(self,x, true_lab, target_label):
        if  self.attackType!="pgdMulti" and self.attackType!="timifgsm" and self.attackType!="mtimifgsm" \
            and self.attackType!="mtimifgsmti" and self.attackType!="sinifgsm" and\
                self.attackType!="sinitifgsm" and self.attackType!="vnifgsm" and \
                    self.attackType!="vnitifgsm" and self.attackType!="AWT" and self.attackType!="PGN" and self.attackType!="FTM":   
            pass             
        elif self.attackType=="pgdMulti" or self.attackType=="timifgsm" or self.attackType=="sinifgsm" or self.attackType=="sinitifgsm" \
            or self.attackType=="vnifgsm" or self.attackType=="vnitifgsm":
            xadv = self.adversary(x,translabel)
            return xadv,None
        elif self.attackType == "mtimifgsm":
            adv0 = self.adversary.forward_ditifgsm(x,true_lab) #tidimifgsm
            xadv = self.adversary.forward_modellist(adv0,true_lab,clean=None)# v2 ,mifgsm
            return xadv,None
        elif self.attackType == "PGN":
            xadv = self.adversary.forward(x, true_lab)
            return xadv,None
        elif self.attackType == "AWT":
            perturbation = self.adversary.forward(x,true_lab,clip=True)
            perturbation = torch.clamp(perturbation, -0.05, 0.05)
            xadv = torch.clamp(x+perturbation,0,1)
            return xadv,None
        elif self.attackType == "mtimifgsmti":
            adv0 = self.adversary.forward_ditifgsm(x,true_lab) #tidimifgsm
            xadv = self.adversary.forward_modellist_ti(adv0,true_lab,clean=None)# v2 ,mifgsm
            return xadv,None    
        elif  self.attackType == "FTM":
            xadv = self.adversary.forward(x,true_lab,target_label)
            return xadv,None
 
def augmentationTest(x, true_lab, target_label,targeted=False,attackType="pgd",adv_models=None,change=True):
    model_idx = np.random.randint(0, len(adv_models))
    model_chosen = adv_models[model_idx]
    if targeted:
        translabel = target_label
    else:
        translabel = true_lab
    if attackType =="pgd":
        adversary = LinfPGDAttack(
            model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
            nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
            clip_max=1.0, targeted=targeted)
    elif attackType=="pgdMulti":

        adversary = PGDMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=30,random_start=True,targeted=targeted
        )  
    elif attackType=="PGN":
        surro_model_idx="all"
        step=5
        print(f"step:{step},surro_model_idx:{surro_model_idx}")
        adversary = PGN(adv_models,eps=0.05,steps=step)
    elif attackType=="timifgsm":
        adversary = timMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=30,random_start=True,targeted=targeted,
            change=change
        )  
    elif attackType=="sinifgsm":
        adversary = siniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
        )  
    elif attackType=="sinitifgsm":
        adversary = siniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted,tidi=True
        ) 
    elif attackType=="vnifgsm":
        adversary = vniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=10,targeted=targeted
        )
    elif attackType=="vnitifgsm":
        step=10
        #print(f"step={step}")
        adversary = vniMultiModel(
            adv_models,eps=0.05,alpha=2/255,steps=step,targeted=targeted,tidi=True
        )         
    elif attackType=="mtimifgsm" or attackType=="mtimifgsmti" :
        net_update_list = []
        for modelname in ref.split(","):
            net_update = utils.load_adv_imagenet([modelname])[0] 
            net_update_list.append(net_update)
        adversary = MMIFGSM(net_update_list,eps=0.05,alpha=2/255,steps=5,decay=1,modelnum=len(net_update_list),targeted=targeted)

    elif attackType =="cw":
        adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                          ,max_iterations=10000) 
    elif attackType =="mifgsm":
        adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 
        
    elif attackType =="multi2":
        

        a = random.randint(0, 1)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        else:
            adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 
                      
        
    elif attackType =="multi":
        

        a = random.randint(0, 1)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        else:
            adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                            ,max_iterations=10000)    
    elif attackType =="multi3":
        

        a = random.randint(0, 2)
        if a ==0:
            adversary = LinfPGDAttack(
                model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
                nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=targeted) 
        elif a==1:
            adversary = CarliniWagnerL2Attack(model_chosen,1000,confidence=0,targeted=targeted,learning_rate=0.2,binary_search_steps=1
                                            ,max_iterations=10000)   
        else:
            adversary = LinfMomentumIterativeAttack(model_chosen,loss_fn=nn.CrossEntropyLoss(reduction="sum"),eps=0.05,targeted=targeted) 

    if   attackType!="pgdMulti" and attackType!="timifgsm" and attackType!="mtimifgsm" \
        and attackType!="mtimifgsmti" and attackType!="sinifgsm" and\
              attackType!="sinitifgsm" and attackType!="vnifgsm" and \
                attackType!="vnitifgsm" and attackType!="AWT" and attackType!="PGN" and attackType!="FTM":                
        with ctx_noparamgrad_and_eval(model_chosen):
            #xadv = adversary.perturb(x, true_lab if targeted else None)
            xadv = adversary.perturb(x, translabel if targeted else None)
            adv_label = adversary._get_predicted_label(xadv)
            #print("xadv norm:{}".format(torch.mean(torch.norm(xadv.flatten(start_dim=1)-x.flatten(start_dim=1),dim=0))))
            #save_image(xadv,'test.png')
            if targeted:
                attack_acc= len(torch.where(adv_label==true_lab)[0])
            else:
                attack_acc= len(torch.where(adv_label!=true_lab)[0])
            #print("augmentation attack acc:{:.4f}".format(attack_acc/len(adv_label)))
            return xadv,adv_label
    elif attackType=="pgdMulti" or attackType=="timifgsm" or attackType=="sinifgsm" or attackType=="sinitifgsm" \
        or attackType=="vnifgsm" or attackType=="vnitifgsm":
        xadv = adversary(x,translabel)
        return xadv,None
    elif attackType == "mtimifgsm":
        adv0 = adversary.forward_ditifgsm(x,true_lab) #tidimifgsm
        xadv = adversary.forward_modellist(adv0,true_lab,clean=None)# v2 ,mifgsm
        return xadv,None
    elif attackType == "PGN":
        xadv = adversary.forward(x, true_lab)
        return xadv,None
    elif attackType == "AWT":
        perturbation = adversary.forward(x,true_lab,clip=True)
        perturbation = torch.clamp(perturbation, -0.05, 0.05)
        xadv = torch.clamp(x+perturbation,0,1)
        return xadv,None
    elif attackType == "mtimifgsmti":
        adv0 = adversary.forward_ditifgsm(x,true_lab) #tidimifgsm
        xadv = adversary.forward_modellist_ti(adv0,true_lab,clean=None)# v2 ,mifgsm
        return xadv,None    
    elif  attackType == "FTM":
        from feature_tuning_mixup import ftmAttack
        alpha = 2
        p=1 #prob for DI
        ftsetting = {
            'ftm_beta':0.01,
            'mixup_layer':'conv_linear_include_last',
'mix_prob':0.1,
'channelwise':True,
'mix_upper_bound_feature':0.75,
'mix_lower_bound_feature':0.,
'shuffle_image_feature':'SelfShuffle',
'blending_mode_feature':'M',
'mixed_image_type_feature':'C',
'divisor':4

        }
        print(f"FTM alpha:{alpha},p={p},ftsetting:{ftsetting}")
        adversary = ftmAttack(source_models=adv_models, p=p,alpha=alpha,ftsetting=ftsetting,\
            targeted=True,attack_type='RTMF',num_iter=300,max_epsilon=0.05*255,mu=1.0,returnGrad=False)
        
        xadv = adversary.forward(x,true_lab,target_label)
        return xadv,None


def adv_loss( y, label,target=-1,models=None,returnlist=False,margin=None,logits=None):
    if returnlist:
        loss = []
    else:
        loss = 0.
    if margin is None:
        margin=innermargin
    else:
        margin = margin
    if models is None:
        models = adv_models
    for adv_model in models:
#            loss = 0.
        with torch.no_grad():
            logits = adv_model(y)

        if target==-1:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
        else:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = target.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
            margin = torch.nn.functional.relu(diff + margin, True) - margin
            #margin = diff
        if returnlist:
            loss=margin
        else:
            loss += margin.mean()
    if not returnlist:
        loss /= len(models)
        
    return loss 

def drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5,yourweightname):
    x = [x for x in range(len(avg_loss))]
    plt.figure()
    plt.plot(x,avg_loss)
    plt.savefig("{}/{}/avg_loss.jpg".format(outputname,yourweightname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss1)
    plt.savefig("{}/{}/avg_loss1.jpg".format(outputname,yourweightname))
    plt.close()
    plt.figure()
    plt.plot(x,avg_loss3)
    plt.savefig("{}/{}/avg_loss3.jpg".format(outputname,yourweightname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss4)
    plt.savefig("{}/{}/avg_loss4.jpg".format(outputname,yourweightname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss5)
    plt.savefig("{}/{}/avg_loss5.jpg".format(outputname,yourweightname))
    plt.close()


def adv_loss_train( y, label,target=False,randommargin=False):
    loss = 0.
    
    for adv_model in adv_models:
        logits = adv_model(y)
        if randommargin:
            randommargin = int(torch.randint(0,innermargin,(1,)))
        if not target:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            if not randommargin:
                margin = torch.nn.functional.relu(diff + innermargin, True) - innermargin
            else:
                margin = torch.nn.functional.relu(diff + randommargin, True) - randommargin
        else:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            one_hot = one_hot.bool()
            diff = torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0] - logits[one_hot]
            if not randommargin:
                margin = torch.nn.functional.relu(diff + innermargin, True) - innermargin
            else:
                margin = torch.nn.functional.relu(diff + randommargin, True) - randommargin
        # print(diff)
        # print(margin)
        loss += margin.mean()
    loss /= len(adv_models)
        
    return loss 





def train(advType = 'pgd',epoch_num=0):
    MSE = nn.MSELoss()
    rm = False
    print("advType={},randommargin={}".format(advType,rm)) 

    #Training
    total_iter = 0
    avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5=[],[],[],[],[]
    for epoch in range(num_epochs):
        total_loss,total_loss1,total_loss2,total_loss3,total_loss4,total_loss5 = 0,0,0,0,0,0
        totalimg = 0
        innerloop=0
        for _,data in enumerate(train_loader):
            model_clean.train()
            model_adv.train()
            batch, label = data       # Get a batch,-1,1
            batch = (batch).cuda()
            label = label.cuda()
            batch_adv,adv_label = augmentation(batch*0.5+0.5,label,targeted=False,attackType=advType)
            batch_adv = (batch_adv-0.5)/0.5
            # ===================forward=====================

            output ,z_vis0,z_vis,z2_vis,z3_vis,z4_vis,\
                z_sem0,z_sem,z2_sem,z3_sem,z4_sem= model_clean(batch)  
            output_adv ,z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,\
                z4_adv_vis,z_adv_sem0,z_adv_sem,z2_adv_sem,z3_adv_sem,z4_adv_sem= model_adv(batch_adv)      
            totalimg += len(batch)
            
            output_inter1 = model_adv.decode(z_vis0,z_vis,z2_vis,z3_vis,z4_vis,z_adv_sem0,z_adv_sem,z2_adv_sem,z3_adv_sem,z4_adv_sem)
            output_inter2 = model_clean.decode(z_adv_vis0,z_adv_vis,z2_adv_vis,z3_adv_vis,z4_adv_vis,z_sem0,z_sem,z2_sem,z3_sem,z4_sem)


            
            loss1= MSE(output,batch)+MSE(output_adv,batch_adv)
            loss3 = MSE(output_inter1,batch)+MSE(output_inter2,batch_adv)
            loss4 = adv_loss_train(output_inter1*0.5+0.5,label,randommargin=rm) 
            loss5 = adv_loss_train(output_inter2*0.5+0.5,label,target=True,randommargin=rm)

            loss =loss1+loss3+loss4+loss5
            
            # ===================backward====================
            optimizer_clean.zero_grad()
            optimizer_adv.zero_grad()
            loss.backward()
            optimizer_clean.step()
            optimizer_adv.step()


            # ===================log========================
            total_loss += loss.data
            total_loss1 += loss1.data
            total_loss3 += loss3.data
            total_loss4 += loss4.data
            total_loss5 += loss5.data

            if innerloop%100 ==0:
                print('\nepoch [{}/{}], loss:{:.6f}, loss1:{:.6f}, loss3:{:.6f}, loss4:{:.6f}, loss5:{:.6f}\n'
                .format(epoch+1, num_epochs, total_loss/totalimg,total_loss1/totalimg,\
                                    total_loss3/totalimg,total_loss4/totalimg,total_loss5/totalimg))
            innerloop+=1
            total_iter+=1
            if total_iter%30==0:
                avg_loss.append(float(total_loss/totalimg))
                avg_loss1.append(float(total_loss1/totalimg))
                avg_loss3.append(float(total_loss3/totalimg))
                avg_loss4.append(float(total_loss4/totalimg))
                avg_loss5.append(float(total_loss5/totalimg))

                drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5)
                # np.save("{}/{}/avg_loss.npy".format(outputname,yourweightname),np.array(avg_loss)
                #         )
                # np.save("{}/{}/avg_loss1.npy".format(outputname,yourweightname),np.array(avg_loss1)
                #         )
                # np.save("{}/{}/avg_loss3.npy".format(outputname,yourweightname),np.array(avg_loss3)
                #         )
                # np.save("{}/{}/avg_loss4.npy".format(outputname,yourweightname),np.array(avg_loss4)
                #         )
                # np.save("{}/{}/avg_loss5.npy".format(outputname,yourweightname),np.array(avg_loss5)
                #         )


            if innerloop%1000==0:
                out = torch.cat((batch[:2],output[:2]),0)
                out = to_img(out.cpu().data)
                
            if innerloop%1000==0:
                save_dict = {
                    'epoch': epoch + 1,
                    'loop':innerloop,
                    'batchsize':len(batch),
                    'state_dict_clean': model_clean.state_dict(),
                    'state_dict_adv': model_adv.state_dict(),
                
                }
                torch.save(save_dict, '{}/{}/Weight_{}_{}.pth.tar'.format(outputname,yourweightname,epoch+1,innerloop))

                save_dict = {
                    'epoch': epoch + 1,
                    'loop':innerloop,
                    'batchsize':len(batch),
                    'optimizer_clean' : optimizer_clean.state_dict(),
                    'optimizer_adv' : optimizer_adv.state_dict()    
                }
                torch.save(save_dict, '{}/{}/Opt.pth.tar'.format(outputname,yourweightname))
                                


           
        save_dict = {
        'epoch': epoch + 1,
        'loop':innerloop,
        'batchsize':len(batch),
        'state_dict_clean': model_clean.state_dict(),
        'state_dict_adv': model_adv.state_dict(),
    
        }

        torch.save(save_dict, '{}/{}/Weight_{}_{}.pth.tar'.format(outputname,yourweightname,epoch+1,innerloop))

        save_dict = {
        'epoch': epoch + 1,
        'loop':innerloop,
        'batchsize':len(batch),
                'optimizer_clean' : optimizer_clean.state_dict(),
                'optimizer_adv' : optimizer_adv.state_dict() 
                } 
        torch.save(save_dict, '{}/{}/Opt_{}_{}.pth.tar'.format(outputname,epoch+1,innerloop))


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setSeed(0)
    outputname='yourpath/weights'
    yourweightname="weight1"
    para_dict={
        #for train
    'num_epochs' :1, 
    'batch_size' :16,
    'learning_rate' : 1e-4,
    'innermargin' :5,
    'epoch_num'  :100000,
    'advType':"pgd",
    'resume':False,
    'resumePath':f"{outputname}/xxx_modelweight.pth.tar",
    'resumePathOpt':f"{outputname}/xxx_opt.pth.tar",

    #for test:
    'targeted': False,
    'targetLabel':torch.tensor([864]).cuda(), 
    'usedownsample':1,
    'scale':4,
    'transType':"PGN", #FTM,UNI,TTA,MEF,PGN,AWT,pgdMulti,timifgsm,None,mtimifgsm,mtimifgsmti,sinifgsm,sinitifgsm,vnifgsm,vnitifgsm;
    'targetmodename':'Resnet18' ###ConvNextBase,EfficientB3,SwinV2T,Resnet101,VGG16,Squeezenet,Googlenet,Resnet18,wrs50

    }

    mode="train"

    num_epochs = para_dict['num_epochs']
    batch_size = para_dict['batch_size']
    learning_rate = para_dict['learning_rate']
    innermargin = para_dict['innermargin']



    model_clean = Autoencoder().cuda()
    model_adv = Autoencoder().cuda()

    #Optimizer
    optimizer_clean = torch.optim.Adam(model_clean.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    optimizer_adv = torch.optim.Adam(model_adv.parameters(), lr=learning_rate,
                                weight_decay=1e-5)
    
    
    if mode=="train":
        save_dict = {
        'epoch': 0,
        'state_dict_clean': model_clean.state_dict(),
        'state_dict_adv': model_adv.state_dict(),
        
    }
        #ref="Resnet101,SwinV2T,ConvNextBase" #ConvNextBase,EfficientB3,SwinV2T,Resnet101
        ref="Resnet18,Googlenet,Squeezenet" #VGG16,Resnet18,Squeezenet,Googlenet

        print(ref)

        adv_models= utils.load_adv_imagenet(ref.split(","),device=device) 



        img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5), (0.5))
        ])

        img_transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
        ])

        imagenet_traindata = ImageFolder('yourimagenetpath/train',transform=img_transform)
        train_loader = torch.utils.data.DataLoader(imagenet_traindata,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)

        if not os.path.exists('{}/{}'.format(outputname,yourweightname)):
            os.mkdir('{}/{}'.format(outputname,yourweightname))


        if para_dict['resume']:
            model_clean.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_clean'])
            model_adv.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_adv'])

        train(advType=para_dict['advType'],epoch_num=para_dict['epoch_num'])
    else:

        ###
        targetmodename = para_dict["targetmodename"]
        targeted = para_dict["targeted"]
        targetLabel = -1 if not targeted else para_dict["targetLabel"]
        if targetmodename in popdict.keys():
            npop = popdict[targetmodename][1 if targeted else 0]
        else:
            raise NotImplementedError
        if targetmodename in refs.keys():
            ref = refs[targetmodename]
            print(ref)
        else:
            raise NotImplementedError        
        
        testSensitivy = False

        graddownsample=False
        compareRandom=False
        openscenario = False
        useBiasedMu = True 
        testRandomInit = False
        transType = para_dict["transType"]
        if transType == "None" :#openset scenarios, mu=0
            modelchoice="None"
        else:##closeset scenarios, use transferable attack methods
            modelchoice="adv"
        usedownsample = para_dict["usedownsample"]
        
        scale = para_dict["scale"]
        print(f"compareRandom={compareRandom};scale;{scale},graddownsample:{graddownsample},useBiasedMu:{useBiasedMu},usedownsample:{usedownsample},targetmodename:{targetmodename},targeted:{targeted},transType:{transType},modelchoice:{modelchoice},val={val}")

        saveres=True
        savemax=True#
        #saveoutputn = f"advimgs/{targetmodename}_{modelchoice}_{transType}_untarget_minl2"
        saveoutputn = f"advimgs/{targetmodename}_{modelchoice}_{transType}_untarget_withlabel_new"
        if savemax:
            saveoutputn += "_maxl2"
        else:
            saveoutputn += "_minl2"

        if targetmodename == 'wrs50':
            from robustness import model_utils
            from robustness import datasets as datasetsr
            modelpath = '../weights/microsoft_robust_models/wide_resnet50_2_linf_eps8.0_imgnet.ckpt'

            print(modelpath)
            net ,_ = model_utils.make_and_restore_model(arch="wide_resnet50_2",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
            net = MyRobustModel(net).eval().cuda()
            modelpath = '../weights/microsoft_robust_models/resnet50_linf_eps8.0_imgnet.ckpt'
            print(f"surrogate model:{modelpath}")
            adv_models ,_ = model_utils.make_and_restore_model(arch="resnet50",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
            adv_models = [MyRobustModel(adv_models).eval()]

        
        if targetmodename != 'wrs50':
        
            net = utils.load_adv_imagenet([targetmodename],device=device)[0] 
            adv_models= utils.load_adv_imagenet(ref.split(",")) 
        
           
        
        if testSensitivy:
            testTranSensitivy()
            
        if transType == "None":
            
            asri,q,medq,all_avgq,all_medq=test(npop=npop,target_label=targetLabel,modelchoice=modelchoice,\
                        outputn=saveoutputn,saveres=saveres,\
                            usedownsample=usedownsample,scale=scale,\
                                device=device,testRandomInit=testRandomInit)
                 
        else:
            
            if useBiasedMu:
                asri,q,medq,all_avgq,all_medq=testBiasedMu(npop=npop,target_label=targetLabel,modelchoice=modelchoice,\
                                attackType=transType,\
                                    outputn=saveoutputn,saveres=saveres,savemax=savemax,usedownsample=usedownsample,\
                                        scale=scale)
            else:
                asri,q,medq,all_avgq,all_medq=testTran(npop=npop,target_label=targetLabel,modelchoice=modelchoice,\
                        attackType=transType,\
                            outputn=saveoutputn,saveres=saveres,\
                                savemax=savemax,adv_models=adv_models,usedownsample=usedownsample,scale=scale)
          