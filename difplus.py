import os
from pgd_multimodel import PGN
from config import refs,popdict,modelpathdict
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack

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


import utils
import random
print("difplus.py") 

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



def upsample(scale=2,npop=5):
    up = torch.nn.Upsample((224,224),mode='bilinear')
    tmp = torch.randn((npop,3,224//scale,224//scale)).cuda()
    uptmp = up(tmp)
    return tmp,uptmp


#testinitcheckrandn
def test(npop,target_label,modelchoice,modelpath="",usedownsample=1,scale = 4):
    with torch.no_grad():
        

        MSE = nn.MSELoss()  

        sigma = 0.1
        sigma_f=0.1 #default 
        lr=0.01 
        i = 0
        linf_con = 0.05 # the same as CGATTACK
    
        model_clean.load_state_dict(torch.load(modelpath)['state_dict_clean'])
        model_adv.load_state_dict(torch.load(modelpath)['state_dict_adv'])
        print(f"{modelpath} loaded")
        model_clean.eval()
        model_adv.eval()
        succ_list=[]
        query_list,l2_list,linf_list,fail_list,all_query_list,all_l2_list,all_linf_list = [],[],[],[],[],[],[]

        imgbase = '/dataset/ImageNetVal_random_Cropped224'      
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
            
            batch = batch.cuda()
            label = label.cuda()
            with torch.no_grad():
                output = net(batch*0.5+0.5)
            pre=torch.argmax(output,dim=1)
            if target_label == "least":

                target_label = torch.argsort(output, dim=1)[:, 0]

            elif target_label == "last":

                # Easiest target: class with the highest logit except the predicted class
                sorted_idx = torch.argsort(output, dim=1, descending=True)
                target_label = sorted_idx[:, 1]

            if target_label>=0 and pre==target_label:
                continue
            elif pre != label:
                continue  
            mu = sigma*torch.randn_like(batch).detach().cuda()
            # ===================forward=====================
            with torch.no_grad():                    
                if modelchoice=="adv":
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

                
                i+= len(batch)
                print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} \n'\
                    .format(str(i),len(succ_res[0])>0,1, float(torch.mean(advl2)),float(torch.mean(advlinf))))
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
                
            i+= len(batch)
            print('Img:{} succ:{} query:{} advL2:{:.6f} advLinf:{:.6f} loss1:{:.6f},  loss3:{:.6f}, minlossblack:{:.6f}\n'\
                .format(str(i),len(succ_res[0])>0,query, float(torch.mean(advl2)),float(torch.mean(advlinf)),loss1.data,loss3.data,\
                        torch.min(loss_black).data))

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


        return len(succ_list)/i,np.mean(np.asarray(query_list)),np.median(np.asarray(query_list)),\
            np.mean(np.asarray(all_query_list)),np.median(np.asarray(all_query_list))

@torch.no_grad()
def testTran(npop,target_label,modelchoice,modelpath="",adv_models=None,usedownsample=1,scale=4):

    MSE = nn.MSELoss()  #define mean square error loss
    sigma = 0.1
    sigma_f=0.1 #default 
    lr=0.01 #default
    i = 0
    linf_con = 0.05 # the same as CGATTACK

    model_clean.load_state_dict(torch.load(modelpath)['state_dict_clean'])
    model_adv.load_state_dict(torch.load(modelpath)['state_dict_adv'])
    print(f"{modelpath} loaded")
    model_clean.eval()
    model_adv.eval()
    succ_list=[]
    query_list,l2_list,linf_list,fail_list,query_fail_list,l2_fail_list,linf_fail_list = [],[],[],[],[],[],[]
    all_query_list,all_l2_list,all_linf_list = [],[],[]

    imgbase = '/home_fmg/liujun/dataset/ImageNet_val_mini/randomCropped224/randomCropped224'      
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
        
        batch = batch.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = net(batch*0.5+0.5)
        pre=torch.argmax(output,dim=1)
        if target_label == "least":

            target_label = torch.argsort(output, dim=1)[:, 0]

        elif target_label == "last":

            # Easiest target: class with the highest logit except the predicted class

            sorted_idx = torch.argsort(output, dim=1, descending=True)

            target_label = sorted_idx[:, 1]

        if target_label>=0 and pre==target_label:
            continue
        elif pre != label:
            continue
        with torch.enable_grad():
            if attackType=="FTM":
                batch_surro_adv,_ = advt(batch.cuda()*0.5+0.5,label.cuda(),target_label)
            else:
                batch_surro_adv,_ = augmentationTest(batch.cuda()*0.5+0.5,label.cuda(),target_label,\
                                                    targeted=targeted,attackType=attackType,adv_models=adv_models,change=change)
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

        usedownsample = usedownsample
        scale=scale
        print(f"usedownsample:{usedownsample},scale:{scale}")
        while query<10000:
            if usedownsample==1:
                mu_z,upmuze = upsample(scale=scale,npop=npop)
                modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                batch_perturb = batch_surro_adv.repeat(npop,1,1,1)+ modify
            
            else:
                upmuze = torch.randn((npop,3,224,224)).cuda()
                mu_z = upmuze
                modify = mu.repeat(npop,1,1,1)+sigma_f*upmuze
                batch_perturb = batch_surro_adv.repeat(npop,1,1,1)+ modify        #while query<10:
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


  
        

class augmentationTestEns(nn.Module):
    def __init__(self,targeted=False,attackType="pgd",adv_models=None,change=True):
        super().__init__()
        self.attackType = attackType
        if attackType=="PGN":
            surro_model_idx="all"
            step=5
            print(f"step:{step},surro_model_idx:{surro_model_idx}")
            #adversary = PGN(adv_models[surro_model_idx],eps=0.05)
            self.adversary = PGN(adv_models,eps=0.05,steps=step)

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
        elif self.attackType == "PGN":
            xadv = self.adversary.forward(x, true_lab)
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
    if attackType=="PGN":
        surro_model_idx="all"
        step=5
        print(f"step:{step},surro_model_idx:{surro_model_idx}")
        adversary = PGN(adv_models,eps=0.05,steps=step)

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
    elif attackType == "PGN":
        xadv = adversary.forward(x, true_lab)
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

def cw_loss(logit, label, target=None):
    if target is not None:
        # targeted cw loss: logit_t - max_{i\neq t}logit_i
        _, argsort = logit.sort(dim=1, descending=True)
        target_is_max = argsort[:, 0].eq(target)
        second_max_index = target_is_max.long() * argsort[:, 1] + (1 - target_is_max.long()) * argsort[:, 0]
        target_logit = logit[torch.arange(logit.shape[0]), target]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return target_logit - second_max_logit
    else:
        # untargeted cw loss: max_{i\neq y}logit_i - logit_y
        _, argsort = logit.sort(dim=1, descending=True)
        gt_is_max = argsort[:, 0].eq(label)
        second_max_index = gt_is_max.long() * argsort[:, 1] + (1 - gt_is_max.long()) * argsort[:, 0]
        gt_logit = logit[torch.arange(logit.shape[0]), label]
        second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
        return second_max_logit - gt_logit

def adv_loss( y, label,target=-1,models=None,returnlist=False,margin=None,logits=None):
    if returnlist:
        loss = []
    else:
        loss = 0.
    if margin is None:
        margin=innermargin
    elif margin ==-100:
        if logits is  None:
            for adv_model in models:
        #            loss = 0.
                with torch.no_grad():
                    logits = adv_model(y)

        loss = cw_loss(logit=logits, label=label, target=None)
        return loss

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

def drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5,modelname):
    x = [x for x in range(len(avg_loss))]
    plt.figure()
    plt.plot(x,avg_loss)
    plt.savefig("{}/{}/avg_loss.jpg".format(outputname,modelname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss1)
    plt.savefig("{}/{}/avg_loss1.jpg".format(outputname,modelname))
    plt.close()
    plt.figure()
    plt.plot(x,avg_loss3)
    plt.savefig("{}/{}/avg_loss3.jpg".format(outputname,modelname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss4)
    plt.savefig("{}/{}/avg_loss4.jpg".format(outputname,modelname))
    plt.close()

    plt.figure()
    plt.plot(x,avg_loss5)
    plt.savefig("{}/{}/avg_loss5.jpg".format(outputname,modelname))
    plt.close()

def adv_loss_train( y, label,target=False,randommargin=False):
    loss = 0.
    
    for adv_model in adv_models:
#            loss = 0.
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




def getRandomLabel(label):
    targetlabel = label.detach().clone() 
    while torch.any(targetlabel==label):
        update = len(torch.where(targetlabel==label)[0])
        targetlabel_tmp = torch.randint(0,1000,(update,)).cuda()
        targetlabel[targetlabel==label] = targetlabel_tmp
    return targetlabel



def augmentation(x, true_lab, targeted=False,attackType="pgd"):

    model_idx = np.random.randint(0, len(adv_models))
    model_chosen = adv_models[model_idx]

    adversary = LinfPGDAttack(
        model_chosen, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.05,
        nb_iter=30, eps_iter=2./255, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=targeted)
    
    if   attackType!="pgdMulti":                
        with ctx_noparamgrad_and_eval(model_chosen):
            xadv = adversary.perturb(x, true_lab if targeted else None)
            adv_label = adversary._get_predicted_label(xadv)
            if targeted:
                attack_acc= len(torch.where(adv_label==true_lab)[0])
            else:
                attack_acc= len(torch.where(adv_label!=true_lab)[0])
            #print("augmentation attack acc:{:.4f}".format(attack_acc/len(adv_label)))
            return xadv,adv_label
    elif attackType=="pgdMulti":
        xadv = adversary(x,true_lab)
        return xadv,None
    
   

def train(advType = 'pgd',epoch_num=0,modelname=""):
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

                drawLoss(avg_loss,avg_loss1,avg_loss3,avg_loss4,avg_loss5,modelname)


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
                torch.save(save_dict, '{}/{}/Weight_{}_{}.pth.tar'.format(outputname,modelname,epoch+1,innerloop))

                save_dict = {
                    'epoch': epoch + 1,
                    'loop':innerloop,
                    'batchsize':len(batch),
                    'optimizer_clean' : optimizer_clean.state_dict(),
                    'optimizer_adv' : optimizer_adv.state_dict()    
                }
                torch.save(save_dict, '{}/{}/Opt.pth.tar'.format(outputname,modelname))
                                


           
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



def print_mem(step,device):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"[{step}] allocated={allocated:.2f}MB, reserved={reserved:.2f}MB, max_allocated={max_allocated:.2f}MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', type=str, default='-1 : untargeted, -2: the second largeted predicted label, 0-999: target class labels ', help='')
    args = parser.parse_args()
    print("target_label:{}".format(args.target_label))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setSeed(0)
    outputname='/home_fmg/liujun/weights/difattackPlus'
    torch.set_num_threads(1)
    #Hyperparameters
    para_dict={
    'change':False,
    'num_epochs' :1, 
    'batch_size' :16,
    'learning_rate' : 1e-4,
    'innermargin' :5,
    'advType':"pgd",
    'resume':False,
    'resumePath':"",
    'resumePathOpt':""

    }


    print("para_dict for training ImageNet:{}".format(para_dict))
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
    
    targetmodename='VGG16'#VGG16,Squeezenet,Googlenet,Resnet18,ConvNextBase,EfficientB3,SwinV2T,Resnet101
    mode="test" # train or test
    if mode=="train":
        save_dict = {
        'epoch': 0,
        'state_dict_clean': model_clean.state_dict(),
        'state_dict_adv': model_adv.state_dict(),
        
    }
        modelname = f"difAE_{targetmodename}"
        if targetmodename in refs.keys():
            ref = refs[targetmodename]
            print(ref)
        else:
            raise NotImplementedError        


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

        imagenet_traindata = ImageFolder('/gs/bs/tgh-26IAW/liujun/us06530/dataset/ImageNet2012',transform=img_transform)
        train_loader = torch.utils.data.DataLoader(imagenet_traindata,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=2)


        if not os.path.exists(f'{outputname}/AE_wo_{targetmodename}'):
            os.mkdir(f'{outputname}/AE_wo_{targetmodename}')


        if para_dict['resume']:
            model_clean.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_clean'])
            model_adv.load_state_dict(torch.load(para_dict['resumePath'])['state_dict_adv'])

            #optimizer_clean.load_state_dict(torch.load(para_dict['resumePathOpt'])['optimizer_clean'])
            #optimizer_adv.load_state_dict(torch.load(para_dict['resumePathOpt'])['optimizer_adv'])

        train(advType=para_dict['advType'],num_epochs= num_epochs,modelname=modelname)
    else:

        minq = 10000
        totalq = []
        asrlist = []
        totalmedq = []
        mina=0
        targeted=False # True or False

        openscenario = False
        usePRGFNES = False
        attackType = "None" #FTM for closed-scenario targeted,PGN for closed-scenario untargeted,None for open-scenarios
        if attackType == "None" :
            modelchoice="None"
        else:
            modelchoice="adv"
        usedownsample = 1
        
        scale = 4

        if targetmodename in popdict.keys():
            npop = popdict[targetmodename][1 if targeted else 0]
            modelpath = modelpathdict[targetmodename][1 if targeted else 0]+".pth.tar"
        else:
            raise NotImplementedError
        if targetmodename in refs.keys():
            ref = refs[targetmodename]
            print(ref)
        else:
            raise NotImplementedError        


        if targetmodename == 'wrs50':
            from robustness import model_utils
            from robustness import datasets as datasetsr
            tmpmodelpath = 'datasetweights/microsoft_robust_models/wide_resnet50_2_linf_eps8.0_imgnet.ckpt'

            print(f"{tmpmodelpath} loaded")
            net ,_ = model_utils.make_and_restore_model(arch="wide_resnet50_2",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
            net = MyRobustModel(net).eval().cuda()

        if targetmodename != 'wrs50':
        
            net = utils.load_adv_imagenet([targetmodename],device=device)[0] 
            adv_models= utils.load_adv_imagenet(ref.split(",")) 
        else:
            tmpmodelpath = '/gs/bs/tgh-26IAW/liujun/us06530/datasetweights/microsoft_robust_models/resnet50_linf_eps8.0_imgnet.ckpt'

            print(f"surrogate model:{tmpmodelpath}")
            adv_models ,_ = model_utils.make_and_restore_model(arch="resnet50",dataset=datasetsr.ImageNet(''), resume_path=modelpath, pytorch_pretrained=None,add_custom_forward=False)
            adv_models = [MyRobustModel(adv_models).eval()]
           
        
        change=True
        
        all_avgq_list,all_medq_list = [],[]
        
        target_label = int(args.target_label)
        if target_label == -2:
            target_label = "last"
        elif target_label == -1:
            target_label = -1
        else:
            target_label=torch.tensor([target_label]).cuda() # or other int in [0,999]
        if attackType == "None": # open scenarios
            asri,q,medq,all_avgq,all_medq=test(npop=npop,target_label=target_label,modelchoice=modelchoice,modelpath=modelpath,\
                        usedownsample=usedownsample,scale=scale)
           
        else:
            asri,q,medq,all_avgq,all_medq=testTran(npop=npop,target_label=target_label,modelchoice=modelchoice,modelpath=modelpath,\
                        adv_models=adv_models,usedownsample=usedownsample,scale=scale)
        
