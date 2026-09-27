refs={
  "VGG16":"SqueezeNet,GoogleNet,ResNet18",
    "SqueezeNet":"VGG16,GoogleNet,ResNet18",
    "GoogleNet":"SqueezeNet,VGG16,ResNet18",
    "ResNet18":"SqueezeNet,GoogleNet,VGG16",
    "SwinV2T":"ConvNextBase,EfficientB3,ResNet101",
    "ConvNextBase":"SwinV2T,EfficientB3,ResNet101",
    "EfficientB3":"ConvNextBase,SwinV2T,ResNet101",
    "ResNet101":"ConvNextBase,EfficientB3,SwinV2T", 
}

#for difattackplus openset
#npop for untargeted,targeted except for wrs50
popdict={
    "VGG16":[5,8],
    "SqueezeNet":[5,10],
    "GoogleNet":[5,10],
    "ResNet18":[5,8],
    "SwinV2T":[8,10], 
    "EfficientB3":[5,15],
    "ResNet101":[10,15],
    "ConvNextBase":[8,15], 
    "wrs50":[15,15] #both 15 for open and closed-scenarios
}

modelpathdict={
    "VGG16":["VGG16_untarget","VGG16_targeted"],
    "SqueezeNet":["SqueezeNet_untarget","SqueezeNet_targeted"],
    "GoogleNet":["GoogleNet_untarget","GoogleNet_targeted"],
    "ResNet18":["ResNet18","ResNet18"],
    "SwinV2T":["SwinV2T_untarget","SwinV2T_targeted"], 
    "EfficientB3":["EfficientB3_untarget","EfficientB3_targeted"],
    "ResNet101":["ResNet101_untarget","ResNet101_targeted"],
    "ConvNextBase":["ConvNextBase_untarget","ConvNextBase_targeted"], 
    "wrs50":["wrs50_untarget","wrs50_targeted"]
}