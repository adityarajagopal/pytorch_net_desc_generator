import os 
import sys
import subprocess
import configparser as cp

import torch 
import torch.nn as nn

import imagenet_models as models

class NetDescGenerator(object): 
#{{{
    def __init__(self): 
        self.inputSizes = {}
        self.outputSizes = {}
        self.hash2LName = {}
        
    def forward_hook(self, module, input, output): 
        mHash = module.__hash__()
        lName = self.hash2LName[mHash]
        self.inputSizes[lName] = input[0].shape 
        self.outputSizes[lName] = output.shape

    def from_model(self, model, dataset, modelConfig): 
    #{{{
        """ Generates a layer by layer test given an input model by identifying conv and fc layers """
        hooks = []
        for n,m in model.named_modules(): 
            if isinstance(m, nn.Conv2d):
                mHash = m.__hash__()
                self.inputSizes[n] = None
                self.outputSizes[n] = None
                self.hash2LName[mHash] = n
                hooks.append(m.register_forward_hook(self.forward_hook))
        
        batchSize = 1
        ipImgSize = [224,224]
        dummyIp = torch.FloatTensor(batchSize, 3, *ipImgSize)
        
        model(dummyIp)        
        
        [h.remove() for h in hooks]

        configOpFile = modelConfig 
        allTests = []
        for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                keys = {'ic':'in_channels', 'oc':'out_channels', 'k':'kernel_size', 'pad':'padding',\
                        'stride':'stride', 'groups':'groups'}
                test = [f"[{n}]"]  
                test.append("bs={}".format(batchSize))
                test.append("bias={}".format(m.bias is not None))
                ipSize = list(self.inputSizes[n])[2:]
                test.append("ip={}".format(ipSize))
                opSize = list(self.outputSizes[n])[2:]
                test.append("op={}".format(opSize))
                for k,v in keys.items():
                    test.append("{}={}".format(k,eval("m.{}".format(v))))
                allTests.append('\n'.join(test))
           
        dirName = os.path.dirname(configOpFile)
        cmd = f"mkdir -p {dirName}"
        subprocess.check_call(cmd, shell=True)
        with open(configOpFile, 'w') as f:
            f.write('\n\n'.join(allTests))
    #}}}
#}}}

def read_model(arch, depth=None):
    if 'resnet' in arch:
        return models.__dict__[arch](num_classes=1000, depth=depth, pretrained=False)
    else:
        return models.__dict__[arch](num_classes=1000, pretrained=False)

def generate_network_desc(modelDict, saveDir):
    model = read_model(modelDict['arch'], modelDict['depth'])

    layerDescFile = os.path.join(saveDir, f"{modelDict['arch']}.ini")
    layerDescGen = NetDescGenerator()
    layerDescGen.from_model(model, 'imagenet', layerDescFile)
    return layerDescFile

if __name__ == '__main__':
    modelDict = {'arch': 'googlenet', 'depth': None}
    saveDir = '/home/ar4414/extras/net_gen/desc_files'
    descFile = generate_network_desc(modelDict, saveDir)
    desc = cp.ConfigParser()
    desc.read(descFile)
    for name in desc.sections():
        print(name)
        print(dict(desc[name]))

