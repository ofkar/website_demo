import torch
from torchvision import transforms

from modules.unet import UNet, UNetReshade
from modules.percep_nets import Dense1by1Net

import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb
from models import DataParallelModel


parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()


root_dir = '/workspace/demo-code/code/models/'
trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.NEAREST),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()


# get target task and model
target_tasks_raw = ['rgb2sfnorm','rgb2depth','reshade', 'rgb2sfnormb','rgb2depthb','reshadeb']
target_tasks = ['normal','depth','reshading', 'normal','depth','reshading']

percep_tasks = ['curvature', 'edge2d', 'edge3d', 'keypoint2d', 'keypoint3d']

try:
    task_index = target_tasks_raw.index(args.task)
except:
    print("task should be one of the following: normal, depth, reshading")
    sys.exit()
models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5), UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]
percep_models =[ Dense1by1Net(), UNet(out_channels=1, downsample=4), UNet(downsample=5, out_channels=1),  UNet(downsample=5, out_channels=1), UNet(downsample=5, out_channels=1)] 
model = models[task_index]
task_name = target_tasks[task_index]

def save_outputs(img_path, output_file_name):
    
    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0)
    print(task_name, img_tensor.size())

    # compute baseline and consistency output
    #for type in ['consistency']:
    #pdb.set_trace()
    if task_index<3: 
        type='consistency' 
    else: 
        type='baseline'
    path = root_dir + 'rgb2'+task_name+'_'+type+'.pth'
    model_state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(model_state_dict)
    baseline_output = model(img_tensor).clamp(min=0, max=1)
    trans_topil(baseline_output[0]).save(args.output_path+'/'+target_tasks_raw[task_index]+'.png')
   
    if task_name is 'normal': #obtain percep outputs
        print("Obtainin perceps!")
        input_normals = baseline_output[0].unsqueeze(0)
        for i in range(len(percep_tasks)):
            #pdb.set_trace()
            percep_model = DataParallelModel(percep_models[i])
            path_percep = root_dir + 'normal2' + percep_tasks[i] + '.pth'
            percep_model_state_dict = torch.load(path_percep, map_location=torch.device('cpu'))
            percep_model.load_state_dict(percep_model_state_dict)
            print("Input size:" , input_normals.size())
            percep_output = percep_model(input_normals).clamp(min=0, max=1)
            trans_topil(percep_output[0]).save(args.output_path+'/'+ percep_tasks[i] + '_' + type + '.png')
            print("Percep saved", args.output_path+'/'+ percep_tasks[i] + '_' + type + '.png')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
