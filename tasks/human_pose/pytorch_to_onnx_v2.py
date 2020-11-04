#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

# In[2]:


import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)


# Next, we'll load our model.  Each model takes at least two parameters, *cmap_channels* and *paf_channels* corresponding to the number of heatmap channels
# and part affinity field channels.  The number of part affinity field channels is 2x the number of links, because each link has a channel corresponding to the
# x and y direction of the vector field for each link.

# In[3]:


import trt_pose.models

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()
#model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()


# Next, let's load the model weights.  You will need to download these according to the table in the README.

# In[4]:


import torch

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
ONNX_OUTPUT = 'resnet18_baseline_att_224x224_A_epoch_249.onnx'
#MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
#ONNX_OUTPUT = 'densenet121_baseline_att_256x256_B_epoch_160.onnx'

model.load_state_dict(torch.load(MODEL_WEIGHTS))


# Convert a pytorch model to ONNX format

# In[5]:


WIDTH = 224
HEIGHT = 224
#WIDTH = 256
#HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

torch_out = model(data)
torch.onnx.export(model,
                  data,
                  ONNX_OUTPUT,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True
)


# Verify converted ONNX model

# In[6]:


import onnx

onnx_model = onnx.load(ONNX_OUTPUT)
onnx.checker.check_model(onnx_model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(onnx_model.graph))


# In[ ]:




