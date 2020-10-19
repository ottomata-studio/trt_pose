#!/usr/bin/env python
# coding: utf-8

# First, let's load the JSON file which describes the human pose task.  This is in COCO format, it is the category descriptor pulled from the annotations file.  We modify the COCO category slightly, to add a neck keypoint.  We will use this task description JSON to create a topology tensor, which is an intermediate data structure that describes the part linkages, as well as which channels in the part affinity field each linkage corresponds to.

# In[1]:


import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)


# Next, we'll load our model.  Each model takes at least two parameters, *cmap_channels* and *paf_channels* corresponding to the number of heatmap channels
# and part affinity field channels.  The number of part affinity field channels is 2x the number of links, because each link has a channel corresponding to the
# x and y direction of the vector field for each link.

# In[2]:


import trt_pose.models

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()


# Next, let's load the model weights.  You will need to download these according to the table in the README.

# In[3]:


import torch

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))


# In order to optimize with TensorRT using the python library *torch2trt* we'll also need to create some example data.  The dimensions
# of this data should match the dimensions that the network was trained with.  Since we're using the resnet18 variant that was trained on
# an input resolution of 224x224, we set the width and height to these dimensions.

# In[4]:


WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()


# Next, we'll use [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) to optimize the model.  We'll enable fp16_mode to allow optimizations to use reduced half precision.

# In[5]:


import torch2trt

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)


# The optimized model may be saved so that we do not need to perform optimization again, we can just load the model.  Please note that TensorRT has device specific optimizations, so you can only use an optimized model on similar platforms.

# In[6]:


OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


# We could then load the saved model using *torch2trt* as follows.

# In[7]:


from torch2trt import TRTModule

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


# We can benchmark the model in FPS with the following code

# In[8]:


import time

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))


# Next, let's define a function that will preprocess the image, which is originally in BGR8 / HWC format.

# In[9]:


import cv2
import torchvision.transforms as transforms
import PIL.Image

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


# Next, we'll define two callable classes that will be used to parse the objects from the neural network, as well as draw the parsed objects on an image.

# In[10]:


from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


# Assuming you're using NVIDIA Jetson, you can use the [jetcam](https://github.com/NVIDIA-AI-IOT/jetcam) package to create an easy to use camera that will produce images in BGR8/HWC format.
# 
# If you're not on Jetson, you may need to adapt the code below.

# Finally, we'll define the main execution loop.  This will perform the following steps
# 
# 1.  Preprocess the camera image
# 2.  Execute the neural network
# 3.  Parse the objects from the neural network output
# 4.  Draw the objects onto the camera image
# 5.  Convert the image to JPEG format and stream to the display widget

# In[11]:


cap = cv2.VideoCapture(0)


# Next, we'll create a widget which will be used to display the camera feed with visualizations.

# In[15]:


import ipywidgets
from IPython.display import display

image_w = ipywidgets.Image(format='jpeg')
display(image_w)


# In[89]:


#from jetcam.utils import bgr8_to_jpeg

global start_time
def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    image = cv2.resize(image,(640,640))
    image = cv2.flip(image, 1)
    image = cv2.putText(image, "FPS: " + str(int(1.0 / (time.time() - start_time))), (500, 50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 0, 0), 2, cv2.LINE_AA)
    #image_w.value = bgr8_to_jpeg(image)
    image_w.value = bytes(cv2.imencode('.jpg', image)[1])



# If we call the cell below it will execute the function once on the current camera frame.

# In[90]:


while True:
    start_time = time.time() # start time of the loop
    ret,frame = cap.read()
    frame = cv2.resize(frame,(224,224))
    execute(frame)
    frame = cv2.resize(frame,(640,640))
    cv2.imshow("test",frame)


cap.release()
cv2.destroyAllWindows()


# Call the cell below to attach the execution function to the camera's internal value.  This will cause the execute function to be called whenever a new camera frame is received.

# In[ ]:





# Call the cell below to unattach the camera frame callbacks.

# In[1]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




