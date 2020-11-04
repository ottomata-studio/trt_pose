import trt_pose.models
import json
import trt_pose.coco

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# Uncomment based on which model you have

# for resnet18_baseline_att_224x224_A_epoch_249
# model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()

# for densenet121_baseline_att_256x256_B_epoch_160
model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links, pretrained=False).cuda().eval()

# MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
# ONNX_OUTPUT = 'resnet18_baseline_att_224x224_A_epoch_249.onnx'

MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
ONNX_OUTPUT = 'densenet121_baseline_att_256x256_B_epoch_160.onnx'

model.load_state_dict(torch.load(MODEL_WEIGHTS))

# WIDTH = 224 # for resnet18_baseline_att_224x224_A_epoch_249
# HEIGHT = 224 # for resnet18_baseline_att_224x224_A_epoch_249

WIDTH = 256 # for densenet121_baseline_att_256x256_B_epoch_160
HEIGHT = 256 # for densenet121_baseline_att_256x256_B_epoch_160

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

torch_out = model(data)

# export the model
torch.onnx.export(model,
                  data,
                  ONNX_OUTPUT,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True
)

import onnx

# load our newly converted ONNX model
onnx_model = onnx.load(ONNX_OUTPUT)

# check our newly converted ONNX model for any errors
onnx.checker.check_model(onnx_model)

# Print a human readable representation of the model graph
print(onnx.helper.printable_graph(onnx_model.graph))
