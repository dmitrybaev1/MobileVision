import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.alexnet import alexnet
import time
import os

print("Process Id:", os.getpid())
start_loading = time.time()
model = alexnet(pretrained=True).eval().cuda()
print("Loading time:", time.time() - start_loading)

input_image = Image.open("./images/dog.jpg")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

input_batch = input_batch.to('cuda')

start_execution = time.time()
with torch.no_grad():
    output = model(input_batch)
print("Execution time:", time.time() - start_execution)
print("---------")

print(output)
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())