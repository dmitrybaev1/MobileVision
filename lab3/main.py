from __future__ import print_function

import torch
import matplotlib.pyplot as plt
import torchvision.models as models

from neuralstyle import run_style_transfer, image_loader, imshow, imsave

device = torch.device("cuda")

style_img = image_loader("./images/wood.jpg",device)
content_img = image_loader("./images/burger.jpg",device)
input_img = content_img.clone()

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img, device)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
imsave(output)