import os
image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from PIL import Image
from helpers import *
import pytorch_msssim

#from matplotlib.pyplot import imshow, show, gcf

# In[3]:

img_size = 512
max_iter = 200
show_iter = 50
save_iter = 50

# In[4]:


# pre and post processing for images

prep = transforms.Compose([transforms.Resize(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])

def postp(tensor):  # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img

# In[5]:

# get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()


# In[6]:


# load images, ordered as [style_image, content_image]
img_dirs = [image_dir, image_dir]
img_names = ['style_image.jpg', 'content_image.jpg']
imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]
if torch.cuda.is_available():
    imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
else:
    imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
style_image, content_image = imgs_torch


# opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
# opt_img = Variable(content_image.data.clone(), requires_grad=True)

# here I initialized my images
stylized_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
reversed_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init

# In[7]:


#display images
#for img in imgs:
#    imshow(img);show()


# In[8]:


# define layers, loss functions, weights and compute optimization targets
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
"""GramMSELoss() is a "neural network" that takes input and target and computes the MSELoss between Gram Matrix of
input and target feature map. So loss_fns declared in the above line is a "neural network" """
""" loss_fns is a "neural network" composed of 5 GramMSELoss() "neural networks" and one MSELoss "neural network". Its inputs 
are the input and the target layers, outputs are the feature maps needed for the loss functions """
if torch.cuda.is_available():
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
    
# these are good weights settings:
"""You can define different style weights for different style layers"""
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
content_weights = [1e0]
similarity_weight = 100000 # added a weight for the similarity loss
weights = style_weights + content_weights

# compute optimization targets
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets


# In[9]:

# run style transfer
"""Optimizer is defined on opt_image, i.e. the pixels of opt_image are the "parameters" of the network that needs to 
be optimized through LBFGS """
optimizer_stylization = optim.LBFGS([stylized_img])
optimizer_reverse = optim.LBFGS([reversed_img])

n_iter=[0]
checkpoint = 1

print("Running style transfer ...")
print("Image size = %d" % img_size)
print("Max number of iterations = %d" % max_iter)
print("Chekpoint every %d iterations" % save_iter)
print("===========================================================================================")

while n_iter[0] <= max_iter:
    """Here I have to redefine a new optimization process, i.e. make clear what my goal is"""

    def closure_stylization():
        optimizer_stylization.zero_grad()
        """Outputs the feature maps for the loss_layers=content_layers+style_layers:"""
        out = vgg(stylized_img, loss_layers)
        """Calculates the loss for each layer:"""
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        """You take the first loss function, multiply it by its weight, its inputs are A (the image that is going to be optimized)
        and targets[a] (the feature map that we want to specify in our loss)"""
        """Total loss is the sum of the loss of each layer:"""
        #loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_image, reversed_img) # added term for similarity loss
        loss = sum(layer_losses) + similarity_weight * pytorch_msssim.msssim(content_image, reversed_img) # added term for similarity loss

        loss.backward()

        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Stylization loss     - Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.data.item()))
            # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss

    def closure_reverse():
        optimizer_reverse.zero_grad()
        """Outputs the feature maps for the loss_layers=content_layers+style_layers:"""
        out = vgg(reversed_img, loss_layers)
        """Calculates the loss for each layer:"""
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        """You take the first loss function, multiply it by its weight, its inputs are A (the image that is going to be optimized)
        and targets[a] (the feature map that we want to specify in our loss)"""
        """Total loss is the sum of the loss of each layer:"""
        #loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_image, reversed_img) # added term for similarity loss
        loss = sum(layer_losses) + similarity_weight * pytorch_msssim.msssim(content_image, reversed_img) # added term for similarity loss

        loss.backward()


        #print loss
        if n_iter[0] % show_iter == (show_iter-1):
            print('Reverse Loss         - Iteration: %d, loss: %f'%(n_iter[0] + 1, loss.data.item()))
            # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
        return loss

    optimizer_stylization.step(closure=closure_stylization)
    optimizer_reverse.step(closure=closure_reverse)

    if n_iter[0] % save_iter == (save_iter - 1):
        st = postp(stylized_img.data[0].cpu().squeeze())
        st.save("Images/stylized_image_check%d.jpg" % checkpoint)
        rv = postp(reversed_img.data[0].cpu().squeeze())
        rv.save("Images/reversed_image_check%d.jpg" % checkpoint)
        checkpoint += 1

    n_iter[0] += 1

# display result
stylized_img = postp(stylized_img.data[0].cpu().squeeze())
reversed_img = postp(reversed_img.data[0].cpu().squeeze())
# imshow(out_img)
# show()

stylized_img.save("Images/stylized_image.jpg")
reversed_img.save("Images/reversed_image.jpg")
# gcf().set_size_inches(10, 10)


#Controlling perceptual factors part removed
