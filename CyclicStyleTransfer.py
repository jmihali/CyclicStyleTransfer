import os
image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'
from torch.autograd import Variable
from torch import optim
from PIL import Image
from helpers import *
import pytorch_msssim
from time import perf_counter

import torch
from PerceptualSimilarity.util import util
import PerceptualSimilarity.models as models

img_size = 512
max_iter = 200
show_iter = 50

#similarity_type = 'lpsis'
#similarity_type = 'content'
#similarity_type = 'msssim'
similarity_type = 'mse'
similarity_weight = 1 # added a weight for the similarity loss

style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

content_weights = [1]
content_layers = ['r42']
similarity_layer = ['r42']

content_image_name='content_image.jpg'
style_image_name = 'style_image.jpg'

output_dir = 'Images' # start without / and end without /

cnt = 1

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


# get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()


# run style transfer
def run_cyclic_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name, content_layers=content_layers, content_weights=content_weights, style_layers=style_layers,
                                style_weights=style_weights, similarity_type=similarity_type, similarity_layer=similarity_layer,
                                similarity_weight=similarity_weight, max_iter=max_iter, show_iter=show_iter, swap_content_style=False,
                              add_index=False, output_dir=output_dir):

    global cnt
    # load images, ordered as [style_image, content_image]
    img_dirs = [image_dir, image_dir]
    if swap_content_style:
        img_names = [content_image_name, style_image_name]
    else:
        img_names = [style_image_name, content_image_name]
    imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    """GramMSELoss() is a "neural network" that takes input and target and computes the MSELoss between Gram Matrix of
    input and target feature map. So loss_fns declared in the above line is a "neural network" """
    """ loss_fns is a "neural network" composed of 5 GramMSELoss() "neural networks" and one MSELoss "neural network". Its inputs 
    are the input and the target layers, outputs are the feature maps needed for the loss functions """
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    weights = style_weights + content_weights

    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    # initialize images:

    # stylized_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
    # reversed_img = Variable(content_image.data.clone(), requires_grad=True)

    stylized_img = Variable(torch.randn(content_image.size()).type_as(content_image.data),
                            requires_grad=True)  # random init
    reversed_img = Variable(torch.randn(content_image.size()).type_as(content_image.data),
                            requires_grad=True)  # random init

    """Optimizer is defined on opt_image, i.e. the pixels of opt_image are the "parameters" of the network that needs to 
    be optimized through LBFGS """
    optimizer_stylization = optim.LBFGS([stylized_img])
    optimizer_reverse = optim.LBFGS([reversed_img])

    if (similarity_type not in ('mse', 'mssim', 'content', 'lpsis')):
        print("Chosen similarity type does not exist. Automatically set to mse")
        similarity_type = 'mse'

    if similarity_type == 'lpsis':
        lpsis_model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available(),
                                            spatial=False)

    n_iter = [0]
    checkpoint = 1

    if add_index:
        print("Running cyclic style transfer %d on " % cnt, os.uname()[1])
    else:
        print("Running cyclic style transfer on ", os.uname()[1])
    print("Content image name:", content_image_name)
    print("Style image name:", style_image_name)
    print("Image size = %d" % img_size)
    print("Max number of iterations = %d" % max_iter)
    print("Show result every %d iterations" % show_iter)
    print("Content layer(s):", content_layers)
    print("Content weight(s):", content_weights)
    print("Style layer(s):", style_layers)
    print("Style weight(s):", style_weights)
    print("Similarity weight:", similarity_weight)
    print("Similarity loss type: ", similarity_type)
    print("\n\n")

    t0 = perf_counter()

    while n_iter[0] <= 2*max_iter:

        def closure_stylization():
            optimizer_stylization.zero_grad()
            """Outputs the feature maps for the loss_layers=content_layers+style_layers:"""
            out = vgg(stylized_img, loss_layers)
            """Calculates the loss for each layer:"""
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
            """You take the first loss function, multiply it by its weight, its inputs are A (the image that is going to be optimized)
            and targets[a] (the feature map that we want to specify in our loss)"""
            """Total loss is the sum of the loss of each layer:"""

            if similarity_type == 'mse':
                loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_image, reversed_img) # added term for similarity loss
            elif similarity_type == 'mssim':
                loss = sum(layer_losses) + similarity_weight * pytorch_msssim.msssim(content_image, reversed_img, normalize=True) # added term for similarity loss
            elif similarity_type == 'content':
                layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                # extract high level feature maps for the content image and the reversed image
                content_fm = vgg(content_image, similarity_layer)[0]
                reversed_fm = vgg(reversed_img, similarity_layer)[0]
                loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_fm, reversed_fm) # added term for high level content similarity loss

            elif similarity_type == 'lpsis':
                sim_loss = lpsis_model.forward(stylized_img, reversed_img)
                loss = sum(layer_losses) + similarity_weight * sim_loss # added term for similarity loss


            loss.backward()
            n_iter[0] += 1

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

            if similarity_type == 'mse':
                loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_image, reversed_img) # added term for similarity loss
            elif similarity_type == 'mssim':
                loss = sum(layer_losses) + similarity_weight * pytorch_msssim.msssim(content_image, reversed_img, normalize=True) # added term for similarity loss
            elif similarity_type == 'content':
                # extract high level feature maps for the content image and the reversed image
                content_fm = vgg(content_image, similarity_layer)[0]
                reversed_fm = vgg(reversed_img, similarity_layer)[0]
                loss = sum(layer_losses) + similarity_weight * nn.MSELoss()(content_fm, reversed_fm) # added term for high level content similarity loss

            elif similarity_type == 'lpsis':
                sim_loss = lpsis_model.forward(stylized_img, reversed_img)
                loss = sum(layer_losses) + similarity_weight * sim_loss # added term for similarity loss


            loss.backward()
            n_iter[0] += 1

            #print loss
            if n_iter[0] % show_iter == (show_iter-1):
                print('Reverse Loss         - Iteration: %d, loss: %f'%(n_iter[0] + 1, loss.data.item()))
                # print([loss_layers[li] + ': ' +  str(l.data[0]) for li,l in enumerate(layer_losses)]) #loss of each layer
            return loss

        optimizer_stylization.step(closure=closure_stylization)
        optimizer_reverse.step(closure=closure_reverse)

    t1 = perf_counter()

    stylized_img = postp(stylized_img.data[0].cpu().squeeze())
    reversed_img = postp(reversed_img.data[0].cpu().squeeze())
    print("Total execution time: %f" % (t1 - t0))
    print("===========================================================================================")
    print("\n\n")

    if add_index:
        stylized_img.save("%s/cst_stylized_image%d.jpg" % (output_dir, cnt))
        reversed_img.save("%s/cst_reversed_image%d.jpg" % (output_dir, cnt))
        cnt += 1
    else:
        stylized_img.save("%s/cst_stylized_image.jpg" % output_dir)
        reversed_img.save("%s/cst_reversed_image.jpg" % output_dir)