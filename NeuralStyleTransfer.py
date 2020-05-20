import os
from torch.autograd import Variable
from torch import optim
from PIL import Image
from helpers import *
from time import perf_counter

image_dir = os.getcwd() + '/Images/'
model_dir = os.getcwd() + '/Models/'

img_size = 512
max_iter = 200
show_iter = 50

# Specify the default arguments for the style transfer function
style_weights = [1e3/n**2 for n in [64,128,256,512,512]]
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']

content_weights = [1]
content_layers = ['r42']
similarity_layer = ['r42']

content_image_name='content_image.jpg'
style_image_name = 'style_image.jpg'

output_dir = 'Images' # start without / and end without /

cnt = 1


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


#get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg.cuda()



def run_neural_style_transfer(content_image_name=content_image_name, style_image_name=style_image_name, content_layers=content_layers, content_weights=content_weights, style_layers=style_layers,
                                style_weights=style_weights, max_iter=max_iter, show_iter=show_iter, swap_content_style=False,
                              add_index=False, output_dir=output_dir):

    global cnt
    # load images, ordered as [style_image, content_image]
    img_dirs = [image_dir, image_dir]
    if swap_content_style:
        img_names = [content_image_name, style_image_name]
    else:
        img_names = [style_image_name, content_image_name]
    imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
    imgs_torch = [prep(img) for img in imgs]
    if torch.cuda.is_available():
        imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
    else:
        imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
    style_image, content_image = imgs_torch

    # opt_img = Variable(torch.randn(content_image.size()).type_as(content_image.data), requires_grad=True) #random init
    opt_img = Variable(content_image.data.clone(), requires_grad=True)

    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
    if torch.cuda.is_available():
        loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]

    weights = style_weights + content_weights

    #compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    #run style transfer
    if add_index:
        print("Running neural style transfer %d on " % cnt, os.uname()[1])
    else:
        print("Running neural style transfer on ", os.uname()[1])
    print("Image size = %d" % img_size)
    print("Max number of iterations = %d" % max_iter)
    print("Show result every %d iterations" % show_iter)
    print("Content layer(s):", content_layers)
    print("Content weight(s):", content_weights)
    print("Style layer(s):", style_layers)
    print("Style weight(s):", style_weights)
    print("\n\n")


    optimizer = optim.LBFGS([opt_img]);

    n_iter = [0]

    t0 = perf_counter()

    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0]+=1
            #print loss
            if n_iter[0]%show_iter == (show_iter-1):
                print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.data.item()))
            return loss

        optimizer.step(closure)

    t1 = perf_counter()

    print("Total execution time: %f" % (t1 - t0))
    print("===========================================================================================")
    print("\n\n")

    out_img = postp(opt_img.data[0].cpu().squeeze())

    if add_index:
        out_img_path = "%s/nst_stylized_image%d.jpg" % (output_dir, cnt)
        cnt += 1
    else:
        out_img_path = "%s/nst_stylized_image.jpg" % output_dir

    out_img.save(out_img_path)
    return out_img_path

