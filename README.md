## Training a DCGAN
GAN stands for Generative Adversarial Network, which is a type of 
deep learning model used for generating synthetic data such as 
images. <br>
The model is composed of two neural networks: a generator and a 
discriminator. <br>
 The generator network takes a random noise input and generates 
a synthetic sample that tries to mimic the real data. <br>
 The discriminator network takes both real and synthetic data and 
tries to distinguish between them. <br>
DCGAN stands for Deep Convolutional Generative Adversarial 
Network. It is a type of Generative Adversarial Network (GAN) that uses 
convolutional neural networks (CNNs) for both the generator and 
discriminator networks. <br>
DCGANs have several advantages over traditional GANs. <br>
 CNNs in both the generator and discriminator networks allows for 
better image synthesis and sharper images. <br>
 The use of convolutional layers helps to capture spatial 
correlations in the data.<br>
### Choice of Hyperparameters:
 ndf=32 & ngf=32: it represents the number of filters in the first 
layer of the discriminator and generator respectively. A smaller 
value for ndf can help the discriminator to learn more 
generalized features <br>
 nz=100: it represents the size of the input noise vector that is fed 
into the generator. A larger noise vector can help the generator to 
learn more diverse and complex patterns <br>
 d_lr=0.0002 & g_lr=0.0002: They represent the learning rates for 
the discriminator and generator, respectively. A lower learning 
rate can help to stabilize the training process and prevent the 
model from diverging. <br>
 use_fixed=True: This parameter determines whether to use a 
fixed input noise vector during testing. Using a fixed noise vector 
can help to generate consistent images across multiple runs of 
the model. <br>
 Loss= BCELoss():BCELoss (Binary Cross Entropy Loss) is wellsuited for binary classification tasks, such as distinguishing 
between real and fake images. The BCELoss function measures 
the binary cross-entropy between the discriminator's output and 
the ground truth label (0 for fake images and 1 for real images). <br>

## StyleGAN
StyleGAN is an extension of the GAN architecture introduced by Nvidia
researchers in December 2018. <br>
It is designed to generate high-quality and realistic images, such as 
human faces, by synthesizing artificial samples. <br>
When generating faces from a GAN, we typically start by sampling a 
random vector (also known as a seed or a noise vector) from a 
standard normal distribution. <br>
This random vector is then passed through the generator, which 
produces an output image that corresponds to the input vector. <br>
To control the style or features of the generated faces, researchers 
often use techniques such as interpolation, truncation, or manipulation 
of the latent space. <br>
By interpolating between two points in the latent space, we can 
generate a series of images that smoothly transition from one style to 
another <br>
To do this, we can use a linear interpolation technique, where we 
take a weighted average of the two seed vectors to generate a 
new intermediate vector that lies somewhere between the two. ****
zinter = (1 - α) * z1 + α * z2 <br>

## References:

'''
REFEREMCES:

for QUESTION 1:

https://www.youtube.com/watch?v=QE-AfZLVX2w
https://mafda.medium.com/gans-deep-convolutional-gans-with-mnist-part-3-8bad9a96ff65
https://stackoverflow.com/questions/68243122/runtimeerror-given-groups-1-weight-of-size-64-32-3-3-expected-input128
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/mnist/01_GAN_MNIST.ipynb
https://github.com/coreprinciple6/GAN-Study/tree/master/DCGAN
https://github.com/AKASHKADEL/dcgan-mnist/tree/master/


for QUESTION 2:

@inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik H\"ark\"onen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
}


@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}


https://github.com/jeffheaton/stylegan2-toys/tree/main
https://github.com/jeffheaton/stylegan2-toys/blob/main/morph_video_real.ipynb
https://github.com/justinpinkney/stylegan-matlab-playground/blob/master/examples/interpolation.md
https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_07_1_gan_intro.ipynb
https://blog.paperspace.com/how-to-set-up-stylegan2-ada-pytorch-on-paperspace/
https://github.com/NVlabs/stylegan2-ada-pytorch#pretrained-networks
https://discuss.pystorch.org/t/running-a-pre-trained-tensorflow-stylegan-model-in-pytorch/42001
https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb


**ADDITIONAL RESOURCES USED:** <br>

perplexity.ai
chat.openai.com
bing.com

'''
