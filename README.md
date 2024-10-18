# WGAN-Image-Generation
Applying DCGAN, WGAN (with weight clipping), and WGAN-GP (with gradient penalty) to FashionMNIST

# Overview
A reimplementation of two distinct Generative Adversarial Network (GAN) architectures using various techniques described in **"Improved Training of Wasserstein GANs", by Gulrajani et al. (2017)**. We add to this paper by testing two architectures with three techniques each (DCGAN, WGAN (weight clipping), and WGAN (gradient penalty)). I work with the FashionMNIST dataset to generate images with generator networks after training is complete.

This project includes a Jupyter Notebook file which creates a training regime for evaluating different models and techniques applied to GANs that generate images based on the FashionMNIST dataset. Specifically, I chose 2 architectures and trained a Deep-Convolutional GAN (DCGAN), a Wasserstein GAN with weight clipping, and a Wasserstein GAN with gradient penalty applied. 

# Dependencies
All dependencies are imported within the markdown file. Simply run the file to import the correct packages. It is recommended to connect to an NVIDIA T4 GPU for training these models. Note that the folder includes model weight paths that can be imported for generation. Please use Python 3.10+


# How to Train the GANs
In order to run our cells for training, it is required that all importing and data processing cells are run and the necessary pre-train tasks are complete. These pre-train tasks include initializing the criterion, fixed noise vectors, the save path for the generator weights, the lists to store the images generated (for analysis), and the generator and discriminator loss lists. 

To train the GANs now, simply run the designated training block within the section that you want to train (i.e. Architecture I, WGAN (with weight clipping). Following the training block, there are some visualizations set up in subsequent blocks to show the generator / discriminator loss, a comparison between real and generated images (with weights from final epoch), and a training walkthrough animation to show the generation progress of the generator at each step in the training. 


# How to Generate with GANs
The exciting part is using the trained generator networks to generate new images. In order to make it easy on the user of this repository, the .pth files that include each generator’s trained weights are included in the primary folder and can be imported for easy generation with the model and architecture of your choice. 

For loading weights and generating images with one of the GANs, use the code below for guidance. 

 1. Instantiate the desired generator model (note, the generator model nn.module class and its dependencies must be instantiated prior)
 2. Provide the path to the desired model weight file 
 3. Give the noise vector size (100)
 4. Provide the device
 5. Run the generate_and_show_image function with appropriate arguments
    
<img width="819" alt="Screenshot 2024-10-16 at 12 22 30 AM" src="https://github.com/user-attachments/assets/e9c3e447-029b-4396-b6b1-9a8836936a49">

<img width="385" alt="Screenshot 2024-10-16 at 12 22 47 AM" src="https://github.com/user-attachments/assets/8b493507-c09e-46db-854e-ac91b991326d">

# Architectures

### Architecture 1
- **Nonlinearity (G)**: ReLU
- **Nonlinearity (D)**: LeakyReLU
- **Depth (G)**: 4
- **Depth (D)**: 4
- **Batch normalization (G)**: True
- **Batch normalization (D)** (Layer norm for WGAN-GP): True
- **Base Filter Count (G)**: 64
- **Base Filter Count (D)**: 64

### Architecture 2
- **Nonlinearity (G)**: ReLU
- **Nonlinearity (D)**: LeakyReLU
- **Depth (G)**: 4
- **Depth (D)**: 4
- **Batch normalization (G)**: True
- **Batch normalization (D)** (Layer norm for WGAN-GP): True
- **Base Filter Count (G)**: 32
- **Base Filter Count (D)**: 32


Architecture 1 produced better images for each model when compared to architecture
2. This is likely the result of the higher capacity to learn detailed features, which is afforded
by the larger base count filter for both the generator and the discriminator (64 vs 32). The
discriminator is likely a bit better at differentiating real from fake images by capturing
richer relationships in the data. In turn, this forces the GAN to perform better as well.
The GAN would be provided a better gradient signal in this case, creating better-quality
images.

Ultimately, the best performing model and architecture combination is the WGAN-GP
(with gradient penalty) on architecture 1 (with a larger base filter count). This genera-
tor was able to produce very high-quality images (based on the provided FashionMNIST
dataset) in comparison to the other generators that produced fuzzier images.
The WGAN-GP enforces the Lipschitz constraint with the gradient penalty during train-
ing. This ensures a smooth gradient flow and therefore improved training. The gradient
penalty specifically makes sure that the critic / discriminator doesn’t give erratic gradients,
allowing the generator to better learn crisp outputs based on the training data.

# Key Functions and Classes

## 'weights_init'
The 'weights_init' function is called to initialize the weights of the layers in our networks prior to training. This function initializes convolutional layers and batch normalization layers properly. This ensures that the network begins with appropriate parameters for stable learning.

## ‘Generator’
The ‘Generator’ class defines a python nn.module for the DCGAN generator. As input, the model takes a random noise vector. The model includes 4 convolutional layers, each with batchnorm and ReLU, followed by a Tanh layer at the end. The output is an image of size 64X64 with a variable number of channels (for RGB if necessary). 

## ‘Generator_WGAN’
The ‘Generator_WGAN’ class defines a python nn.module for the WGAN generators.This is functionally the same as the ‘Generator’ class. 

## ‘Critic_WGAN’
The ‘Critic_WGAN’ class defines a python nn.module for the WGAN with weight clipping critic / discriminator. The ‘Critic_WGAN’ is functionally very similar to the ‘Discriminator’ class with, but there is one key difference. The critic outputs a raw scalar value (without sigmoid) which makes it suitable for the Wasserstein approach. 

## ‘Critic_WGAN_GP’
The ‘Critic_WGAN_GP’ class defines a python nn.module for the WGAN with gradient penalty critic / discriminator. The ‘Critic_WGAN_GP’ is functionally very similar to the ‘Critic_WGAN’ class with, but it uses layernorm (more below) rather than batchnorm for stabilization. The critic also outputs a raw scalar value (without sigmoid) which makes it suitable for the Wasserstein approach. 

## 'Discriminator'
The Discriminator class defines a python nn.module for the DCGAN discriminator. As input, the model takes an image (64 x 64 x number of channels). The model includes 4 convolutional layers, each with stride 2, reducing the size of the image. The layers use LeakyReLU and batchnorm (besides the first layer). The output is a singular scalar from a final sigmoid layer, and this represents the probability that the input is real. 

## ‘load_generator_weights’
This function takes in a generator model nn.module class as well as a file path in order to apply the saved weights from trained model to the passed-in model. This allows for inference – or generation – without the need to retrain the model everytime the notebook restarts. 

## 'generate_and_show_image'
This function takes in a generator model nn.module class as well as a file path, the size of the noise vector (latent space) and the device (CPU / CUDA) in order to apply saved weights to a generator model that is passed-in. Next, the function generates an image from this generator. This function is used to generate with the GANs without needing to run training every time the notebook restarts.

## ‘LayerNorm2d’
This class defines a custom layer normalization layer for 4d tensors, applied independently over each channel. This is used to help stabilize training by ensuring consistent activations without needing to rely on batch statistics 


