BesselGAN is a preliminary implementation of a local descriptor based GAN for novel structure predictions.

The goal of BesselGAN is to generate structures which locally "look like" (have similar local descriptors) the training structures. This gives the possibility to generate atoms in completely different settings than the training data:
- Make molecules with different number of atoms
- Generate bulk materials with different unit cells
- Put atoms on surfaces, between layers

The advantage compared to random searches is that the output of the GAN is supposed to be already close to physically relevant structures, so relaxations are faster and less wasteful.

# Preparation
BesselGAN relies on [NeuralIL](https://github.com/Madsen-s-research-group/neuralil-public-releases/tree/main/src/neuralil) for the descriptor generaton, so it has to be installed in the same environment.

# Usage
Two use cases are shown in the example jupyter notebooks:
- Training a GAN for small molecules and evaluating it (the training can be slow on older machines)
- Filtering out repeated structures from the GAN results so the analysis can be carried out on unique structures

# Notes
- This project is still "work-in-progress": the architectures, the training scheme, the evaluation and the used descriptors are not yet optimized
- One might find it useful to train multiple GANs, since one GAN is prone to experience "mode collapse" into a subset of possible physical results