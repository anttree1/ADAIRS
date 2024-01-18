# ADAIRS
## Python & Library Version
### python 3.9.2
### pytorch 1.8.0
### cuda 11.3

## Code Description
### autoagment.py: Mapping vectors to the corresponding data augmentation policies.
### class_model.py: Models for various classifiers.
### general_augment.py: Natural image data augmentation model.
### material.py: Partition of adjacent brain regions set.
### operations.py: Data augmentation operations and other machine learning operations.
### picture_sen1.py & picture_sen2: Drawing pictures.
### pso_model.py: Particle swarm optimization process.
### test_ADA.py: Test the searched data augmentation policies.
### test_Augment.py: Test the searched data augmentation policies.
### test_model.py: The fitness evaluation process of particle swarm optimization.
### utils.py: Various tools used in the code.

## Operating Guide
### In pso_model.py, set the parameters you need in args. 
### In test_model, set the classifier you need to evaluate the individual fitness. 
### The list_ind variable of test_ADA.py contains the vector of the data augmentation strategy found, and the model is set to the classifier used to select the test.
### In material.py, set up other possible sets of brain regions.

## Dataset acquisition
### ABIDE I: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
### ADHD-200: https://fcon_1000.projects.nitrc.org/indi/adhd200/
### AAL & CC200 atlas: http://preprocessed-connectomes-project.org/abide/Pipelines.html
