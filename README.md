# Assessing Surgical Ergonomics in 3D
### Kevin Zhou

This repository contains the code for the above senior project. The "tools" folder contains scripts for unzipping data, 
synchronizing images, calculating camera extrinsic parameters, and measuring posture via back-projection and 
surface reconstruction. The "model" folder contains the network used in unsupervised learning to estimate the pose 
transformation across cameras. The "segmentation" folder contains
the U-Net architecture used to perform semantic segmentation, along with the trained weights in the "trained_models" subfolder. The "ergonomics_pipeline" notebook illustrates usage of the relevant
components of the project, and the necessary data for the demo is found in the "data" folder. To obtain 
the complete dataset, please contact
Professor Alex Wong at alex.wong@yale.edu.
