# Overview
This repo contains code for a biophysics project. The goal is to infer binding statistics and properties from images of cells. 
Each image has a certain 'mean' intensity, which can be thought of as a background or baseline intensity. We can identify the location of bound particles by measuring the variance in intensity.
Areas of the cell where proteins are bound to operators will have a higher intensity than the mean, which will show up as a significantly high variance. 
Collecting data for different concentrations in cells gives us an array of different mean-variance pairs, the goal is to use this data to infer how many bound proteins there are, along with other properties of interest like degree of cooperativity in binding, dissociation constant, etc.

# Types of files 
This repo has .py files which contain the logic to generate synthetic data based on a specific binding scheme, and notebooks which analyze either the synthetic data or real data from experiment.
