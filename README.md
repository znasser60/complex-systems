# Tumor Growth Analysis

## tumor_growth.py
Implements basic skeleton model for tumor growth. 

Plots the growth of tumor cells for each 5th time step. 


![frame_5](https://github.com/user-attachments/assets/3d93e803-d0e7-40e3-b8f2-77fdee8edb88)

![frame_25](https://github.com/user-attachments/assets/314a44e9-1ed4-46df-802b-0ea4a80520b5)


### Input parameters 
Receives input parameters `-N`, `-T`, `-p`, `-d` for grid size, number of time steps, growth probability and death probability, respectively. 


## critical_point.py
Analysis of tumor growth size vs ratio of growth/death probability. 


![Tumorsize_ratio_Np30_Nd30_T30_N50_log](https://github.com/user-attachments/assets/26d07ef0-1b6e-461c-95cf-1db77b9ea78d)


### Input parameters
Receives input parameters `-N`, `-T`, `-Np`, `-Nd` for grid size, number of time steps, Number of values assumed for growth probability and death probability, respectively. If `-Np=30`, for instance, 30 values for the growth probability p are created. The values are equally spaced between 0.01 and 0.99. 



