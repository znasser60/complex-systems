# Tumor Growth Analysis

## tumor_growth.py
Implements basic skeleton model for tumor growth. 

Plots the growth of tumor cells for each 5th time step. 


![frame_5](https://github.com/user-attachments/assets/3d93e803-d0e7-40e3-b8f2-77fdee8edb88)

![frame_25](https://github.com/user-attachments/assets/314a44e9-1ed4-46df-802b-0ea4a80520b5)


### Input parameters 
Receives input parameters `-N`, `-T`, `-p`, `-d`, `-m` for grid size, number of time steps, growth probability, death probability, and mutation probability, respectively. 


## critical_point.py
Analysis of tumor growth size vs ratio of growth/death probability. 


![Tumorsize_ratio_Np30_Nd30_T30_N50_log](https://github.com/user-attachments/assets/26d07ef0-1b6e-461c-95cf-1db77b9ea78d)

The following plot adds the number of dead cells depending on the ratio of growth/death probability. 

![Tumorsize_ratio_Np20_Nd20_T50_N50_log_with_death](https://github.com/user-attachments/assets/fa2ad329-91b2-408f-b922-fc7a884b6959)


### Input parameters
See section "avalanche_sizes.py".
 

## avalanche_sizes.py
Investigate the distribution of avalanche sizes (= number of cell state changes in one time step). Plot the frequency of avalanche sizes on a loglog plot. Fit a powerlaw distribution through the data and evaluate its fit using Kolmogorov-Smirnov test and p-value. Plot the slope of fits for multiple p-values.

![avalanche_sizes_dist_N100_T500_p0 3_d0 05_nit_20_loglog](https://github.com/user-attachments/assets/095c2e2b-9257-4fc5-a1ca-fdc41051d4fc)

![image](https://github.com/user-attachments/assets/8068a2e8-08ec-4739-a5ac-738e845121c5)

![image](https://github.com/user-attachments/assets/3ded5735-d3ef-4133-9720-2c54f1313388)





Fitting the model for different parameters of growth and death probabilities: 
![avalanche_sizes_multiple_p_constant_d_N100_T500_nit_2_loglog](https://github.com/user-attachments/assets/31275915-bce7-418f-b596-281935a8d05f)

![avalanche_sizes_multiple_d_constant_p_N100_T500_nit_2_loglog](https://github.com/user-attachments/assets/ba99e092-04c0-40d9-b935-600a3ca8e148)

Plotting the slope of fits for different ratios: 

![image](https://github.com/user-attachments/assets/fbf20237-d22e-4c26-95f4-2457abc36da7)



### Input parameters
Receives input parameters `-N`, `-T`, `-p`, `-d` for grid size, number of time steps, growth probability and death probability (multiple values possible), respectively. Additional parameters are `-num_it` for the number of experiments of each parameter setting, and `-input`, for the filepath to a pandas dataframe object stored with pickle which contains data from a previous experiment. If such input argument is not given, this file is created. 

## fractal_dimension.py 
Calculate the fractal dimension of the simulation after T timesteps, using the box count method. Plot the fractal dimension (i.e. the exponent of the power-law fit through the box-size vs. count data). 

![image](https://github.com/user-attachments/assets/5a68222f-f8bd-4c8f-b7e6-0870cc79d966)

### Input parameters 
See avalanche_sizes.py

## tumor_growth_with_oxygen.py
Implements a tumor growth simulation with oxygen diffusion and the addition of quiescent cells.

Plots the growth of tumor cells for each 5th time step. 

### Input parameters 
Recieves input parameters `-N`, `-T`, `-p`, `-od`, `-cu`, `-nu`, `-o`, `-nt1`, `-nt2`, `-ct1`, `-ct2`, `-qt` for grid size, number of time steps, growth probability, oxygen diffusion rate, cancer cell oxygen uptake, normal cell oxygen uptake, normal cell oxygen threshold when surrounded by more normal cells, normal cell oxygen threshold when surrounded by more cancer cells, cancer cell oxygen threshold when surrounded by more cancer cells, cancer cell oxygen threshold when surrounded by more normal cells, and the number of time steps that a quiescent cell can stay alive under the oxygen threshold. 

## oxygen_phase_transition.py 
Finds the oxygen level (g) in which the system undergoes a phase transition. 

### Input parameters 
Recieves all input parameters (see tumor_growth_with_oxygen.py) to run the simulation for varying values of the initial oxygen level. 

## oxygen_cluster_sizes.py 
Calculates and plots the largest cluster sizes for all cell types for three initial oxygen levels: 0.00035g (near the critical point), 0.011g (near the true simulation value), and 0.1g (a high level with no cell death) with shaded error areas over 10 simulations. 

### Input parameters 
Recieves all input parameters (see tumor_growth_with_oxygen.py) to calculate largest cluster size for many simulations. 

## oxygen_fractal_dimensions.py 
Calculates and plots the fractal dimensions of the tumor simulation at timestep T=50 for varying levels of oxygen. 

### Input parameters 
Recieves all input parameters (see tumor_growth_with_oxygen.py) to analyze the fractal dimension of the cellular automata that is plotted in tumor_growth_with_oxygen.py. 







