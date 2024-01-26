```python
import os

root_dir = os.getcwd()
rel_dir = 'datasets/ninapro/db1/s1/S1_A1_E1.mat'
file_path = os.path.join(root_dir, rel_dir)
```


```python
import numpy as np
from scipy.io import loadmat as ld

mat_data = ld(rel_dir, appendmat=False)
```

## Available fields in the ninapro dataset


```python
keys = list(mat_data.keys())
keys
```




    ['__header__',
     '__version__',
     '__globals__',
     'emg',
     'stimulus',
     'glove',
     'subject',
     'exercise',
     'repetition',
     'restimulus',
     'rerepetition']



## Loading all the subjects' data into a 'struct'.
This struct 'data', will have the following structure:

Data is an array of n subjects
each subject containing 3 experiments
each experiment containing a dictionary (key-value pair)
we'll load the sEMG signals from all the channels, and the cyberglove data, for the hand joint angle regression task.

We also computed the angle differences between samples, to model the correlation between joint angle variation and the incoming EMG signal. This variation per sample is given by the following equation:

\begin{align*}
    \delta_i\left[n\right] &= \begin{cases}
        0 &\quad,\text{if } n \leq 1\\
        g_i\left[n+1\right]-g_i\left[n\right]
    \end{cases}
\end{align*}


Where $i$ is a cyberglove joint ($i \in \left[1,22\right], i \in \mathbb{R}$), $\delta\left[n\right]$ is the angle variation (in $\degree$) at sample $n$, $g\left[n\right]$ is the cyberglove's uncalibrated angle at sample $n$.

## Normalize data
The emg signal was normalized from each channel's relative minimum and max, to $\left[0,1\right]$
while the glove angles were normalized, from $\left[0,360\right]$ to $\left[0,1\right]$, using:


$$x' = \frac{x - \text{min}~x}{\text{max}~x - \text{min}~x}$$

And 'delta', was normalized from $\left[-360,360\right]$ to $\left[-1,1\right]$, using:

$$x'' = 2 \frac{x - \text{min}~x}{\text{max}~x - \text{min}~x}$$


```python
import os
from scipy.io import loadmat
import numpy as np

# Define the structure array to store the data
num_subjects = 27
data = []

def normalizer(data_array,min_val,max_val,type):
    
    if min_val == max_val:  # if no difference, no norm
        return np.zeros_like(data_array)
    match type:
        case "uni": 
            return (data_array - min_val) / (max_val - min_val)            
        case "bi":
            return 2 * ( (data_array - min_val) / (max_val - min_val ) ) - 1


# Loop through each subject
for subj_idx in range(1, num_subjects + 1):
    # Set up the subject directory name and path
    subject_dir = f"s{subj_idx}"
    subject_path = os.path.join("datasets", "ninapro", "db1", subject_dir)

    # Initialize lists to hold the experimental data
    exps = []

    # Loop through each experiment for the current subject
    for exp_idx in range(1, 4):
        # Set up the experiment file name and path
        exp_name = f"S{subj_idx}_A1_E{exp_idx}.mat"
        exp_path = os.path.join(subject_path, exp_name)

        # Load the MATLAB file
        mat_file = loadmat(exp_path)

        # Extract the relevant information from the loaded dictionary
        emg = mat_file['emg']
        min_emg = np.min(emg)
        max_emg = np.max(emg)

        glove = mat_file['glove']
        
        # glove delta
        delta = np.diff(glove, axis=0) # compute differences
        delta = np.pad(delta, ((1, 0), (0, 0)), mode='constant') # first sample delta = 0

        # normalize data
        norm_emg = normalizer(emg,min_emg,np.max(emg),"uni") # min-max channel, [0;1]
        norm_glove = normalizer(emg,min_emg,np.max(glove),"uni") # min-max channel, [0;1]
        

        # Add the extracted data to the list of experiments
        exps.append({
            'emg': norm_emg,
            'glove': glove
       #     'glove_delta' : norm_glove_d
        })

    # Append the list of experiments to the data list
    data.append(exps)

# Convert the data list to NumPy arrays
data = np.array(data)
```


```python
np.shape(emg)
```




    (222194, 10)



## Post-processing
We'll normalize the sEMG channels, relative to the channel's max/min of that experiment. \[0,1\] (consultar com o prof. Pinheiro) 

We'll normalize the glove's angles from \[0,360\] to \[0,1\], and the angle variation 'delta', from \[-360,360\] to \[-1,1\]


```python
# itera articulacao
joint_num = np.shape(glove)[1]
sample_num = np.shape(glove)[0]
delta = np.zeros([sample_num,joint_num])
for joint in range(joint_num):
    # itera sample
    for sample in range(sample_num-1):
        if sample < 1:
            delta[sample][joint] = 0
        else:
            delta[sample][joint] = glove[sample+1][joint]-glove[sample][joint]               
```


```python
np.shape(np.diff(glove,axis=0))

```




    (222193, 22)




```python
sample_num
```




    222194


