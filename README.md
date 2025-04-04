# DSMLock: Low Overhead Logic Locking for System Security with Design Space Modeling 
DSMLock is an open-source example script implementing Design Space Modeling for obtaining a satisfying locking configuration given an arbitary IC. It provides a means for Python users to utilize R functions, specifically SSANOVA modeling, while staying in the Python domain. An extensive overview of the design space modeling algorithm can be found in the TCAD'25 (if accepted):  

    Paul, Robi, Lam, Long, Melnyk, Maksym, and Zuzak, Michael, DSMLock: Low Overhead Logic Locking for System Security with Design Space Modeling 
    
This code-base (DSE_main_GENcontains the example Python script for the design space modeling algorithm described in Section 5 of the paper. Note that this code-base modeling algorithm is a local TCP client, which will request points (independent variables) for their response. 
## Running 
This was tested and run using PyCharm 2019. A virtual environment is used to download all dependencies. The DSM algorithm will send over a list of points (to the local TCP server) to be sampled at each iteration. 
## Dependencies: 
The following Python packages are required: 
* multiprocessing
* numpy
* rpy2
  * ==rpy2== is annoying to setup >:( To download this package, you need to first have:
  * Python version 3.7 or later
  * R version 4.0 or later
  * NOTE: rpy2 might not work if you install R after you install the package, so make sure to already have R installed before installing it.
* PySerial
* socket
* statistics
* os
* secrets
* pandas
## Initial Parameters 
The provided code is configured to request design points from the design space modeling for the following configurations: 

    Locking type : [SFLL, AntiSAT, SLL]  
    Locking size : Add up to 64 bits. 
    Lockable modules : [alu, decoder, branch] 
    
Current parameters: 
* Initial points = 25  (see line 825) 
* Maximum number of iterations: 30 (see line 828) 
* Number of response variables = 4  (see line 837) 
* Number of independent variables = 3(modules) * 3(locking techniques) = 9  (see line 831) 
* Additional corner points were added to prevent out-of-bounds interpolation (line 888-907).

## Example 
The code can be run using Pycharm after installing all the packages. The following output should be generated with the initial parameters listed above. 

    Running accelerator for lock size [[7, 0, 0], [12, 0, 0], [0, 3, 0], [0, 9, 0], [0, 0, 3], [32, 0, 0], [0, 16, 0], [0, 0, 32], [32, 32, 32] ... and 13 more randomly selected points]

The server (design point generator) should recieve those locking configurations. The DSM algorithmn is expecting to recieve the design metric values given those locking configurations. 
  
## Region of Interest 
See function find_ROI for the configurations for the region of interest  

Said function also incoprate GenSA to perform optimization with the predicted space 

## Old DSM code used for ISLPED'24 
See DSE_main.py along with README_old.md 

## Citations:
If you have found the design space modeling algorithm useful for your research, we would greatly appreciate citations to the original work. 

    Paul, Robi, Lam, Long, Melnyk, Maksym, and Zuzak, Michael, DSMLock: Low Overhead Logic Locking for System Security with Design Space Modeling 
    
    @inproceedings{dsm_for_ll,
        author = {Paul, Robi, Lam, Long, Melnyk, Maksym, and Zuzak, Michael},
        title = {DSMLock: Low Overhead Logic Locking for System Security with Design Space Modeling},
        year = {2025},
        publisher = {TBD},
        address = {New York, NY, USA},
        booktitle = {TBD}
    }
## Final remarks:
Please do not hesitate to reach out to us with any questions/comments/issues regarding the repo or the work itself:  

    Long Lam <ll5530@rit.edu> / Michael Zuzak <mjzeec@rit.edu>  
    Department of Computer Engineering    
    Rochester Institute of Technology (RIT) 
    
## Acknowledgements:
This work is supported by National Science Foundation grant 2245573
 
