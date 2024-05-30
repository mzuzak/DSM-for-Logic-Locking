# Low Overhead Logic Locking for System-Level Security : A Design Space Modeling Approach

DSM for Logic Locking is an open-source example script implementing Design Space Modeling for obtaining the satisfying locking configuration given an arbitary IC. It provides a mean for Python users utilizing R-function,,specifically SSANOVA modeling, while staying in the Pythoin domain An extensive overview of the design space modeling algorithm can be found in the ISLPED'24 paper titled: 

    Low Overhead Logic Locking for System-Level Security : A Design Space Modeling Approach 
    Authors: Long Lam. Maksym Melnyk, Michael Zuzak


This code-base contains the example python script for the design space modeling algorithm decsribed in Section 5 of the paper. Note that this code-base this modeling algorithm is a local TCP client, which will request points (indepdent variables) for their response. 

## Running  

This is tested and run using PyCharm 2019. Virtual environment is used to download all dependencies. DSM algorithm will send over a list of points (to the local TCP server) to be sampled at each iteration. 


## Pre-Req: 

A ton of Python Packages: 

* multiprocessing 
* numpy 
* rpy2 
    * ==rpy2== is annoying to setup >:( To download this package, you need to first have: 
        * Python version 3.7 or later 
        * R version 4.0 or later
    * NOTE: rpy2 might not work if you install R after you install the package, so make sure to already have R installed before installing it 
* PySerial 
* socket 
* statistics 
* os 
* secrets 
* pandas


## Initial Parameters 

The provided code is configured to request design point from the design space modeling for the following configurations: 

Locking type : [SFLL, AntiSAT, SLL]  
Locking size : Add up to 64 bits  
Lockable modules : [alu, decoder, branch] 

Current parameters: 

Initial points = 25  (see line 405) 
Number of iterations = 30  (see line 408) 
Number of response variables = 4  (see line 417) 
Number of independent variables = 3  (see line 411) 

Additional corners points were added to prevent out of bound interpolation (line 476 - 518) 

## Region of Interest 

See function find_ROI for the configurations for the region of interest  


## Citations:

If you have found the design space modeling algorithm useful for your research, We would greatly appreciate citations to the original work. 

    LAM, L., MELNYK, M., AND ZUZAK, M. Low overhead logic locking for system-level security: A design space modeling approach. In Proceedings of the International Symposium on Low Power Electronic Design (ISLPED), 2024


## Final remarks:

Please do not hesitate to reach out to us with any questions/comments/issues regarding the repo or the work itself:
   
    Long Lam <ll5530@rit.edu> (MS'24) / Michael Zuzak <mjzeec@rit.edu> (Assistant Professor)
    Department of Computer Engineering
    Rochester Institute of Technology (RIT) 


## Acknowledgements:
This work is supported by National Science Foundation grant 2245573

