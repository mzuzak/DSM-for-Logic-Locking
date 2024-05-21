# DSM_for_logic_locking
Design Space Modeling Algorithms used for the paper Low Power Logic Locking using Design Space Modeling

Author: Long Lam (ll5530@rit.edu)

* Descripition: 

This tool performs design space modeling for the open-source RISC-V processor using Design Space Modeling, which also acts as a server-client system, with the design space modeling tool being the client, and the automation tool being the server. 

The automation tool is not included in this repo.

The DSM tool request point from the automation tool at each iteration. 

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

## Example code explaination 

The provided code is configured to request design point from the design space modeling for the following configurations: 

Locking type : [SFLL, AntiSAT, SLL]  
Locking size : Add up to 64 bits  
Lockable modules : [alu, decoder, branch] 

Current parameters: 

Initial points = 25  
Number of iterations = 30  
Number of goal variables = 4  
Number of independent variables = 3   

The script is tested on PyCharm using virtual environment. Most of the configurations are harded coded with in the script. See lines 500 to 627 for the configurations


## This work is supported by National Science Foundation grant 2245573

