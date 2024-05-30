"""
DSE_main.py

Python program to perform DSE methods to obtain "optimal" design configuration

Author : Long Lam

Requirements:
    -Python version >= 3.7 
    -if the gss folder is not cloned from git, then please install it from
        https://cran.r-project.org/web/packages/gss/index.html
            Window binaries: r-devel
    - IMPORTANT - if you can't install the latest version of rpy2 then please install version 3.5.12
            "pip install rpy2==3.5.12"
    - If there is any error related to the "R_HOME", then put R_HOME=C:<where you install R>\R\R-4.3.1  in the list
            of environmental variables

Brief descripition: 
    This is a client end of the tool. The tool request points to the server (which is the automation collecting points), it end uses DSM 
    to compute the satifsying configuration based on the ROI defined by the user 

"""

# Below are all the packages we need
import os
import random
import json
import socket
from itertools import product
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, robjects

# Constant host and port numbers for local TCP server-client
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on

# To save time from typing
r = robjects.r 
'''To run typed R codes in Python , used r(code)'''

# importing the gss package into gss variable
path = os.path.dirname(__file__)
# gss = importr('gss')               ## uncommment this if you were able to move the gss folder into the R directory
gss = importr('gss', lib_loc=path)   ## uncommment this if you can't move the gss package into the R directory

# Verbose - True if you want every little detail of the tool to be printed, False otherwise
verbose = False  # Set this to false for no verbosity

"""
Below are some utility functions used for this work 
"""

def convert2R(name, value):
    """
    Convert Python variables to R environment variables.
    """
    if isinstance(value[0], str):
        robjects.globalenv[name] = robjects.StrVector(value)
        r(f"{name} <- factor({name})")
    else:
        robjects.globalenv[name] = robjects.FloatVector(value)
        print(f"Converted {name} with value : {value} to R")
    # r(f"print({name})")


def SSANOVA(formula):
    """
    Fit SSANOVA model and return the model name 

    the model name is a R variale in the R enivronment, which is only accessiable via R  
    """
    r_code = f"""
        
       model.SSANOVA_fit <- tryCatch({{
          model.SSANOVA_fit <- ssanova0({formula})
        }}, error = function(e) {{
          cat("An error occurred in SSANOVA0:", conditionMessage(e), "\n using maxiter = 200 instead\n")

           tryCatch({{
            model.SSANOVA_fit <- ssanova0({formula}, maxiter=50000)
           }}, error = function(e) {{
             cat("An error occurred with maxiter=50000:", conditionMessage(e), "\n")
               model.SSANOVA_fit <- ssanova({formula})
            }}) 
        }}, finally = {{
          cat("\n")
        }})
        """

    verbose_code = f"print(summary(model.SSANOVA_fit))\n"

    r_code = r_code + verbose_code if verbose == True else r_code
    # Execute the R code in Python
    r(r_code)

    SSANOVA_fit = get_R2("model.SSANOVA_fit")

    print(f"R^2 of the SSANOVA fit:{SSANOVA_fit}")

    return f"model.SSANOVA_fit"


# This function is not used in the main code - left over from testing.... 
def logitisc(formula):
    """
    Fit one dimensioanl logitisc model of  and return the model name 

    the model name is a R variale in the R enivronment, which is only accessiable via R  
    """

    r_code = f"model.logit_fit <- lm( log({formula}) \n"
    verbose_code = f"print(summary(model.logit_fit))\n"

    r_code = r_code + verbose_code if verbose == True else r_code
    # Execute the R code in Python
    r(r_code)

    logit_fit = get_R2("model.logit_fit")

    print(f"R^2 of a logistic fit:{logit_fit}")

    return f"model.logit_fit"

# This function is not used in the main code - left over from testing.... 
def polynomial(num, order):
    """
    Fit one dimensioanl polynomial model and return the model name 
    order of the polynomial is a user parameter 

    the model name is a R variale in the R enivronment, which is only accessiable via R  
    """

    r_code = f"model.poly_fit <- lm(y_{str(num)} ~ poly(x0,{str(order)})) \n"
    verbose_code = f"print(summary(model.poly_fit))\n"

    r_code = r_code + verbose_code if verbose == True else r_code
    # Execute the R code in Python
    r(r_code)

    poly_fit = get_R2("model.poly_fit")

    print(f"R^2 of a poly fit (Order = {str(order)}):{poly_fit}")

    return f"model.poly_fit"


def get_R2(model_name):
    """
    Return the R^2 value of the specified model.
    """
    r_code = f"R2 <- summary({model_name})$r.squared"
    r(r_code)
    return float(robjects.globalenv["R2"][0])


def get_R2_bar(R2, s, m):
    """
    This function returns the adjusted R^2 value based on the following equation

    R^2_bar = 1 - (1 - R^2) * (s-1)/ (s-m-1)

    :param R2: see equation above
    :param S: ^^
    :param m: ^
    :return: adjusted R2
    """

    return 1 - (1 - R2) * (s - 1) / (s - m - 1)


def make_model():
    # 3 varables fitting....first draft of the work uses the same modeling equation
    # f(x0) + f(x1) + f(x2) + f(x1,x2) + f(x0,x2) + f(x1,x2) 
    # x0:x2 implies the interaction of two variables, i.e. f(x0,x2) 
    return f"x0+x1 +x2+x0:x1+x1:x2+x0:x2"

def predict(formula, list_to_pred, list_of_var):
    """
    This function return the prediction vector of the model

    :param formula: Formula of the model
    :param list_of_var : list of variable correspond to list to pred
    :param list_to_pred : 2D array of prediciton list. list_to_pred[0] = SFLL, list_to_pred[1] = AntiSAT... etc
    :return: list containing the prediction value
    """

    model_name = SSANOVA(formula)

    # this is to convert the list_to_pred variable from python to R as a 2D matrix, such that is what R uses to generate prediction 
    for i in range(0, len(list_to_pred)):
        temp = f"px{i} <- c("
        for val in list_to_pred[i]:
            temp = temp + str(val) + ","
        temp = temp[:-1] + ")"
        r(temp)

    # converting the generated matrix into a data frame for prediction 
    pred_str = f"""pred_frame <- data.frame("""
    for i in range(len(list_of_var)):
        pred_str = pred_str + f"px{i},"
    pred_str = pred_str[:-1] + ")"
    r(pred_str)

    # to assiocate the column of the dataframe to the given list of variables 
    for i in range(0, len(list_of_var)):
        r(f"names(pred_frame)[{i + 1}] = \"x{list_of_var[i]}\"")

    # Generating the actual prediction 
    r_code = f"prediction <- predict({model_name}, pred_frame, se.fit=TRUE)"
    verbose_code = "\n print(prediction$se.fit)"

    r_code = r_code+ verbose_code if verbose == True else r_code
    r(r_code)

    r("pred_fit <- prediction$fit")

    prediction_r = robjects.globalenv['pred_fit']  # grabbing this from R
    prediction_1_list = list(prediction_r)

    return prediction_1_list

#Not used - left over from previous testing 
def find_smallest_non_negative(lst):
    '''
    This function finds the smallest non-negative number in a list 

    '''
    non_negative_numbers = [num for num in lst if num >= 0]
    if non_negative_numbers:
        return min(non_negative_numbers)
    else:
        return None

# Not used - left over from previous testing 
def find_index(list, point):
    '''
    This function returns the index of a point in a list, if the point is in a list 
    '''
    try:
        index = list.index(point)
        return index
    except ValueError:
        # If the element is not found, index() raises a ValueError
        return -1


def find_ROI(pred_list, mod):
    '''
    This function is a big one....
    it first defines the ROI (region of interest, not return on investment) based on the predicted points, 
    and find points that are within that ROI, regardless of how many there are.... 

    '''


    """
    Here are some of the area and power information about the base line modules that we are locking. 
    Be aware that you might not get the same data as it depends on a lot of things: 
        1) Synthesis setting
        2) Synthesis efforts 
        3) RAW RTL - Behavoiral Verilog file 
        4) Netlist RTL - this is if you have already converted the raw to bench, and go from bench to Verilog as there are optimizations done there 
    """
    unlock_area = 186597.809013 
    alu_power_base = 433.72
    branch_power_base = 317.467
    decoder_powewr_base = 30
    power_base = (5.5319 + 121.3738) * 1000
    
    # user ROI variables ... 
    key_size = 32             # how many key bits are we using ? 
    time = 24 * 60 * 60       # what is the desired SAT attack time ? 
    p_over_perct = 1.01       # what is the power overhead ? 1% ? 
    a_over_perct = 1.05       # what is the area overhead? 5% ? 
    s_preferred = (9 * 10 ** (-5)) # what is the desired attack resilience ? 

    power_overhead = power_base * p_over_perct
    area_overhead = unlock_area * a_over_perct
    # this is to choose the power based on the modules 
    mod_power = alu_power_base if mod == "alu" else branch_power_base if mod == "branch" else decoder_powewr_base 

    phi_s = 10
    phi_area = 1000
    phi_power = .1


    if verbose == True:
        print(pred_list) 
    
    # Obtaining the "best" point that meet all constraints
    non_negative_values = [(x_0, x_1, x_2, S, A, P, SAT) for x_0, x_1, x_2, S, A, P, SAT in pred_list if
                            S <  s_preferred and A < area_overhead and P - mod_power + power_base < power_overhead and SAT > time and x_0 + 2 * x_1 + x_2 == key_size]


    # this find the lowest value given our initial budgeting equation
    top_combination = min(non_negative_values, key=lambda point: (point[5]))
    if verbose == True:
        print("Here is the point in the ROI with the lowest Power consumption")
        print(top_combination)


    # S metric
    s_p = top_combination[3]

    # area
    a_p = top_combination[4]

    # power
    p_p = top_combination[5]


    ROI = []
    ROI_only_points = []

    # grabbing points within distance (defined above) from the best predicted point 
    for x_0, x_1, x_2, S, A, P, SAT in non_negative_values: # and
        if abs((s_p - S)/s_p) < phi_s and  abs(A - a_p) < phi_area and  abs((P - p_p) / p_p) < phi_power: #abs(P - p_p) < phi_power:
            ROI.append((x_0, x_1, x_2, S, A, P, SAT))

    sorted_ROI = sorted(ROI, key=lambda point: (point[5]), reverse=False)
    print(len(sorted_ROI))

    # this will only return the indices of the identified points 
    ROI_only_points = [(x_0, x_1, x_2) for x_0, x_1, x_2, _, _, _, _ in sorted_ROI]

    return sorted_ROI, ROI_only_points


def predict_SAT_times(x0, x1, x2, y, values_to_pred):
"""
 this function returns the predicted SAT attack time based on the initial points sampled for each locking techniques 

"""
    x0 = np.array(x0)
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = np.array(y)

    # remove NaN (outlier)
    valid_indices = ~np.isnan(y)
    x0 = x0[valid_indices]
    x1 = x1[valid_indices]
    x2 = x2[valid_indices]
    y = y[valid_indices]

    # Define the multi-exponential model
    def multi_exp_model(x, a, b, c, d, e, f):
        return a * (2 ** (b * x[0] - f)) + c * (2 ** (d * x[1])) + e * x[2]

    # Fit the model using curve_fit
    popt, pcov = curve_fit(multi_exp_model, (x0, x1, x2), y, p0=(1, 0.1, 0.1, 0.1, 0.1, 0.1))

    # Extract the optimized parameters
    a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt

    # Construct the fitted equation string
    fitted_equation = f"y = {a_opt:.4f} * (2 ** ({b_opt:.4f} * x[0] - {f_opt:.4f})) + {c_opt:.4f} * (2 ** ({d_opt:.4f} * x[1])) + {e_opt:.4f} * x[2]"

    if verbose == True: 
        print("Fitted equation:")
        print(fitted_equation)

    y_pred = multi_exp_model((x0, x1, x2), *popt)
    r2 = r2_score(y, y_pred)

    if verbose == True: 
        print(f"R^2 of SAT fit : {r2}")

    new_x0 = np.array(values_to_pred[0])
    new_x1 = np.array(values_to_pred[1])
    new_x2 = np.array(values_to_pred[2])

    # Use the fitted model to predict the corresponding y values for new data
    predicted_y = multi_exp_model((new_x0, new_x1, new_x2), *popt)

    return predicted_y.tolist()

"""
Below is the "main function" that I didnt put in a main function 

it goes through the list of the moduels (mod), and performs logic locking on one module, 
freeze it, then repeat until went through all of the modules in the array

"""

mod = ["decoder", "branch", "alu"]
input  = [32, 64, 64]


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

    # Start the server (automation tool first), then run this 
    s.connect((HOST, PORT))

    # go through all of the modules 
    for ii in range(len(mod)):

        # num_points - Number of points to sample
        num_points = 25

        # num_simulation - Number of simulations to run
        num_simulation = 30

        # num_x_variables - Number of independent variable
        num_x_variables = 3

        # number of points for the initial sampling
        inital_points = num_points

        # num_y_variables - Number of dependent variable
        num_y_variables = 4

        # variable distrbution - input = {key_SFLL, key_Anti_SAT, key_SLL}
        SFLL = [0]
        anti_sat = [0]
        SLL = [0]

        # this stores the input space of the design
        indep_var = []

        for _ in range(num_x_variables):
            indep_var.append([])

        # creating the cartesian product of the input space 
        # it is also represented by K_SFLL X K_ANTISAT X K_SLL 
        for SFLL_size in range(1, input[ii] + 1):  # 1,68
            SFLL.append(SFLL_size)
        for anti_sat_size in range(3, input[ii] + 1):
            anti_sat.append(anti_sat_size)
        for sll_size in range(3, input[ii] + 1):
            SLL.append(sll_size)  # 3,200

        for SFLL_size in SFLL:
            for anti_sat_size in anti_sat:
                for SLL_size in SLL:
                    if anti_sat_size == 0 and SLL_size == 0 and SFLL_size == 0:
                        continue
                    else:
                        if SFLL_size + 2*anti_sat_size + SLL_size == 64:  # change this to fit another budegt requirement 
                            indep_var[0].append(SFLL_size)
                            indep_var[1].append(anti_sat_size)
                            indep_var[2].append(SLL_size)

        # the real input space here 
        x_val = []
        for i in range(len(indep_var[0])):
            x_val.append([indep_var[0][i], indep_var[1][i], indep_var[2][i]])

        # these two arrays store the sampled points from the design space 
        sampled_dep = []
        sampled_indep = []

        for i in range(num_x_variables):
            sampled_indep.append([])

        for i in range(num_y_variables):
            sampled_dep.append([])

        # a dictionary here just to ensure that we dont sample points that have been sampled 
        dsm_dict = dict()
        existing_points_set = set()

        """
        Data collection process

        """

        old_ROI = []

        if mod[ii] == "alu":
            operations = [
                # corner cases to prevent out of bound interpolation
                [1, 0, 0],  
                [8, 0, 0],
                [0, 3, 0],
                [0, 6, 0],
                [0, 0, 3],
                [input[ii], 0, 0],
                [0, input[ii]/2, 0],
                [0, 0, input[ii]],
                [input[ii], input[ii], input[ii]]
            ]
        elif mod[ii] == "decoder":

            # this is to obtain the corner cases of the variable
            operations = [
                # corner cases to prevent out of bound interpolation
                [7, 0, 0],
                [12, 0, 0],
                [0, 3, 0],
                [0, 9, 0],
                [0, 0, 3],
                [input[ii], 0, 0],
                [0, input[ii]/2, 0],
                [0, 0, input[ii]],
                [input[ii], input[ii], input[ii]]

            ]

        elif mod[ii] == "branch":

            # this is to obtain the corner cases of the variable
            operations = [

                 [3, 0, 0],
                [9, 0, 0],
                [0, 3, 0],
                [0, 6, 0],
                [0, 0, 3],
                [input[ii], 0, 0],
                [0, input[ii]/2, 0],
                [0, 0, input[ii]]
            ]

        # adding the predefined corner cases into the dictionary 
        for operation in operations:
            index = find_index(x_val, operation)
            for i in range(num_x_variables):
                sampled_indep[i].append(operation[i])
            existing_points_set.add(index)

        # Uniform sampling method here 
        interval = len(indep_var[0]) / (inital_points)
        point = [int(round(i * interval)) for i in range(len(indep_var[0]))]

        
        while (len(existing_points_set)) < inital_points - 4:
            for new_point in point:
                if (len(existing_points_set)) == inital_points - 4:
                    break
                if new_point not in existing_points_set:  # we dont want to add point that we already have
                    existing_points_set.add(new_point)
                    new_x = x_val[new_point]
                    for i in range(num_x_variables):
                        sampled_indep[i].append(new_x[i])
                    operations.append(new_x)


        # Send the entire operations array over to the accelerator
        message = json.dumps({"lock": operations, "mod": mod[ii], "done": None})
        print("Running accelerator for lock size" + str(operations))
        s.send(message.encode())

        # Now we will receive the data back from the accelerator
        result = s.recv(9000)
        processed_data = json.loads(result.decode())
        s_recv = processed_data.get("s")
        area_recv = processed_data.get("area")
        power_recv = processed_data.get("power")
        sat_time_recv = processed_data.get("sat")
        for i in range(inital_points):
            index = find_index(x_val, operations[i])
            sampled_dep[0].append(s_recv[i])
            sampled_dep[1].append(area_recv[i])
            sampled_dep[2].append(power_recv[i])
            sampled_dep[3].append(sat_time_recv[i])

            # we then add the values into a dict, so that we dont sample them again and we get a quick value access
            dsm_dict[index] = [s_recv[i], area_recv[i], power_recv[i], sat_time_recv[i]]

        # initial sampling process is now completed

        """=======================iterative processs to find the sub-optiaml configuration=============================="""

        old = [9999999, 9999999, 99999999]

        # help me = tool please help me to find the values :)
        for help_me in range(0, num_simulation):
            print(f"Iteration {str(help_me)}")

            # First we need to store the data into the R environment
            x_var_name = []
            for i in range(0, num_x_variables):
                x_var_name.append("x" + str(i))
                convert2R(x_var_name[i], sampled_indep[i])

            if verbose == True: 
                print(sampled_dep[0])
                print(sampled_dep[1])
                print(sampled_dep[2])
                print(sampled_dep[3])

            convert2R("y0", sampled_dep[0])  # S metric
            convert2R("y1", sampled_dep[1])  # Area
            convert2R("y2", sampled_dep[2])  # Power
            # we are not using R to fit SAT time

            # this will make the model
            formula = make_model(0)


            indep_var_need = ["0", "1", "2"]

            formula0 = "y0~" + formula # for S metric
            formula1 = "y1~" + formula # for Area 
            formula2 = "y2~" + formula # for Power 

            # Making predictions 
            S_pred = predict(formula0, indep_var, indep_var_need)
            area_pred = predict(formula1, indep_var, indep_var_need)
            power_pred = predict(formula2, indep_var, indep_var_need)
            sat_pred = predict_SAT_times(sampled_indep[0], sampled_indep[1], sampled_indep[2], sampled_dep[3],
                                         indep_var)

            # Wriring the predictions to the log 
            s_pred_log = open("s_pred_log", "w")
            s_pred_log.write(str(S_pred))
            s_pred_log.close()

            a_pred_log = open("a_pred_log", "w")
            a_pred_log.write(str(area_pred))
            a_pred_log.close()

            p_pred_log = open("p_pred_log", "w")
            p_pred_log.write(str(power_pred))
            p_pred_log.close()

            # here is the entire design space, input and output space combined 
            values = [
                (indep_var[0][i], indep_var[1][i], indep_var[2][i], S_pred[i], area_pred[i], power_pred[i], sat_pred[i])
                for i in
                range(len(indep_var[1]))]

            ROI, ROI_points = find_ROI(values, mod[ii])

            
            goal_input = []
            new_indep = []
            new_dep = []
            goal_index = 0
            # Print the results
            print("ROI points:")
            for i, combination in enumerate(ROI, start=1):
                x_0, x_1, x_2, S, A, P, SAT = combination
                x = [x_0, x_1, x_2]
                new_index = find_index(x_val, x)
                print(f"{i}. Key Bits: {x},  S: {S}, Area: {A}, Power : {P}, SAT Time : {SAT}")
                if new_index not in existing_points_set:
                    existing_points_set.add(new_index)
                    new_indep.append(x)
                    num_points = num_points + 1
                else:
                    print("This point is already in the list")
                if i == 1:
                    model_pred = [S, A, P]
                    goal_input = [x_0, x_1, x_2]
                    goal_index = new_index
                    print("goal index " + str(goal_index))
                if len(new_indep) >= 5:
                    break  # we got enough points from the ROI

            old_num_points = len(existing_points_set)
            if len(ROI) == 0:
                while (len(existing_points_set)) < old_num_points + 5:
                    new_point = random.sample(range(len(indep_var[0])), 1)[0]
                    if new_point not in existing_points_set:  # we dont want to add point that we already have
                        existing_points_set.add(new_point)
                        new_x = x_val[new_point]
                        for i in range(num_x_variables):
                            new_indep[i].append(new_x[i])

            if help_me < 2 and len(new_indep) == 0:
                print("Uh oh, looks like we didnt sample enough points for initial modeling. Picking 5 additional "
                      "random points")
                while (len(existing_points_set)) < old_num_points + 5:
                    new_point = random.sample(range(len(indep_var[0])), 1)[0]
                    if new_point not in existing_points_set:  # we dont want to add point that we already have
                        new_x = x_val[new_point]
                        if new_x[0] + 2 * new_x[1] + new_x[2] == 64:
                            existing_points_set.add(new_point)
                            new_x = x_val[new_point]
                            new_indep.append(new_x)
                            num_points = num_points + 1

            print(sampled_indep)
            print(new_indep)
            for point in new_indep:
                for i in range(num_x_variables):
                    var = 9999
                    if only_AntiSAT:
                        var = 1
                    elif only_SLL:
                        var = 2
                    else:
                        var = i
                    sampled_indep[i].append(point[var])
            print(sampled_indep)

            # Now that we have the new points that we want to simulate, it is time to send them
            # Send the entire operations array over to the accelerator
            message = json.dumps({"lock": new_indep, "mod": mod[ii], "done": None})
            print("Running accelerator for lock size" + str(new_indep))
            s.send(message.encode())

            # Now we will receive the data back from the accelerator
            result = s.recv(8192)
            processed_data = json.loads(result.decode())
            s_recv = processed_data.get("s")
            area_recv = processed_data.get("area")
            power_recv = processed_data.get("power")
            sat_time_recv = processed_data.get("sat")
            # print("Received S: " + str(1/s_recv/5))
            # print("Received Area: " + str(area_recv))
            # print("Received Power: " + str(power_recv))

            for i in range(len(new_indep)):
                index = find_index(x_val, new_indep[i])
                sampled_dep[0].append((s_recv[i]))
                sampled_dep[1].append(area_recv[i])
                sampled_dep[2].append(power_recv[i])
                sampled_dep[3].append(sat_time_recv[i])
                # we then add the values into a dict, so that we dont sample them again and we get a quick value access
                dsm_dict[index] = [s_recv[i], area_recv[i], power_recv[i], sat_time_recv[i]]

            print("=================================================================================")
            print("New inputs for simulation: \n y = "
                  + str(sampled_indep))
            print("=================================================================================")

            print("=================================================================================")
            print("New outputs for simulation: \n y = "
                  + str(sampled_dep))
            print("=================================================================================")

            new = dsm_dict.get(goal_index)
            new_ROI = ROI_points
            print(f"Power iteration {help_me} {num_points}: {new[2], new[0], new[1]} at {new_ROI[0]}")
            # if (help_me == num_simulation - 1) or (abs(old[0] - new[0]) / old[0] < .01) and (
            #         abs(old[1] - new[1]) / old[1] < .01) and (abs(old[2] - new[2]) / old[2] < .01):  # """and what < .20"""
            if (num_points == 90 or help_me == num_simulation - 1 or old_ROI[:3] == new_ROI[:3]):# or (help_me == num_simulation - 1):
                configurations = new_ROI[0]
                message = json.dumps({"lock": new_indep, "mod": mod[ii], "done": configurations})
                s.send(message.encode())
                print(f"""
    =================================================================================
    DONE! {mod[ii]} configuration {new_ROI}
    S- Val : 
    Actual : {str(new[0])} , predicted : {str(model_pred[0])} , {abs(new[0] - model_pred[0]) / model_pred[0] * 100}% difference 
    Area: 
    Actual : {str(new[1])} , predicted : {str(model_pred[1])} , {abs(new[1] - model_pred[1]) / model_pred[1] * 100}% difference 
    Power: 
    Actual : {str(new[2])} , predicted : {str(model_pred[2])} , {abs(new[2] - model_pred[2]) / model_pred[2] * 100}% difference 
    
    Number of simulated points : {num_points}  Maximum number of data points : {len(indep_var[0])}
    """)
                print("percent of points simulated : " + str(
                    "{:.2%}".format(1 - abs(num_points - len(indep_var[0])) / len(indep_var[0]))))
                print("=================================================================================\n\n")

                break
            else:
                print("=================================================================================")
                print("Queued up ROI for next simulation")
                print("=================================================================================\n\n")

            old = new
            old_ROI = new_ROI
