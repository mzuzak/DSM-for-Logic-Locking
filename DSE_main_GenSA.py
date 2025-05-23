"""
DSE_main_nonGreedy.py

Python program to perform DSE methods to obtain optimal design configuration

  - implemented NonGreedy approach that consider all variables at once 
  - addded integer linear programming solver to solve with respect to an objective value

Author : Long Lam

Note:
    -Python needs to be at least version 3.7
    -if the gss folder is not cloned from git, then please install it from
        https://cran.r-project.org/web/packages/gss/index.html
            Window binaries: r-devel
    - IMPORTANT - if you can't install the latest version of rpy2 then please install version 3.5.12
            "pip install rpy2==3.5.12"
    - If there is any error related to the "R_HOME", then put R_HOME=C:<where you install R>\R\R-4.3.1  in the list
            of environmental variables
"""

# Below are all the packages we need
import heapq
import itertools
import math
import re

from collections import Counter

import numpy as np
import os
import pandas
import random
from rpy2.robjects.packages import \
    importr  # IMPORTANT : install version 3.5.12 of this package if installing the latest one doesn't work
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import socket
import json

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on

# To save time from typing
r = robjects.r
# importing the gss package into gss variable
path = os.path.dirname(__file__)
print(path)
# gss = importr('gss')  ## uncommment this if you were able to move the gss folder into the R directory
gss = importr('gss', lib_loc=path)  ## uncommment this if you can't move the gss package into the R directory
importr('GenSA', lib_loc=path)

# Verbose - True if you want every little detail of the tool to be printed, False otherwise
verbose = False  # Set this to false for no verbosity


def convert2R(name, value):
    """
    This function set all the variables needed into the R environment
    the reason we need to do this is so that the R code can access those data

    :param name : name of the variable in R
    :param value: the data of variable "name"
    :return: Nothing :)
    """

    if isinstance(value[0], str):
        robjects.globalenv[name] = robjects.StrVector(value)
        r(f"{name} <- factor({name})")
    else:
        robjects.globalenv[name] = robjects.FloatVector(value)
        # print(f"Converted {name} with value : {value} to R")
        print(f"={value}")
    # r(f"print({name})")


def SSANOVA(formula, num):
    """
    This function returns the SSANOVA model fitting of the data provided

    :param formula: relation of the fitting in terms
    :return: Name of the model as a String
    """
    try:

        # Below is the R code used to generate the SSANOVA fit
        r_code = f"""
            
           model.SSANOVA_fit_{num} <- tryCatch({{
              model.SSANOVA_fit_{num} <- ssanova0({formula})
            }}, error = function(e) {{
              cat("")
    
               tryCatch({{
                model.SSANOVA_fit_{num} <- ssanova0({formula}, maxiter=50000)
               }}, error = function(e) {{
                 cat("")
                   model.SSANOVA_fit_{num} <- ssanova({formula})
                }}) 
            }}, finally = {{
              cat("")
            }})
            """

        verbose_code = f"print(summary(model.SSANOVA_fit_{num}))\n"

        r_code = r_code + verbose_code if verbose == True else r_code
        # Execute the R code in Python
        r(r_code)

        SSANOVA_fit = get_R2(f"model.SSANOVA_fit_{num}")

        # print(f"R^2 of the SSANOVA fit:{SSANOVA_fit}")

        return f"model.SSANOVA_fit_{num}"
    except:
        print("Model creation not possible")
        return False


def get_R2(model_name):
    """
    This function returns the R^2 value of the model

    :param model_name: Name of the interested model
    :return: R^2 value of the given model
    """

    if model_name == False:
        return -999
    else:
        # Obtaining the R^2 value in R
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


def make_model(num):
    '''
    This function creates the SSANOVA model by consider only all of the 1st order and 2nd order term
    Choose the model with the highest adjusted R^2 with only number of terms less than or equal to term limit

    '''
    # Generate regression model with the sampled data
    first_term = []
    second_term = []
    num_terms = 0
    model = {}  # a dictionary for the model term
    formula = f"y{num}~"
    num_model_terms = 0

    term_limit = num_points // 20
    matrix = np.zeros(
        (num_x_variables, num_x_variables))  # 2D array here to indicate which 2nd order term has been added

    current_R2_bar = -999
    keep_going_1st = True
    while keep_going_1st:
        # Adding first order term that is not in the model
        for i in range(0, num_x_variables):
            if not (i in first_term):
                # pair : {variable name, R^2}
                if formula == f"y{num}~":
                    model[i] = get_R2(SSANOVA(formula + "x" + str(i), num))  # creating the model
                else:
                    model[i] = get_R2(SSANOVA(formula + "+x" + str(i), num))  # creating the model
                new_R2_bar = get_R2_bar(model[i], num_points, num_model_terms)
                model[i] = new_R2_bar
                if verbose:
                    print("Adjusted R^2 = " + str(new_R2_bar) + "\n\n")
        max_R2_variable = max(model, key=model.get)
        # adding this variable into the list, so that we don't pick this variable again
        first_term.append(max_R2_variable)
        #num_model_terms = num_model_terms + 1
        new_R2_bar = get_R2_bar(model[max_R2_variable], num_points, num_model_terms)

        if new_R2_bar > current_R2_bar:
            current_R2_bar = new_R2_bar
            num_model_terms = num_model_terms + 1
            if formula == f"y{num}~":
                formula = formula + " x" + str(max_R2_variable)
            else:
                formula = formula + " + x" + str(max_R2_variable)

            # print("=================================================================================")
            # print("First order model term added : x" + str(max_R2_variable) + " with adjusted R^2 = " + str(
            #    current_R2_bar))
            # print("=================================================================================")

        else:  # first termination condition, if adding any more first order term won't help
            # print("=================================================================================")
            # print(
            # print("================="Adding new first order term does not help - Model creation done")================================================================")
            break  # break this loop

        # second termination condition - there is no more first order term to add
        keep_going_1st = False if len(first_term) == num_x_variables else True

        # first_term.append(max_R2_variable)  # storing the model term into a list
        # num_model_terms = num_model_terms + 1
        if (num_model_terms >= math.ceil(term_limit)):
            print("============================================================================")
            print("Final Formula = " + formula + " with adjusted R^2 of " + str(current_R2_bar))
            print("============================================================================")
            return formula[3:]
        # keep going flag to indicate if the tool should keep trying to find a second order term to add
        keep_going_2nd = True  # Will set to false once adding second order term does not help
        # iterating through all the variables for the second order term interaction
        while keep_going_2nd:
            model[max_R2_variable] = -999
            for i in range(0, num_x_variables):
                if not (i == max_R2_variable):  # we ain't checking the interaction of the term with itself lmfao
                    if matrix[max_R2_variable][i] == 0 and matrix[i][max_R2_variable] == 0:
                        formula_test = formula + " + x" + str(max_R2_variable) + ":x" + str(i)
                        model[i] = get_R2(
                            SSANOVA(formula_test, num))  # grabbing the R^2 of the model after adding interaction=
                        new_R2_bar = get_R2_bar(model[i], num_points, num_model_terms)
                        model[i] = new_R2_bar
                        if verbose:
                            print("Adjusted R^2 = " + str(new_R2_bar) + "\n\n")

            # Finding the next high adjusted R2 from 2nd order term
            old_R2_variable = max_R2_variable
            max_R2_variable = max(model, key=model.get)
            new_R2_bar = model[max_R2_variable]

            # if adding help
            if new_R2_bar > current_R2_bar:
                current_R2_bar = new_R2_bar
                num_model_terms = num_model_terms + 1
                formula = formula + " + x" + str(old_R2_variable) + ":x" + str(max_R2_variable)
                # print("=================================================================================")
                # print("Added second order model term : x" + str(old_R2_variable) + ":x" + str(
                #    max_R2_variable) + " with adjusted R^2 = " + str(current_R2_bar))
                # print("=================================================================================")
                second_term.append(max_R2_variable)
                num_model_terms = num_model_terms + 1
                if (num_model_terms >= math.ceil(term_limit)):
                    print("============================================================================")
                    print("Final Formula = " + formula + " with adjusted R^2 of " + str(current_R2_bar))
                    print("============================================================================")
                    return formula[3:]
                matrix[old_R2_variable][max_R2_variable] = 1
                matrix[max_R2_variable][old_R2_variable] = 1
                max_R2_variable = old_R2_variable

            else:  # doesn't help, time to move on
                keep_going_2nd = False
                # print("=================================================================================")
                # print("No more second order model term will help, moving onto the next first order model term")
                # print("=================================================================================")
    print("============================================================================")
    print("Final Formula = " + formula + " with adjusted R^2 of " + str(current_R2_bar))
    print("============================================================================")

    # this is to make sure that we store the model in the R domain
    SSANOVA(formula, num)

    return formula[3:]


def sort_by_number(item):
    return int(re.search(r'\d+', item).group())


def predict(formula, list_to_pred, list_of_var):
    """
    This function return the prediction vector of the model

    :param formula: Formula of the model
    :param list_of_var : list of variable correspond to list to pred
    :param list_to_pred : 2D array of prediciton list. list_to_pred[0] = SFLL, list_to_pred[1] = AntiSAT... etc
    :return: list containing the prediction value
    """

    model_name = SSANOVA(formula)

    i = 0
    for var in list_of_var:
        temp = f"px{var} <- c("
        for val in list_to_pred[i]:
            temp = temp + str(val) + ","
        temp = temp[:-1] + ")"
        i = i + 1
        r(temp)

    pred_str = f"""pred_frame <- data.frame("""
    for var in list_of_var:
        pred_str = pred_str + f"px{var},"
    pred_str = pred_str[:-1] + ")"
    r(pred_str)

    for i in range(0, len(list_of_var)):
        r(f"names(pred_frame)[{i + 1}] = \"x{list_of_var[i]}\"")

    r_code = f"prediction <- predict({model_name}, pred_frame, se.fit=TRUE)"
    # comment this out, but I am printing out the prediction and error
    # r_code += "\n print(prediction$se.fit)"
    r_code = r_code
    r(r_code)

    r("pred_fit <- prediction$fit")

    prediction_r = robjects.globalenv['pred_fit']  # grabbing this from R
    prediction_1_list = list(prediction_r)

    return prediction_1_list


def find_smallest_non_negative(lst):
    non_negative_numbers = [num for num in lst if num >= 0]

    if non_negative_numbers:
        return min(non_negative_numbers)
    else:
        return None


def find_index(list, point):
    try:
        index = list.index(point)
        return index
    except ValueError:
        # If the element is not found, index() raises a ValueError
        return -1


def SAT_model_fit(x0, x1, x2, y):
    """
    This function returns the a,b,c,d,e,f coefficient of the SAT model
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
        return a * (2 ** (b * x[0] - c)) + d * (2 ** (e * x[1])) + f * x[2]

    # Fit the model using curve_fit
    popt, pcov = curve_fit(multi_exp_model, (x0, x1, x2), y, p0=(1, 0.1, 0.1, 0.1, 0.1, 0.1))

    # Extract the optimized parameters
    a_opt, b_opt, c_opt, d_opt, e_opt, f_opt = popt

    # Construct the fitted equation string
    fitted_equation = f"y = {a_opt:.4f} * (2 ** ({b_opt:.4f} * x[0] - {c_opt:.4f})) + {d_opt:.4f} * (2 ** ({e_opt:.4f} * x[1])) + {f_opt:.4f} * x[2]"

    if verbose == True:
        print("Fitted equation:")
        print(fitted_equation)

    y_pred = multi_exp_model((x0, x1, x2), *popt)
    r2 = r2_score(y, y_pred)

    if verbose == True:
        print(f"R^2 of SAT fit : {r2}")

    return [a_opt, b_opt, c_opt, d_opt, e_opt, f_opt]


## new functions added for the non-greedy approach
def find_ROI(model_0_1_2_var, formula_0_1_2, model_3, num_ROI_points, prev_pred):
    '''
    Function : find_ROI
    :param model_0_1_2_var: 3xN array consisting the name of the variables need for the models
    :param formula_0_1_2: array consisting of the three fomrulas  used to fit each model
    :param model_3: 3x6 array consisting the coefficient for the SAT run time fit
    :return: list of points that are within the ROI
    '''

    # Combine variable lists and create bounds and initial guesses
    combined = set()
    combined.update(*model_0_1_2_var)
    combined_list = [str(val) for val in sorted([int(val) for val in combined])]
    r_code = "data <- data.frame("
    for var in combined_list:
        r_code = r_code + f"x{var} = x{var},"
    r_code = r_code + "y0 = y0, y1=y1, y2=y2)"
    r(r_code)

    # r("print(data)")

    # for the logic locking design space
    # model_0 = S metric, model_1 = Area, model_2 = Power, and model_3 = SAT runtime
    # also all of this is done in the R domain

    r(f"ss_anova_model_0 <- ssanova(y0 ~ {formula_0_1_2[0]}, data = data)")
    r(f"ss_anova_model_1 <- ssanova(y1 ~ {formula_0_1_2[1]}, data = data)")
    r(f"ss_anova_model_2 <- ssanova(y2 ~ {formula_0_1_2[2]}, data = data)")


    # need to create a position vector for the variable so that we can index them properly
    pos_vec = [[], [], []]  # 3 x whatever array
    pos_counter = 0
    for var in combined_list:
        for bruh in range(3):
            if var in model_0_1_2_var[bruh]:
                pos_vec[bruh].append(pos_counter + 1)
        pos_counter = pos_counter + 1


    # Creating a prediction function for each model in the R domain
    for i in range(0, 3):
        r_code = f"""
        predict_response_{i} <- function(x) {{ 
            new_data <- data.frame(
        """
        ii = 0
        for var in model_0_1_2_var[i]:
            r_code += f"x{var} = x[{pos_vec[i][ii]}],"
            ii += 1
        r_code = r_code[:-1] + ")"

        r_code += f"""
            pred <- predict(ss_anova_model_{i}, newdata = new_data) 
            if (is.na(pred)) return (Inf) 
            return (pred)
        }}
        """

        r(r_code)

    # Now we also need to create the function used for the SAT model
    sat_counter = 1  # R is 1 index
    sat_eqn = "0"
    sum_eqn = "0"

    for i in range(len(mod)):

        if str(num_technique*i+0) in combined_list:
            sat_eqn = sat_eqn + f"+ {model_3[i][0]}*2^({model_3[0][1]}*x[{sat_counter}] + {model_3[i][2]})"
            sum_eqn = sum_eqn + f"+x[{sat_counter}]"
            sat_counter = sat_counter + 1
        if str(num_technique*i+1) in combined_list:
            sat_eqn = sat_eqn + f"+ {model_3[i][3]}*2^({model_3[i][4]}*x[{sat_counter}])"
            sum_eqn = sum_eqn + f"+2*x[{sat_counter}]"
            sat_counter = sat_counter + 1
        if str(num_technique*i+2) in combined_list:
            sat_eqn = sat_eqn + f"+ {model_3[i][5]}*x[{sat_counter}]"
            sum_eqn = sum_eqn + f"+x[{sat_counter}]"
            sat_counter = sat_counter + 1


    r_code = f"""
    sat_response <- function(x) {{        
        return ({sat_eqn})
    }}

    key_response <- function(x) {{
        return ({sum_eqn})
    }}
    """
    r(r_code)

    # Define the constraint here for S metric, Area, and SAT runtime

    #For RISC V
    s_max = str(10 ** (-4))
    area_max = str(186702.8 * 1.03)
    sat_min = str(7 * 24 * 60 * 60)


    # For APRSC
    # s_max = str(10 ** (-4))
    # area_max = str(25628.74* 1.10)
    # sat_min = str(7 * 24 * 60 * 60)





    r(f"s_max <-{s_max}")
    r(f"area_max <- {area_max}")
    r(f"sat_min <-  {sat_min}")

    # Creating the objective function here
    # we will use optimization with penalty term
    #           penalty_key <- ifelse(key_response(x) != 160, 9999999999999999999, 0)

    r(f"""
        library(GenSA)
        objective_function <- function(x) {{
          s_pred <- predict_response_0(x)
          area_pred <- predict_response_1(x)
          power_pred <- predict_response_2(x)
          sat_pred <- sat_response(x)
          penalty_1 <- ifelse(10000*s_pred > 10000*s_max, 10000000*(s_max - s_pred)^2, 0)
          penalty_2 <- ifelse(10000*area_pred > 10000*area_max, 10000*(area_max - area_pred)^2 , 0)
          penalty_3 <- ifelse(sat_pred < sat_min, (sat_min)^3 + sat_pred, 0)
          penalty_key <- ifelse(key_response(x) > 160, 9999999999999999999, 0)
          result <- 10000*power_pred + penalty_1 + penalty_2 + penalty_3 + penalty_key
          return(result)
        }}
        
        # Ensure integer guesses by rounding within GenSA optimization
        integer_objective_function <- function(x) {{
          obj_value <- objective_function(round(x))
          return(obj_value)
        }}

        

    """)

    #          #cat("Parameters:", x, "Objective Value:", obj_value, "\n")

    min_data_list = "c("
    max_data_list = "c("
    mean_data_list = "c("
    #combined_list_sorted = [str(var_var) for var_var in sorted([int(var) for var in combined_list])]
    for var in combined_list:
        min_data_list += f"floor(min(data$x{var}))+3,"
        max_data_list += f"ceiling(max(data$x{var})),"
        mean_data_list += f"mean(data$x{var}),"

    min_data_list = min_data_list[:-1] + ")"
    max_data_list = max_data_list[:-1] + ")"
    mean_data_list = mean_data_list[:-1] + ")"

    # Ensure bounds and initial guess are correctly defined
    r(f"lower_bounds <- {min_data_list}")
    r(f"upper_bounds <- {max_data_list}")
    if prev_pred == None:
        r(f"initial_guess <- round({mean_data_list})")
    elif iteration_count:
        inital_guess = []
        for var in combined_list:
            inital_guess.append(str(prev_pred[int(var)]))
        inital_guess = ",".join(inital_guess)
        r(f"initial_guess <- c({inital_guess})")

    r(f"""

    set.seed(69)
     result <- GenSA(
      par = initial_guess,
      fn = integer_objective_function,
      lower = lower_bounds,
      upper = upper_bounds,
      control = list(verbose = TRUE, max.time={"100" if iteration_count <= 5 else "10" if iteration_count < 12 else "2" if iteration_count < 14 else "0.001"})
      )
    """)

    # Extract the optimal point and the corresponding respones
    r("""
        optimal_point <- round(result$par)
        s_d <- predict_response_0(optimal_point) 
        a_d <- predict_response_1(optimal_point) 
        p_d <- predict_response_2(optimal_point) 
        t_d <- sat_response(optimal_point) 
        
        print(list(s_d, a_d, p_d, t_d))
        
    """)
    # Now that we have point d , the predicted optimal point from the models, we will do a search around this point for our ROI

    # these are the stopping criteria
    # Defining the radii of the ROI
    # phi_s = 3 * 10 ** (-5)
    # phi_t = .02
    # phi_a = .05
    # phi_p = .05

    phi_s = 6 * 10 ** (-5)  # Increase sensitivity to changes
    phi_t = 0.025  # Slightly larger tolerance
    phi_a = 0.05  # Allow more variation in area
    phi_p = 0.04  # Permit wider power variations

    r(f"""
        phi_s <- {phi_s} 
        phi_t <- {phi_t} 
        phi_a <- {phi_a} 
        phi_p <- {phi_p} 
    """)

    # define a fucntion in R to check if a point is within the ROI or not
    r("""
        is_within_roi <- function(point){
          
          s_i <- predict_response_0(point)
          a_i <- predict_response_1(point)
          p_i <- predict_response_2(point)
          t_i <- sat_response(point)
          
          within_roi <- (abs(s_d - s_i) < phi_s) & (abs(p_d - p_i)/p_d < phi_p) & (abs(a_d - a_i)/a_d < phi_a) & (abs(t_d - t_i)/t_d < phi_t)
          predictions <- list(s_i = s_i, a_i = a_i, p_i = p_i, t_i = t_i)
          
          return(list(within_roi = within_roi, predictions = predictions))
       }
    """)

    column_name_str = "c("
    for var in combined_list:
        column_name_str += f"\"x{var}\","

    column_name_str = column_name_str[:-1] + ")"

    r(f"""

        generate_points_within_radius <- function(input_point, R) {{
          # Get the number of dimensions
          n_dims <- length(input_point)
          
          # Generate all possible combinations of points within the cube of side 2R centered at the input_point
          offset_grid <- expand.grid(rep(list(seq(-R, R)), n_dims))
        
          # Initialize an empty list to store valid points
          valid_points <- list()
          
          # Check each point in the grid
          for (i in 1:nrow(offset_grid)) {{
            offset <- as.numeric(offset_grid[i, ])
            candidate_point <- input_point + offset
            
            # Safeguard to ensure no element is less than 3
            if (all(candidate_point >= lower_bounds) && all(candidate_point <= upper_bounds)) {{
              valid_points[[length(valid_points) + 1]] <- candidate_point
            }}
          }}
          
          # Convert the list of valid points to a data frame
          points_df <- do.call(rbind, valid_points)
          points_df <- as.data.frame(points_df)
          col_name <- {column_name_str}
          colnames(points_df) <- col_name
        

          return(points_df)
        }}

    """)

    # generating points around the optimal points and find points within the ROI
    r(f"""
        get_points_within_roi <- function(points_df) {{
          roi_points <- list()

          for (i in 1:nrow(points_df)) {{
            point <- points_df[i, ]
            result <- is_within_roi(point)
          
            
            if (result$within_roi) {{
              point_with_power <- c(point, power = result$predictions$p_i)
              roi_points <- append(roi_points, list(point_with_power))
            }}
          }}
          
          # Convert list to data frame
          if (length(roi_points) > 0) {{
            roi_points_df <- do.call(rbind, lapply(roi_points, as.data.frame))
            
            # Sort the points by power
            roi_points_df <- roi_points_df[order(roi_points_df$power), ]
            roi_points_df$power <- NULL
            return(roi_points_df)
          }} else {{
            return(NULL)
          }}
        }}
    
    """)

    radius = "10" if len(combined_list) < 4 else "3" if len(combined_list) < 5 else "2" if len(
        combined_list) < 8 else '1'

    r(f"""
        set.seed(69)
        R <- {radius} 
        points_within_radius <- generate_points_within_radius(optimal_point, R)
        roi_points <- get_points_within_roi(points_within_radius)
    """)

    # r("print(optimal_point)")
    # r("print(roi_points)")

    roi_points = r('roi_points')

    # Convert the R dataframe to a pandas dataframe
    pandas_dataframe = pandas2ri.rpy2py(roi_points)

    # Convert the pandas dataframe to a numpy array
    ROI_numpy = pandas_dataframe.to_numpy()

    optimal_point = np.array(list(robjects.globalenv['optimal_point']))

    optimal_point = optimal_point.reshape(1, -1)

    ROI_points = np.vstack((optimal_point, ROI_numpy))

    # Convert the numpy array to a regular Python list
    ROI = ROI_points.tolist()

    return ROI


def reshape_to_array(flat_list, N):
    # Check if the length of the list is divisible by N
    if len(flat_list) % N != 0:
        raise ValueError("The length of the flat list is not divisible by N")

    # Calculate M
    M = len(flat_list) // N

    # Reshape the list into an N by M array using list comprehension
    reshaped_array = [flat_list[i * M:(i + 1) * M] for i in range(N)]
    return reshaped_array


def generate_integer_uniform_points_with_min(dimensions, v_x_min, v_x_max, num_points):
    """
    Generate points that are equally spaced in an n-dimensional design space,
    where each variable is an integer and has a specified minimum and maximum.

    Parameters:
        dimensions (int): Number of dimensions (e.g., 18).
        v_x_min (list): List of minimum values for each dimension, length = dimensions.
        v_x_max (list): List of maximum values for each dimension, length = dimensions.
        num_points (int): Number of points to generate.

    Returns:
        list: A list of equally spaced integer points in the design space.
    """
    # Validate input
    assert len(v_x_min) == dimensions, "v_x_min must have the same length as the number of dimensions."
    assert len(v_x_max) == dimensions, "v_x_max must have the same length as the number of dimensions."

    # Create points
    points = []
    for i in range(num_points):
        # Generate a point by scaling i/num_points across each dimension
        point = [
            int(round(v_min + (v_max - v_min) * (i / (num_points - 1))))
            for v_min, v_max in zip(v_x_min, v_x_max)
        ]
        points.append(point)

    return points

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec1 (list): A list of numbers representing the first vector.
    vec2 (list): A list of numbers representing the second vector.

    Returns:
    float: The cosine similarity between the two vectors, ranging from -1 to 1.

    Raises:
    ValueError: If the vectors are of different lengths.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    # Calculate the dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate the magnitudes (Euclidean norms) of the vectors
    magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude2 = sum(b ** 2 for b in vec2) ** 0.5

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return 1 - (dot_product / (magnitude1 * magnitude2))



"""=======================Setting up variables========================================="""

## for RISC V 
mod = ["alu", "branch", "decoder"]
input = [64, 64, 32]

## for APRSC
# mod = ["addc", "limc", "floata", "upb", "accum", "fmult"] # TODO: add the name of modules of interest
# input = [30, 16, 16, 34, 282, 104]  # TODO: add the input size to the modules 

num_technique = 3                   # TODO: change the number of techniques of interest
sampling_mode = 1  # 0 for random and 1 for uniform

s_max = 10 ** (-4)
area_max = 186702.8 * 1.03

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    goal_point = None

    s.connect((HOST, PORT))

    # num_points - Number of points to sample
    num_points = 20

    # num_simulation - Number of simulations to run
    num_simulation = 100

    # num_x_variables - Number of independent variable
    num_x_variables = len(mod) * num_technique  # 3 modules and 3 locking per modules

    # number of points for the initial sampling
    inital_points = num_points

    # num_y_variables - Number of dependent variable
    num_y_variables = 4

    indep_var = []

# Early convergence settings
    early_convergence = False  # New feature added
    early_convergence_array = []
    early_convergence_similarity_percent = 0.05  # Reduced from 0.05 to avoid premature stopping
    early_convergence_similarity_iteration = 3  # Increased from 3 to allow more iterations before stopping

    """
    indep_var is a 2D array 
    [X0, X1, X2, ... , Xn] 
    where Xn is an array representing one dimension 
    """
    for _ in range(num_x_variables):
        indep_var.append([])  # empty array just to initialize the 2D array

    # first step - random (or uniform) sampling  + corner cases

    if sampling_mode == 0:
        for _ in range(inital_points - 6):
            for iii in range(len(mod)):
                # Three techniques per modules
                indep_var[num_technique * iii].append(random.randint(3, input[iii]))
                indep_var[num_technique * iii + 1].append(random.randint(3, input[iii] // 2))
                indep_var[num_technique * iii + 2].append(random.randint(3, input[iii]))
    else:
            # uniform sampling
        max_val, min_val  = [], []
        anti_sat_index = [3*i+1 for i in range(len(mod))]
        for value in input:
            max_val.extend([value]*3)
        for jjj in range(len(mod)*num_technique):
            min_val.append(6) if jjj in anti_sat_index else min_val.append(3)
        points = generate_integer_uniform_points_with_min(num_x_variables, min_val, max_val, 14)
        for point in points:
            for iii in range(len(mod)*num_technique):
                indep_var[iii].append(point[iii]//2) if iii in anti_sat_index else indep_var[iii].append(point[iii])


    # this part essentially transpose the 2D array....
    x_val = []
    #for i in range(1):
    for i in range(len(indep_var[0])):
        x_array = [indep[i] for indep in indep_var]
        x_val.append(x_array)  # each element in this array is now a point in the input space

    # alu, branch, decoder, 3 values for each each indicates how many bits per flocking
    # sfll anti sat sll
    #
    x_val.append([1, 0, 0, 3, 0, 0, 7, 0, 0])  # SFLL lower bound
    x_val.append([8, 0, 0, 9, 0, 0, 12, 0, 0])  # SFLL upper bound
    x_val.append([0, 3, 0, 0, 3, 0, 0, 3, 0])  # Anti SAT lower bound
    x_val.append([0, 6, 0, 0, 6, 0, 0, 9, 0])  # Anti SAT upper bound
    x_val.append([0, 0, 3, 0, 0, 3, 0, 0, 3])  # SLL lower bound
    x_val.append([0, 0, 64, 0, 0, 64, 0, 0, 32])  # SLL upper bound



    # this array represents a lsit of points for the first variable

    indep_var[0].extend([1, 8, 0, 0, 0, 0])
    indep_var[1].extend([0, 0, 3, 6, 0, 0])
    indep_var[2].extend([0, 0, 0, 0, 3, 64])
    indep_var[3].extend([3, 9, 0, 0, 0, 0])
    indep_var[4].extend([0, 0, 3, 6, 0, 0])
    indep_var[5].extend([0, 0, 0, 0, 3, 64])
    indep_var[6].extend([7, 12, 0, 0, 0, 0])
    indep_var[7].extend([0, 0, 3, 9, 0, 0])
    indep_var[8].extend([0, 0, 0, 0, 3, 32])

    initial_guess = None
    goal_point = [99999 for _ in range(num_x_variables)]
    goal_n_1 = [99999 for _ in range(num_x_variables)]


    initial_guess = None
    goal_point = [99999 for _ in range(num_x_variables)] 
    goal_n_1 = [99999 for _ in range(num_x_variables)] 

    # send the locking configurations over to the automation tool1]]
    lock_config = [x for x in x_val]
    message = json.dumps({"lock": lock_config})
    print_accelerator_message(lock_config)
    s.send(message.encode())
    print("Waiting for response.....")

    # Now we will receive the data back from the accelerator
    result = s.recv(9000)
    processed_data = json.loads(result.decode())
    s_recv = processed_data.get("s")
    area_recv = processed_data.get("area")
    power_recv = processed_data.get("power")
    sat_time_recv = processed_data.get("sat")
    sampled_dep = []
    sampled_dep.append(s_recv)
    sampled_dep.append(area_recv)
    sampled_dep.append(power_recv)
    sampled_dep.append(sat_time_recv)


    # initial sampling process is now completed

    """=======================iterative processs to find the sub-optiaml configuration=============================="""

    for iteration_count in range(num_simulation):
        print(f"Iteration {str(iteration_count)}")
        print("Here are the data we have collected so far....")

        # First we need to store the data into the R environment
        x_var_name = []
        for i in range(0, num_x_variables):
            x_var_name.append("x" + str(i))
            print(f"indep_var[{i}] ", end="")
            convert2R(x_var_name[i], indep_var[i])

        print(f"sampled_dep[0] ", end="")
        convert2R("y0", sampled_dep[0])  # S metric
        print(f"sampled_dep[1] ", end="")
        convert2R("y1", sampled_dep[1])  # Area
        print(f"sampled_dep[2] ", end="")
        convert2R("y2", sampled_dep[2])  # Power

        for i in range(len(mod)):
            print(f"sampled_dep[3][{i}] = ", sampled_dep[3][i])

        # we are not using R to fit SAT time

        # this will make the model for each design value (S metric, area, amd power)
        formula_s = make_model(0)
        formula_a = make_model(1)
        formula_p = make_model(2)


        def find_var_need(formula):
            indep_var_need = set(re.findall(r'x(\d+)', formula))
            return sorted(list(indep_var_need), key=int)


        indep_var_need_s = find_var_need(formula_s)
        indep_var_need_a = find_var_need(formula_a)
        indep_var_need_p = find_var_need(formula_p)

        # Performing fitting for the SAT models
        # Parameters
        group_size = num_technique  # Number of indep_var elements to group for each SAT_model_fit call
        start_index = 0  # Starting index for sampled_dep
        dependent_index = 3  # Index in sampled_dep to use for dependent variables

        # Generate SAT coefficients dynamically
        sat_coefficients = [
            SAT_model_fit(
                *indep_var[group:group + group_size],
                sampled_dep[dependent_index][i]
            )
            for i, group in enumerate(range(0, len(indep_var), group_size))
        ]

        ROI_points = find_ROI([indep_var_need_s, indep_var_need_a, indep_var_need_p], [formula_s, formula_a, formula_p],
                              sat_coefficients, 500, initial_guess)

        # need to pick 5 points from the ROI list, those points can not already been sampled

        # go through each point in the ROI, add in the missing variables (add zero in the slot of the variables that aren't used)
        # check to see if point has been sampled - if it hasn't then add it for the new points to simulate, otherwise move on to the next one

        new_point_counter = 0  # increment this counter everytime a new point is added for sampling, should go up to 5

        new_x_val = []  # list containing new points to sample
        goal_point = [int(point) for point in ROI_points[0]]
        for iii in range(num_x_variables):
            if not (str(iii) in indep_var_need_s or str(iii) in indep_var_need_p or str(iii) in indep_var_need_a):
                goal_point.insert(iii, 0)
        counter = 0
        while (new_point_counter < 5 and counter < len(ROI_points)):
            new_point_2_check = ROI_points.pop(0)
            # pad the point with 0 for the unused variable
            for iii in range(num_x_variables):
                if not (str(iii) in indep_var_need_s or str(iii) in indep_var_need_p or str(iii) in indep_var_need_a):
                    new_point_2_check.insert(iii, 0)
            # check to see if the new point has already been sampled
            new_point_2_check = [int(bruh) for bruh in new_point_2_check]
            # print(new_point_2_check)
            if new_point_2_check not in x_val and new_point_2_check not in new_x_val:
                print(f"New point to sample: {new_point_2_check}")
                new_x_val.append(new_point_2_check)
                new_point_counter = new_point_counter + 1
                num_points = num_points + 1
            counter = counter + 1
        x_val.extend(new_x_val)
        new_x_tranpose = [[new_x_val[j][i] for j in range(len(new_x_val))] for i in range(len(new_x_val[0]))]

        for iiii in range(len(new_x_tranpose)):
            indep_var[iiii].extend(new_x_tranpose[iiii])

        # send the locking configurations over to the automation tool
        print("=================================================================================")
        print("New inputs for simulation:")
        print("=================================================================================")

        lock_config = [x for x in new_x_val]
        print(lock_config)
        message = json.dumps({"lock": lock_config})
        print_accelerator_message(lock_config)
        s.send(message.encode())
        print("Waiting for response.....")

        # Now we will receive the data back from the accelerator
        result = s.recv(9000)
        processed_data = json.loads(result.decode())
        s_recv = processed_data.get("s")
        area_recv = processed_data.get("area")
        power_recv = processed_data.get("power")
        sat_time_recv = processed_data.get("sat")

        sampled_dep[0].extend(s_recv)
        sampled_dep[1].extend(area_recv)
        sampled_dep[2].extend(power_recv)

        for i in range(len(mod)):
            sampled_dep[3][i].extend(sat_time_recv[i])

        # for debugging now
        print("For debug, S, Area, Power, SAT ALU, SAT Branch, SAT Decoder")
        print(s_recv)
        print(area_recv)
        print(power_recv)
        for i in range(len(mod)):
            print(sat_time_recv[i])

        print("=================================================================================")
        print(" New design values from simulations: ")
        print(f" S-metric : {s_recv}")
        print(f" Area : {area_recv}")
        print(f" Power : {power_recv}")
        print(
            f" SAT : {[sat_time_recv[0][i] + sat_time_recv[1][i] + sat_time_recv[2][i] for i in range(len(sat_time_recv[2]))]}")
        print("=================================================================================")

        ''' this block of code is used to determine if we have early convergence or not'''
        if len(early_convergence_array) == 0:
            cur_sol_index = x_val.index(goal_point)
            if sampled_dep[1][cur_sol_index] < area_max and sampled_dep[0][cur_sol_index] < s_max:
                early_convergence_array.append(goal_point)
        else:
            # need to copmpare our current solution with the previously idenitifed one
            prev_sol = early_convergence_array[len(early_convergence_array)-1]
            similarity = cosine_similarity(prev_sol, goal_point)
            prev_sol_index = x_val.index(prev_sol)

            if sampled_dep[1][cur_sol_index] < area_max and sampled_dep[0][cur_sol_index] < s_max:
                if 0 < similarity <= early_convergence_similarity_percent:
                        early_convergence_array.append(goal_point)
                else:
                    early_convergence_array = [goal_point]
            else:
                early_convergence_array = []

        early_convergence = True if len(early_convergence_array) == early_convergence_similarity_iteration else False


        print(f"Power iteration {iteration_count} simulated {num_points}")
        if (num_points == 500 or iteration_count == num_simulation - 1 or goal_point == goal_n_1 or (early_convergence == True and initial_guess in early_convergence_array)) and iteration_count > 10:
            # we are done! Let's report the values we got
            goal_index = x_val.index(initial_guess)
            goal_point  = x_val[goal_index]
            goal_power = sampled_dep[2][goal_index]
            goal_area = sampled_dep[1][goal_index]
            goal_s = sampled_dep[0][goal_index]`
            # goal_sat = sampled_dep[3][0][goal_index] + sampled_dep[3][1][goal_index] + sampled_dep[3][2][goal_index]

            goal_sat = 0
            used_modules_counter = 0
            for sat_coefficient in sat_coefficients:
                goal_sat += sat_coefficient[0] * (2 ** (sat_coefficient[1] * goal_point[used_modules_counter*3+0] - sat_coefficient[2])) + sat_coefficient[3] * (
                            2 ** (sat_coefficient[4] * goal_point[used_modules_counter*3+1])) + sat_coefficient[5] * goal_point[used_modules_counter*3+2]
                used_modules_counter += 1
            # goal_sat = sum([sat_coefficients[i] * goal_point[i] for i in range(num_x_variables)])

            print(f"""
            ==============================================================================================
            DONE! 
            Predicted point : {goal_point}
            Power : {goal_power}
            Area : {goal_area}
            S-metric : {goal_s}
            Predicted SAT Run time : {goal_sat}
            ==============================================================================================
            """)

            exit()

        else:


            print("=================================================================================")
            print("Queued up ROI for next simulation")
            print("=================================================================================\n\n")
            # Also, we need to store the previous couple optimal points like a shift register
            goal_n_1 = goal_point

            # Also also need to provide the next guess
            if initial_guess == None:
                initial_guess = goal_point
            else:
                goal_index = x_val.index(goal_point)
                initial_index = x_val.index(initial_guess)
                sat_min = 7 * 24 * 60 * 60
                if (sampled_dep[2][initial_index] > sampled_dep[2][goal_index] and sampled_dep[0][goal_index] < s_max and sampled_dep[1][goal_index] < area_max) or (sampled_dep[0][initial_index] >= s_max or sampled_dep[1][initial_index] >= area_max):
                    initial_guess = goal_point
                
