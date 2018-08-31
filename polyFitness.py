#This file stores fitness functions which may be used in the genetic algorithm.

import numpy as np
import sympy as sp
from polynet import x

z = sp.symbols('z')

#This function will determine the fitness of a particular specimen by figuring out
#how well it matches the value of the target function at a certain number of even samples.
#@specimen: the polynet for which to calculate fitness
def absoluteDifferenceFitness(specimen,samples,targetFunction):
    # sp.pprint(specimen)
    #First we create an evenly spaced sampling interval
    interval = np.linspace(-np.pi, np.pi, samples)

    #To perform evaluation we convert symbolic functions to numpy functions
    specimenFunc = sp.lambdify(x,specimen,'numpy')
    target = sp.lambdify(x,targetFunction,'numpy')
    
    #Calculate the difference between the samples at each value sampled
    difference = np.absolute(specimenFunc(interval)-target(interval))
    
    #Sum the difference across all of the samples. This is the number we will try to reduce to 0
    #and multiply by the number of terms because we also want to keep as few terms as possible.
    fitness = np.sum(difference)
    return fitness

#This function uses mean square error to determine the fitness of a specimen.
def meanSquareErrorFitness(specimen, samples, targetFunction, domain, problemVars):
    #First we create an evenly spaced sampling interval
    interval = np.linspace(domain[0][0], domain[0][1], samples)

    #To perform evaluation we convert symbolic functions to numpy functions
    specimenFunc = sp.lambdify(x,specimen,'numpy')
    target = sp.lambdify(problemVars,targetFunction,'numpy')
    
    #Calculate the difference between the samples at each value sampled
    mean_sq_err = np.square(np.absolute(specimenFunc(interval)-target(interval))).mean()
    
    return mean_sq_err

#This function uses mean square error to determine the fitness of a specimen.
def meanSquareErrorFitnessWithVariance(specimen, samples, targetFunction, domain, problemVars):
    #First we create an evenly spaced sampling interval
    interval = np.linspace(domain[0][0], domain[0][1], samples)

    #To perform evaluation we convert symbolic functions to numpy functions
    specimenFunc = sp.lambdify(x,specimen,'numpy')
    target = sp.lambdify(problemVars,targetFunction,'numpy')

    specimenValues = specimenFunc(interval)
    targetValues = target(interval)
    
    #Calculate the difference between the samples at each value sampled
    mean_sq_err = np.square(np.absolute(specimenValues - targetValues)).mean()
    
    specimenVariance = np.var(specimenValues, ddof=1)
    targetVariance = np.var(targetValues, ddof=1)
    
    varianceFactor = abs(specimenVariance - targetVariance)
    
    #We want the varianceFacotr to get to 0 which would make our result also 0 aka ideal match
    result = mean_sq_err * varianceFactor

    return result

#This function uses the correlation coefficient between data to determine fitness of a specimen.
# a higher correlation coefficient means a better score.
def correlationCoefficientFitness(specimen, samples, targetFunction, domain, problemVars):
    #First we create an evenly spaced sampling interval
    interval = np.linspace(domain[0][0], domain[0][1], samples)

    #To perform evaluation we convert symbolic functions to numpy functions
    specimenFunc = sp.lambdify(x, specimen,'numpy')
    target = sp.lambdify(problemVars,targetFunction,'numpy')

    #To find correlation coerfficient we get values of the desired funciton and the specimen along the interval
    specimenValues = specimenFunc(interval)
    targetValues = target(interval)

    result = np.corrcoef(specimenValues, targetValues)

    #Because the correlation coefficient generates a value from 0-1 (with 1 being the best) we convert the range to be
    # from (1-0) since our fitness is minimized 
    # print(result[0][1])
    return (1 - result[0][1])

#This fitness function attempts to use correlation and mean square error together by putting the mean square error
# through the funciton 1/1+e^x which constrains the value domain to [0,1] just like the correlation value.
def fixedDomainCorrelationAndMeanSquareFitness(specimen, samples, targetFunction, domain, problemVars):
     #First we create an evenly spaced sampling interval
    interval = np.linspace(domain[0][0], domain[0][1], samples)

    #To perform evaluation we convert symbolic functions to numpy functions
    specimenFunc = sp.lambdify(x, specimen,'numpy')
    target = sp.lambdify(problemVars,targetFunction,'numpy')

    #To find correlation coerfficient we get values of the desired funciton and the specimen along the interval
    specimenValues = specimenFunc(interval)
    targetValues = target(interval)

    #Calculate the difference between the samples at each value sampled
    mean_sq_err = np.square(np.absolute(specimenValues-targetValues)).mean()

    #Calculate correlation coefficient
    #Because the correlation coefficient generates a value from 0-1 (with 1 being the best) we convert the range to be
    # from (1-0) since our fitness is minimized 
    result = (1 - abs(np.corrcoef(specimenValues, targetValues)[0][1]))
    if np.isnan(result):
        #Handling nan exception for correlation coefficient. Not sure exactly yet why this happens. Seems to be 
        # occuring inside the numpy calculation so the fix is simply to detect and ignore.
        result = 0.1

    # Prevent overflows by limiting minimum correlation
    result = max(result, .001)
    expander = abs(1 / (1 - sp.exp(result**5)))
    cumulativeResult = mean_sq_err / expander

    # print("MSE: ", mean_sq_err, "corrc: ", result, " expander: ", expander, "result: ", cumulativeResult)
    return cumulativeResult