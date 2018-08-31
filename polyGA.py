### Modules used for mathematical operations and computations
import numpy as np
import sympy as sp
import random

### General Modules for multiprocessing, timing, etc
from multiprocessing import Pool
from sys import argv
import time

### Utility
from operator import itemgetter
from matplotlib import pyplot as plt

### Modules necessary for different functionalities of the Algorithm
from polynet import *

from polyFitness import (
    fixedDomainCorrelationAndMeanSquareFitness, 
    meanSquareErrorFitness,
    meanSquareErrorFitnessWithVariance,
    correlationCoefficientFitness)

from polyMutations import (
    cellularMethodVariableTermMutation, 
    cellularMethodMultipleTermMutation, 
    cellularMethodSingleTermMutation, 
    noMutation)

from polyCrossover import (
    singlePointCrossoverStages, 
    singlePointCrossoverTerm, 
    noCrossover)

### Sympy symbols used globally for multi-variable problems
x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = sp.symbols("x1,x2,x3,x4,x5,x6,x7,x8,x9,x10")
n = sp.symbols("n")

####-------------------------------------------------------------------------------------------------------------####
#####################################################################################################################
####-------------------------------------------------------------------------------------------------------------####

#Stores parameters to be set by user to be passed into the genetic polynomial algorithm.
# convenient control system for changing everything in one place.
class algorithmParameters:

    def __init__(self):
        #Sympy expression variables used for problem definition. 
        #Specify the number of independent variables necessary for this particular problem here.
        self.p_problemVars = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

        #Default parameters for test. Can change for different results
        #A list of domain values for each of the problem variables. List of tuples matches one to one with problemVars.
        # so [(-np.pi,np.pi)] is a domain from -pi to pi for var x1.
        self.p_problemDomain = []

        #Default parameters for test. Can change for different results
        self.p_initialPopulation = 30
        self.p_populationCap = 30 
        self.p_specimensPruned = 29
        self.p_mutationRate = 0.001
        self.p_targetFitness = 0

        # Repersents the function used as the root part of each polynet. This is the foundational unit of expansion.
        self.p_basisFunction = (a*x+b)

        # Function we are attempting to fit to.
        self.p_objectiveFunction = (sp.sin(x) + x**2 - x**3 - 4)

        # The constraints are formulated as inequalities that will be evaluated for solution feasibility in the optimization.
        self.p_constraints = []

        # This will represent how many points to evalutate on the polyNet curve when doing fitness calculations.
        self.p_fitnessSamples = 100

        self.p_maxOrder = 3
        self.p_maxStages = 4

        # Specification of functions for running the GA.
        self.p_fitnessFunc = fixedDomainCorrelationAndMeanSquareFitness
        self.p_reproductionMethod = cellularMethodVariableTermMutation

    #Redefines the parameters to match given parameters
    def setExample1(self):
        #Example of how to define and set a problem using the parameter class.
        self.p_initialPopulation = 50
        self.p_populationCap = 50 
        self.p_specimensPruned = 45
        self.p_mutationRate = 0.001
        self.p_basisFunction = (a*x)
        self.p_targetFitness = 0

        self.p_problemVars = [x1]
        self.p_problemDomain = [(-2,2)]
        self.p_objectiveFunction = (sp.sin(x1) + x1**2 - x1**3 - 4)
        self.p_constraints = []

        self.p_fitnessSamples = 100
        self.p_maxOrder = 3
        self.p_maxStages = 4
        self.p_fitnessFunc = meanSquareErrorFitness
        self.p_reproductionMethod = cellularMethodVariableTermMutation

    #Redefines the parameters to match given parameters
    def setExample2(self):
        #Example of how to define and set a problem using the parameter class.
        self.p_initialPopulation = 50
        self.p_populationCap = 50 
        self.p_specimensPruned = 45
        self.p_mutationRate = 0.001
        self.p_basisFunction = (a*x)
        self.p_targetFitness = 0

        self.p_problemVars = [x1]
        self.p_problemDomain = [(1,30)]
        self.p_objectiveFunction = (sp.ln(x1))
        self.p_constraints = []

        self.p_fitnessSamples = 101
        self.p_maxOrder = 3
        self.p_maxStages = 4
        self.p_fitnessFunc = meanSquareErrorFitness
        self.p_reproductionMethod = cellularMethodVariableTermMutation

    #Redefines the parameters to match given parameters
    def setWeierstrassFunction(self):
        #Example of how to define and set a problem using the parameter class.
        self.p_initialPopulation = 50
        self.p_populationCap = 50 
        self.p_specimensPruned = 45
        self.p_mutationRate = 0.001
        self.p_basisFunction = (a*x)
        self.p_targetFitness = 0

        self.p_problemVars = [x1]
        self.p_problemDomain = [(-2,2)]
        self.p_objectiveFunction = sp.Sum(0.9**n*sp.cos(7**n * np.pi * x1), (n, self.p_problemDomain[0][0], self.p_problemDomain[0][1])).doit()
        self.p_constraints = []

        self.p_fitnessSamples = 101
        self.p_maxOrder = 3
        self.p_maxStages = 4
        self.p_fitnessFunc = fixedDomainCorrelationAndMeanSquareFitness
        self.p_reproductionMethod = cellularMethodVariableTermMutation

    #Redefines the parameters to match given parameters
    def setSineExample(self):
        #Example of how to define and set a problem using the parameter class.
        self.p_initialPopulation = 50
        self.p_populationCap = 50 
        self.p_specimensPruned = 45
        self.p_mutationRate = 0.001
        self.p_basisFunction = (a*x)
        self.p_targetFitness = 0

        self.p_problemVars = [x1]
        self.p_problemDomain = [(-2*np.pi,2*np.pi)]
        self.p_objectiveFunction = sp.sin(x1)
        self.p_constraints = []

        self.p_fitnessSamples = 101
        self.p_maxOrder = 3
        self.p_maxStages = 4
        self.p_fitnessFunc = meanSquareErrorFitness
        self.p_reproductionMethod = cellularMethodVariableTermMutation

    def __del__(self):
        pass

def main():

    problem = algorithmParameters()
    problem.setWeierstrassFunction()
    
    test = geneticPolynomials()
    test.setupGAProblem(problem)

    if len(argv) == 2:
        test.runGeneticPolynomials(int(argv[1]))
    else:
        test.runGeneticPolynomials(100)

class geneticPolynomials:
    
    def __init__(self):
        #Stores the population of the genetic algorithm at any given generation
        self.population = []
        #Stores the paramters passed in by a problem definition
        self.executionParameters = 0

        #Stores the best specimen curves of each generation for a time-plot
        self.functionEvolutionSamples = []
        self.converged = False

    #Call to handle setup of the genetic algorithm for each run.
    def setupGAProblem(self, algorithmParameters):
        #Setup the problem and the relevent execution parameters.
        self.executionParameters = algorithmParameters

        #Generate population using given parameters.
        self.generateInitialPopulation()

        # This is our pool of multi processing workers. They will be the size of the population since that is our unit
        # of parallelization.
        self.workers = Pool(processes=len(self.population))

    #Runs the genetic algorithm with the preselected options.
    def runGeneticPolynomials(self,generations):
        start = time.clock()

        random.seed(None)
        for i in range(0,generations):
            print("\n--------------------------------------Generation %d--------------------------------------------"%i)
            self.performSelectionOnGeneration()

            #Target fitness hit
            if self.population[0][1] < self.executionParameters.p_targetFitness:
                break

            #This is a cellular-mitosis style population variation technique. Where one of the top 3 members is selected
            # to be a single parent of a new similar child.
            while len(self.population) < self.executionParameters.p_populationCap:
                for pop in self.population[:(self.executionParameters.p_populationCap-self.executionParameters.p_specimensPruned)]:
                    childNet = self.specimenReproduction(cellularMethodVariableTermMutation,pop[0])
                
                #To prevent a natural tendency toward highly linear solutions we provide a minimum complexity
                if childNet.complexity > 0:
                    #Defualt value for new child.
                    self.population.append([childNet, 100000])
        
        #Plot the final generation to see the optimal solution found
        print(self.population[0][0])

        end = time.clock()
        print("\n----------------------------ELAPSED TIME-------------------------------")
        print("\n\t\t\t\t",end-start)
        print("\n-----------------------------------------------------------------------")

        # Close our worker pool to prevent hanging processes.
        self.workers.close()
        self.workers.join()

        self.printPopulationFitness(True)

        interval = np.linspace(self.executionParameters.p_problemDomain[0][0], self.executionParameters.p_problemDomain[0][1], 101)
        temp = self.population[0][0].constructedPolynomial.xreplace({a:self.population[0][0].constantValueA,b:self.population[0][0].constantValueB})

        specimenValues = sp.lambdify(x, temp,'numpy')(interval)
        targetValues = sp.lambdify(x1,self.executionParameters.p_objectiveFunction,'numpy')(interval)
        result = np.corrcoef(specimenValues, targetValues)
        print("CORRELATION: ", result[0][1])

        self.plotPopulationSamplesOverTime()

    #Here we generate a random population of starting polyNets using the specified attributes
    def generateInitialPopulation(self):
        random.seed(None)
        
        for _ in range(self.executionParameters.p_initialPopulation):
            exists = True
            while(exists):
                exists = False
                #Generate a random new polynet with the parameters listed
                temp = polyNet(self.executionParameters.p_maxStages, 
                                self.executionParameters.p_basisFunction, 
                                self.executionParameters.p_maxOrder)
                
                #Make sure that the current pool doesn't already contain that specimen 
                for pop in self.population:
                    if x in temp.constructedPolynomial.free_symbols:
                        if pop[0].constructedPolynomial == temp.constructedPolynomial:
                            exists = True
                    else:
                        #This handles the fact that some of the polynets come out to an integer which
                        #makes the equivalence between integer and polynomial fail and crash out.
                        if x not in pop[0].constructedPolynomial.free_symbols:
                            if pop[0].constructedPolynomial == temp.constructedPolynomial:
                                exists = True
                            
                #The population gets a default fitness of 100,000. It is recalculated after this generation 
                if exists is False: self.population.append([temp,100000])
        
    #Calculates the fitness of every memeber of the population and sorts them based on fitness.
    #Then it will remove the worst polynets based on the pruning parameter
    def performSelectionOnGeneration(self):
        #Define an input dataset for optimization on constants we will skip the num 1 spot so as to avoid running 
        # the optimization multiple times on the same speciman since it is retained between generations
        inputData = [[pop[0], 
                        self.executionParameters.p_objectiveFunction, 
                        self.executionParameters.p_problemDomain, 
                        self.executionParameters.p_fitnessSamples,
                        self.executionParameters.p_problemVars] for pop in self.population]

        #Perform the optimization on constants using our input data. Returns a 2D arary holding the constant values
        optimizations = self.workers.starmap(multiProcess_performOptimizationOnConstants,inputData)
        
        #Plug the constants derived by multiprocessing back into the population
        for i in range(1,len(self.population)):
            self.population[i][0].constantValueA = optimizations[i][0]
            self.population[i][0].constantValueB = optimizations[i][1]

        #Next we need to calculate the fitness with the new optimzied constant values. Here again we multiprocess
        #The first step is to multiprocess subsitution.
        inputData = [pop[0] for pop in self.population]
        substituions = self.workers.map(multiProcess_populationSubsitutions, inputData)

        #Next step is to calculate the fitness value of each population member
        inputData = [[self.executionParameters.p_fitnessFunc,
                        pop,
                        self.executionParameters.p_fitnessSamples, 
                        self.executionParameters.p_objectiveFunction, 
                        self.executionParameters.p_problemDomain,
                        self.executionParameters.p_problemVars] for pop in substituions]
        fitnessValues = self.workers.starmap(multiProcess_specimenFitness,inputData)


        #Now we have to set the population fitness from the output of the multiprocessor. 
        # In this step we also calculate the complexity value for completness.
        for i in range(0,len(self.population)):
            self.population[i][1] = fitnessValues[i]
            self.population[i][0].getComplexityValue_FilteredNaive()

        # Then we sort the population based on the fitness. The most fit go to the beginning of the list.
        self.population.sort(key=itemgetter(1))

        # Add the best sample from this generation to our samples to plot later.
        self.functionEvolutionSamples.append(self.population[0][0])

        # Plot the population of this generation
        self.printPopulationFitness()

        # Remove the worst specimens by slicing.
        placeToSlice = len(self.population) - self.executionParameters.p_specimensPruned
        self.population = self.population[:placeToSlice]

    #Calculates the fitness of every memeber of the population and sorts them based on fitness.
    #Then it will remove the worst polynets based on the pruning parameter
    def performTwoStepSelectionOnGeneration(self):
        if self.converged == False:
            #Define an input dataset for optimization on constants we will skip the num 1 spot so as to avoid running 
            # the optimization multiple times on the same speciman since it is retained between generations
            inputData = [[pop[0], 
                            self.executionParameters.p_objectiveFunction, 
                            self.executionParameters.p_problemDomain, 
                            self.executionParameters.p_fitnessSamples,
                            self.executionParameters.p_problemVars] for pop in self.population]

            #Perform the optimization on constants using our input data. Returns a 2D arary holding the constant values
            optimizations = self.workers.starmap(multiProcess_performOptimizationOnConstants,inputData)
            
            #Plug the constants derived by multiprocessing back into the population
            for i in range(1,len(self.population)):
                self.population[i][0].constantValueA = optimizations[i][0]
                self.population[i][0].constantValueB = optimizations[i][1]

            #Next we need to calculate the fitness with the new optimzied constant values. Here again we multiprocess
            #The first step is to multiprocess subsitution.
            inputData = [pop[0] for pop in self.population]
            substituions = self.workers.map(multiProcess_populationSubsitutions, inputData)

            #Next step is to calculate the fitness value of each population member
            inputData = [[self.executionParameters.p_fitnessFunc,
                            pop,
                            self.executionParameters.p_fitnessSamples, 
                            self.executionParameters.p_objectiveFunction, 
                            self.executionParameters.p_problemDomain,
                            self.executionParameters.p_problemVars] for pop in substituions]
            fitnessValues = self.workers.starmap(multiProcess_specimenFitness,inputData)
        else:
            #Define an input dataset for optimization on constants we will skip the num 1 spot so as to avoid running 
            # the optimization multiple times on the same speciman since it is retained between generations
            inputData = [[pop[0], 
                            self.executionParameters.p_objectiveFunction, 
                            self.executionParameters.p_problemDomain, 
                            self.executionParameters.p_fitnessSamples,
                            self.executionParameters.p_problemVars] for pop in self.population]

            #Perform the optimization on constants using our input data. Returns a 2D arary holding the constant values
            optimizations = self.workers.starmap(multiProcess_performOptimizationOnConstants,inputData)
            
            #Plug the constants derived by multiprocessing back into the population
            for i in range(1,len(self.population)):
                self.population[i][0].constantValueA = optimizations[i][0]
                self.population[i][0].constantValueB = optimizations[i][1]

            #Next we need to calculate the fitness with the new optimzied constant values. Here again we multiprocess
            #The first step is to multiprocess subsitution.
            inputData = [pop[0] for pop in self.population]
            substituions = self.workers.map(multiProcess_populationSubsitutions, inputData)

            #Next step is to calculate the fitness value of each population member
            inputData = [[correlationCoefficientFitness,
                            pop,
                            self.executionParameters.p_fitnessSamples, 
                            self.executionParameters.p_objectiveFunction, 
                            self.executionParameters.p_problemDomain,
                            self.executionParameters.p_problemVars] for pop in substituions]
            fitnessValues = self.workers.starmap(multiProcess_specimenFitness,inputData)


        #Now we have to set the population fitness from the output of the multiprocessor. 
        # In this step we also calculate the complexity value for completness.
        for i in range(0,len(self.population)):
            self.population[i][1] = fitnessValues[i]
            self.population[i][0].getComplexityValue_FilteredNaive()

        # Then we sort the population based on the fitness. The most fit go to the beginning of the list.
        self.population.sort(key=itemgetter(1))

        # Add the best sample from this generation to our samples to plot later.
        self.functionEvolutionSamples.append(self.population[0][0])

        # Plot the population of this generation
        self.printPopulationFitness()

        # Remove the worst specimens by slicing.
        placeToSlice = len(self.population) - self.executionParameters.p_specimensPruned
        self.population = self.population[:placeToSlice]

        if self.population[0][1] < 100000:
            self.converged = True

    #Print the fitness values of each population member and optionally plot the best specimen.
    #@plotBest: pass in true to have this plot the best function from the current generation
    def printPopulationFitness(self,plotBest=False):
        count = 1
        for pop in self.population:
            print(count,'  complexity:', pop[0].complexity,'\t', pop[1],'\n', pop[0].constructedPolynomial, '\n')
            print(pop[0].constructedPolynomial.xreplace({a:pop[0].constantValueA,b:pop[0].constantValueB}), '\n')
            #Use this print to get a more consice representation, but a much slower processing.
            # print(sp.simplify(pop[0].constructedPolynomial.subs({a:pop[0].constantValueA,b:pop[0].constantValueB})), '\n')
            print("Constants: (a):",pop[0].constantValueA, " (b):", pop[0].constantValueB, "\n")
            count += 1
            if count == 3:
                break

        #This plots the current best specimen versus the target function.
        if plotBest:
            temp = self.population[0][0].constructedPolynomial.xreplace({a:self.population[0][0].constantValueA,b:self.population[0][0].constantValueB})

            p = sp.plot(temp,
                        self.executionParameters.p_objectiveFunction.xreplace({x1:x}),
                        (x, self.executionParameters.p_problemDomain[0][0], self.executionParameters.p_problemDomain[0][1]), 
                        ylim=(-10,10),
                        show=False,
                        legend=True)

            p[0].line_color = 'y'
            p[0].label = "num1"
            p.show()

    #Plots a set of functions which were all the number 1 rated function in each of their generations
    def plotPopulationSamplesOverTime(self):
        figuresToPlot = []
        interval = np.linspace(self.executionParameters.p_problemDomain[0][0], 
                                        self.executionParameters.p_problemDomain[0][1], 
                                        self.executionParameters.p_fitnessSamples)
        for pop in self.functionEvolutionSamples:
            temp = sp.lambdify(x, pop.constructedPolynomial.xreplace({a:pop.constantValueA,b:pop.constantValueB}),'numpy')
            values = temp(interval)
            figuresToPlot.append(values)
        target = sp.lambdify(x1, self.executionParameters.p_objectiveFunction, 'numpy')(interval)

        length = len(figuresToPlot)
        delta = 1
        if length > 8:
            delta = int(length / 8)
        
        fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True)
        index = 0
        for row in ax:
            for col in row:
                col.plot(interval,figuresToPlot[index], color='r')
                col.plot(interval,target, color='g')
                # row.set_ylim(-100, 100)
                col.set_title('Generation {}'.format(index))
                if (index + delta) < length: 
                    index += delta
                else:
                    break
            if (index + delta) >= length: 
                break
        ax[-1][-1].plot(interval,figuresToPlot[index], color='b')
        ax[-1][-1].plot(interval,target, color='g')
        # ax[-1][-1].set_ylim(-10, 10)
        ax[-1][-1].set_title('Generation {}'.format(length))
        plt.show()

    #This function will determine the fitness of a particular specimen by using the function passed as argument
    #@specimen: the polynet for which to calculate fitness
    #@fitnessFunction: the function to use for calculating fitness
    def specimenFitness(self,fitnessFunction,specimen):
        return fitnessFunction(specimen, self.executionParameters.p_fitnessFunc, self.executionParameters.p_objectiveFunction)
    
    #This function will carry out any relavent mutations on a particular specimen using the function provided as argument.
    #@specimen: the polynet to mutate.
    #@mutationFunction: the function to use for performing mutation
    def mutateSpecimen(self,mutationFunction,specimen):
        mutationFunction(self.executionParameters.p_mutationRate,specimen)

    #This funciton will carry out a reproduction operation on one specimen. The inspiration comes from cellular mitosis
    # and meiosis whereby one specimen splits into two new ones with the same chromosomes. We introduce a mutation on the
    # new speciman so as to create variability.
    #@specimen: the single parent "cell" to be used
    #@mutationFunction: the function to use for mutating the child. This should be different than the regular mutation functions
    def specimenReproduction(self, mutationFunction, specimen):
        return mutationFunction(self.executionParameters.p_mutationRate, specimen)

    #This function will carry out the crossover operation specified by the crossoverfunction passed to it as argument
    #The result is a new polynet with the crossed genome.
    #@specimenA: a polynet to be used as parent 1
    #@specimenB: a polynet to be used as parent 2
    def crossoverSpecimen(self,crossoverFunction,specimenA,specimenB):
        return crossoverFunction(specimenA,specimenB)

    def __del__(self):
        pass


#Function to allowe calculation of specimen fitness in parallel
def multiProcess_specimenFitness(fitnessFunction, specimen, samples, target, domain, problemVars):
    return fitnessFunction(specimen, samples, target, domain, problemVars)

#This function is used to support pool multiprocessing calculations for constant Optimization
def multiProcess_performOptimizationOnConstants(polynet, optimizationTarget, domain, samples, problemVars, numIterations=10):
    # 1. we calculate the partial derivatives for the constants. Use dummy values for the constants.
    polynet.calculatePartialDerivatives()

    # 2. Define an interval with samples.
    interval = np.linspace(domain[0][0], domain[0][1], samples)

    # 2.5. Conver the target only once since it does not change.
    target = sp.lambdify(problemVars, optimizationTarget, 'numpy')

    for _ in range(0,numIterations):
        # 3. convert the current and target polynomial to numpy calculable expressions so that 
        #   we can optimally do operations on them.
        current = sp.lambdify(x,polynet.constructedPolynomial.xreplace({a:polynet.constantValueA,b:polynet.constantValueB}),'numpy')

        # 4. Get x value where the error between current and target is largest. We use signed error to dictate direction
        #   of change of the 'a' constant. The absolute error is necessary to accurately find the highest magnitude error. 
        signedErrors = current(interval) - target(interval)
        absoluteErrors = np.absolute(current(interval) - target(interval))
        indexOfMaximumError = np.argmax(absoluteErrors)
        xValueOfMaximumError = interval[indexOfMaximumError]

        # 5. Here we take the maximum signed error and divide it by the partial derivative with respect to 'a' of the polynomial.
        #   This way we determine how to change the 'a' to reduce the absolute error at a particular x value. Repeat for success.
        derivVal_A = abs(polynet.derivativeA.xreplace({a:polynet.constantValueA,b:polynet.constantValueB,x:xValueOfMaximumError}))

        if derivVal_A != 0:
            #Sometimes the derivative evaluates to 0 in which case divinding by zero causes errors.
            stepSizeA = signedErrors[indexOfMaximumError] / derivVal_A
            # if adjustedVal >= 1:
            #     #In the case that the derivative evaluates to a very small number i.e. 10e-9, dividing causes an overflow
            #     stepSizeA = signedErrors[indexOfMaximumError] / adjustedVal
            # else:
            #     stepSizeA = signedErrors[indexOfMaximumError] * adjustedVal
            polynet.constantValueA = polynet.constantValueA + stepSizeA
        
    delta = target(interval).mean() - current(interval[int(interval.size / 2)])
    polynet.constantValueB = polynet.constantValueB + delta

    #For the multiprocessing to work we need to return the constant values
    return [polynet.constantValueA, polynet.constantValueB]

#Expedites subsitutions into polynomials by making them a parallel prcessable.
def multiProcess_populationSubsitutions(specimen):
    return specimen.constructedPolynomial.xreplace({a:specimen.constantValueA, b:specimen.constantValueB})

if __name__ == '__main__':main() 