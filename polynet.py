import sympy as sp
import numpy as np
import random

#Mathematical symbol variables used
#Avoid using single letter variables in the code to allow math symbol use
# and not have ambiguity between symbols and variables.
x = sp.symbols('x')
s = sp.symbols('s')
a = sp.symbols('a') #a and b symbols will be replaced with numbers.
b = sp.symbols('b')

class polyNet:
    #This class variable will control whether we will allow our polynomials
    #to contain negative exponents.
    allowNegativeExponents = False

    #@stages: Number of stages not including the basis function. Default is 1
    #@basis: sympy expression of basis function. Default is ax+b
    #@maxOrder: determines the highest term order allowed in any stage. 10-> x^10
    def __init__(self, stages, basis, maxOrder):
        #These variables will store the potential constant values for this polynomial
        self.constantValueA = 1
        self.constantValueB = 0

        #Store the derivatives of the polynomial function
        self.derivativeA = 0    #partial diff wrp. a
        self.derivativeB = 0    #partial diff wrp. b

        #The basis function will be what is being expanded.
        self.basis = basis if basis is not None else s
        self.maxOrder = maxOrder

        #The stages are the expansion steps for the basis function
        if stages != None:
            self.stages = [[0]*(maxOrder) for i in range(stages)]
        else:
            self.stages = [[0]*(maxOrder)]

        #This will internally store the sympy function representation of this polyNet
        self.constructedPolynomial = 0
        
        #Randomize our stages on initialization
        self.randomizeStages()
        #Construct sympy functions based on the genome.
        self.constructPolynomial()

        #Calculate complexity value of the polynomial.
        self.complexity = 0
        
    #This function is basically a central utility to refresh the internal values of the polynet
    # if there is a change in the stage structure. If you change stages, always call this function.
    def refreshPolynet(self):
        self.constructPolynomial()
        self.getComplexityValue_FilteredNaive()
        self.calculatePartialDerivatives()

    #Initializes our polyNet stages to random values.
    def randomizeStages(self):
        for stage in self.stages:
            #If a stage is all zero then a funciton cannot be made of it
            while (stage.count(0) == len(stage)):
                for i in range(0,len(stage)):
                    #We randomize the stage value here between -2 and 2 inclusive.
                    #The sign in front of the value indicates the sign of the exponent
                    #The values [1,-1] would indicate the term is added
                    #The values [2,-2] would indicate the term is subtracted
                    random.seed(None)

                    #We use the class variable to determine whether our polynomials will have negative powers
                    if self.allowNegativeExponents == True:
                        stage[i] = (random.randint(0,4)-2)
                    else:
                        #Here we only allow positive exponents but negative terms.
                        stage[i] = random.randint(0,2)

    #Constructs the sympy polynomial by passing the basis function through stages
    def constructPolynomial(self):
        self.constructedPolynomial = 0  #reset
        builtpoly = x
        for stage in self.stages:
            eq = 0
            #Build this stage and store the polynomial in z
            for i in range(0,len(stage)):
                if stage[i] != 0: 
                    #Add term based on the appropriate stage information
                    #NOTE: the 'i+1' is meant to prevent the case where an entire stage only has a constant term.
                    # this results in the constant being plugged in below during the substitution and breaking the
                    # whole polynomial. As such we simply eliminate the possibilty of constants by shifting the exponent
                    # range to start from 1.
                    if stage[i] == -2:
                        eq = eq - self.basis**(-i+1)
                    if stage[i] == -1:
                        eq = eq + self.basis**(-i+1)
                    if stage[i] == 1:
                        eq = eq + self.basis**(i+1)
                    if stage[i] == 2:
                        eq = eq - self.basis**(i+1)
            if sum(stage) == 0:
                #If the current stage is all zero then any subesequent stages should be ignored.
                self.constructedPolynomial = builtpoly
                return 
            builtpoly = builtpoly.subs(x,eq)
            
        #Here we add a fixed constant to the end of every equation. This ensures the ability to make different y-intercept,
        # otherwise our equations will always be "stuck" to zero.
        self.constructedPolynomial = builtpoly + b

    #Plots the function produced by this polynet.
    #@ylimits: this constrains the y axis to a tuple of values (ymin,ymax)
    #@xlimits: this sets the interval to plot the function on (xmin,xmax)
    #@samplePoints: If desiring uniform sampling, specify how many points to sample.
    def plotPolyNet(self,ylimits=(-100,100),xlimits=(x,-10,10),samplePoints=None):
        if samplePoints == None:
            sp.plot(self.constructedPolynomial, xlimits, ylim=ylimits)
        else:
            sp.plot(self.constructedPolynomial, xlimits, ylim=ylimits, adaptive=False, nb_of_points=samplePoints)
        
    #This function will calcualte the partial derivatives with respece to the binomial
    # constants a and b for the polynomial. Will be used in finding constant values
    def calculatePartialDerivatives(self):
        if self.constructedPolynomial is not None:
            if self.derivativeA == 0:
                #Calculate partial derivatives with respect to constants.
                self.derivativeA = sp.diff(self.constructedPolynomial, a)
                self.derivativeB = sp.diff(self.constructedPolynomial, b)
        return

    #Returns the polynomial function but with substituted values for the constants a and b
    # if valA and valB are provided, those values are substituted and an equation is returned.
    # otherwise the constant values for this polynet are used. 
    def substituteConstantValues(self,valA=None,valB=None):
        if valA is None and valB is None:
            return self.constructedPolynomial.xreplace({a:self.constantValueA,b:self.constantValueB})
        else:
            return self.constructedPolynomial.xreplace({a:valA,b:valB})

    #Returns the partialA derivative with substituted values for x and b constants
    # because the partial wrp/a tells us how changing the a value at a certain x will change
    # the funciton value at that x.
    def getSubstitutedDerivativeA(self,valX=None,valB=None):
        if self.derivativeA is not None:
            if valX is None and valB is None:
                return self.derivativeA.xreplace({x:0,b:self.constantValueB})
            else:
                return self.derivativeA.xreplace({x:valX,b:valB})

    #Calculates the complexity of a polynet based on the number of non zero terms it has.
    # The more terms, the more complex.
    def getComplexityValue_Naive(self):
        complexity = 0
        for stage in self.stages:
            for element in stage:
                if element != 0:
                    complexity += 1
        self.complexity =  complexity
        
    #Calculate complexity of polynet with a filter to prevent zero stages from being counted.
    # the complexity is also influenced by stage number. Higher stages have higher weight.
    def getComplexityValue_FilteredWeighted(self):
        complexity = 0
        for i in range(0,len(self.stages)):
            stageComplexity = 0
            for j in range(0,len(self.stages[i])):
                if self.stages[i][j] != 0:
                    stageComplexity = stageComplexity + (i+1)*(j+1)
            if stageComplexity == 0:
                self.complexity = complexity
                return
            else:
                complexity = complexity + stageComplexity
        self.complexity = complexity

    #Calculates complexity of polynet similar to above method, but no weighting system is used
    # for higher stage number. Instead, it is simply the number of non zero elements in non zero stages.
    def getComplexityValue_FilteredNaive(self):
        complexity = 0
        for i in range(0,len(self.stages)):
            stageComplexity = 0
            for j in range(0,len(self.stages[i])):
                if self.stages[i][j] != 0:
                    stageComplexity = stageComplexity + 1
            if stageComplexity == 0:
                self.complexity = complexity
                return
            else:
                complexity = complexity + stageComplexity
        self.complexity = complexity

    #Quick way to get detailed information. About specific polynet
    def __repr__(self):
        print("\n--------------------Internal Structure--------------------")
        print(self.stages)
        print("----------------------------------------------------------\n")
        sp.init_printing()
        count = 1
        for stage in self.stages:
            eq = 0
            for i in range(0,len(stage)):
                if stage[i] == -2:
                    eq = eq - self.basis**(-i+1)
                if stage[i] == -1:
                    eq = eq + self.basis**(-i+1)
                if stage[i] == 1:
                    eq = eq + self.basis**(i+1)
                if stage[i] == 2:
                    eq = eq - self.basis**(i+1)
            print("|--------------------STAGE: ", count, "--------------------|\n")
            sp.pprint(eq)
            print("----------------------------------------------------\n")
            count += 1

        print("---------------------------------------EXPANDED POLYNOMIAL----------------------------------------\n")
        sp.pprint(self.constructedPolynomial)
        print("--------------------------------------------------------------------------------------------------\n")
        print("Complexity", self.complexity)
        print("ConstantA: ", self.constantValueA, " ConstantB: ", self.constantValueB)
        return "\n\n"

    def __del__(self):
        # print("Died")
        pass