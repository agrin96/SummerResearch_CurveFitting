#This file will store all of the possible mutation functions which may be used in the genetic algorithm
from polynet import *

#This function will carry out a mutation on a polynet. The mutation chance is checked on each
# term of the polynet therefor there may be multiple point mutations on each polynet.
# This method shouldn't be used with a high mutation rate because of the number of possible mutations.
#@specimen: the polynet to mutate
def pointMutationOnAllTerms(mutationRate,specimen):
    #We define two options with their probabilites. 0 is don't mutate, 1 is mutate.
    options = [0, 1]
    probabilities = [1-mutationRate,mutationRate]
    
    # print("before: ", specimen)
    #For each term we will select if there is a point mutation
    for stage in range(0,len(specimen.stages)):
        for term in range(0,len(specimen.stages[stage])):
            #Select if we are mutating.
            choice = np.random.choice(options,1,probabilities)
            if choice == 1:
                #We use the class variable to determine whether our polynomials will have negative powers
                if specimen.allowNegativeExponents == True:
                    specimen.stages[stage][term] = (random.randint(0,4)-2)
                else:
                    #Here we only allow positive exponents but negative terms.
                    specimen.stages[stage][term] = random.randint(0,2)
    #After completing the the point mutations for the specimen we reconstruct the polynomial.
    specimen.refreshPolynet()
    # print("after: ", specimen)
    return
    
#This function will carry out a mutation on a polynet by choosing one term at random and then deciding
# based on the mutation chance if that term should be changed.
def pointMutationOnSingleTerm(mutationRate,specimen):
    #We define two options with their probabilites. 0 is don't mutate, 1 is mutate.
    options = [0, 1]
    probabilities = [1-mutationRate,mutationRate]

    #Pick a term at random as a mutation candidate
    mutationTerm = random.randint(1, len(specimen.stages)*len(specimen.stages[0])-1)
    mutationPositionInStage = mutationTerm % len(specimen.stages[0])
    mutationStage = mutationTerm//len(specimen.stages[0])

    #Select if we are mutating.
    choice = np.random.choice(options,1,probabilities)
    if choice == 1:
        #We use the class variable to determine whether our polynomials will have negative powers
        if specimen.allowNegativeExponents == True:
            specimen.stages[mutationStage][mutationPositionInStage] = (random.randint(0,4)-2)
        else:
            #Here we only allow positive exponents but negative terms.
            specimen.stages[mutationStage][mutationPositionInStage] = random.randint(0,2)
    
    specimen.refreshPolynet()
    return

#This special mutation type emulates cellular Mitosis. We select one parent specimen and it is copied with the
# potential for mutation ('error') in the genome. This way we propogate successful specimen but also create variability.
def cellularMethodMultipleTermMutation(mutationRate,specimen):
    #We define two options with their probabilites. 0 is don't mutate, 1 is mutate.
    options = [0, 1]
    probabilities = [1-mutationRate,mutationRate]

    newSpecimen = polyNet(len(specimen.stages),specimen.basis,specimen.maxOrder)
    for i in range(0,len(specimen.stages)):
        for j in range(0,len(specimen.stages[i])):
            newSpecimen.stages[i][j] = specimen.stages[i][j]

            #Select if we are mutating. This is done as you are copying each part of the 'genome'. Thus
            # we are simulating the potential for error in copying and thus creating variability.
            choice = np.random.choice(options,1,probabilities)
            if choice == 1:
                #We use the class variable to determine whether our polynomials will have negative powers
                if specimen.allowNegativeExponents == True:
                    newSpecimen.stages[i][j] = (random.randint(0,4)-2)
                else:
                    #Here we only allow positive exponents but negative terms.
                    newSpecimen.stages[i][j] = random.randint(0,2)
    newSpecimen.refreshPolynet()
    return newSpecimen

#This special mutation type emulates cellular Mitosis. We select one parent specimen and it is copied with the
# potential for mutation ('error') in the genome. This way we propogate successful specimen but also create variability.
def cellularMethodVariableTermMutation(mutationRate,specimen,maxMutations=3):
    #We define two options with their probabilites. 0 is don't mutate, 1 is mutate.
    mutationCount = 0
    options = [0, 1]
    probabilities = [1-mutationRate,mutationRate]

    newSpecimen = polyNet(len(specimen.stages),specimen.basis,specimen.maxOrder)
    for i in range(0,len(specimen.stages)):
        for j in range(0,len(specimen.stages[i])):
            newSpecimen.stages[i][j] = specimen.stages[i][j]

            #Select if we are mutating. This is done as you are copying each part of the 'genome'. Thus
            # we are simulating the potential for error in copying and thus creating variability.
            choice = np.random.choice(options,1,probabilities)
            if choice == 1:
                #We use the class variable to determine whether our polynomials will have negative powers
                if specimen.allowNegativeExponents == True:
                    newSpecimen.stages[i][j] = (random.randint(0,4)-2)
                else:
                    #Here we only allow positive exponents but negative terms.
                    if mutationCount < maxMutations:
                        newSpecimen.stages[i][j] = random.randint(0,2)
                    else:
                        newSpecimen.refreshPolynet
                        return newSpecimen
    newSpecimen.refreshPolynet()
    return newSpecimen

#This special mutation type emulates cellular Mitosis. We select one parent specimen and it is copied with the
# potential for only 1 mutation ('error') in the genome. This way we propogate successful specimen but also create variability.
def cellularMethodSingleTermMutation(mutationRate,specimen):
    #We define two options with their probabilites. 0 is don't mutate, 1 is mutate.
    options = [0, 1]
    probabilities = [1-mutationRate,mutationRate]

    newSpecimen = polyNet(len(specimen.stages),specimen.basis,specimen.maxOrder)
    for i in range(0,len(specimen.stages)):
        for j in range(0,len(specimen.stages[i])):
            newSpecimen.stages[i][j] = specimen.stages[i][j]

    #Pick a term at random as a mutation candidate
    mutationTerm = random.randint(1, len(specimen.stages)*len(specimen.stages[0])-1)
    mutationPositionInStage = mutationTerm % len(specimen.stages[0])
    mutationStage = mutationTerm//len(specimen.stages[0])

    #Select if we are mutating.
    choice = np.random.choice(options,1,probabilities)
    if choice == 1:
        #We use the class variable to determine whether our polynomials will have negative powers
        if specimen.allowNegativeExponents == True:
            newSpecimen.stages[mutationStage][mutationPositionInStage] = (random.randint(0,4)-2)
        else:
            #Here we only allow positive exponents but negative terms.
            newSpecimen.stages[mutationStage][mutationPositionInStage] = random.randint(0,2)
    newSpecimen.refreshPolynet()

    return newSpecimen

#This is a defualt function which specifies no actual mutation be performed on the specimen
def noMutation(mutationRate,specimen):
    return