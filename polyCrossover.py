#In this file we are going to store all the various crossover functions we may need.
from polynet import *

#This function will carry out a cross over of population DNA between two polynets by
# swapping alternating stages of the polynets. So A-A and B-B would be A-B, A-B
def singlePointCrossoverStages(specimenA, specimenB):
    #We select a crossover point. Note, the crossover point cannot be the very first or very last stage
    #Otherwise we are simply copying one parent and excluding the other.

    random.seed(None)
    crossoverPoint = random.randint(1,len(specimenA.stages)-1)
    
    #The polynet which will be the result of crossover between the two parents A and B
    childPolynet = polyNet(len(specimenA.stages),specimenA.basis,specimenA.maxOrder)

    #Stores the stage structure to be applied ot the child polynet
    childPolynet.stages = specimenA.stages[:crossoverPoint] + specimenB.stages[crossoverPoint:]
    # print(specimenA.stages)
    # print(specimenB.stages)
    # print(newStages)

    #Reconstruct the polynomial based on the new stage structure.
    childPolynet.refreshPolynet()
    return childPolynet

#This funciton will carry out a cross over of population DNA between two polynets by
# combining terms along a crossover term selected.
def singlePointCrossoverTerm(specimenA, specimenB):
    random.seed(None)
    numberOfTerms = len(specimenA.stages)*len(specimenA.stages[0])
    #This tells us the term at which we are crossing over.
    crossoverPoint = random.randint(1,numberOfTerms-1)
    #This tells us the stage in which the crossover term is.
    crossoverStage = crossoverPoint//(len(specimenA.stages[0]))

    #The polynet which will be the result of crossover between the two parents A and B
    childPolynet = polyNet(len(specimenA.stages),specimenA.basis,specimenA.maxOrder)
    #This mixes the stage at which the crossover term was selected.
    
    mixedStage = specimenA.stages[crossoverStage][crossoverPoint%len(specimenA.stages[0]):] + specimenB.stages[crossoverStage][:crossoverPoint%len(specimenA.stages[0])]
    
    #Combines the stage from A, mixed stage, and stage from B
    childPolynet.stages = specimenA.stages[:crossoverStage] + [mixedStage] + specimenB.stages[(crossoverStage+1):]
    childPolynet.refreshPolynet()
    return childPolynet

#This is a defualt crossover that just returns a new child generated randomly without any input
# from the parents
def noCrossover(specimenA, specimenB):
    return polyNet(len(specimenA.stages), specimenA.basis, specimenA.maxOrder)