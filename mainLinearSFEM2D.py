'''
    Linear Smoothed Finite Element Method
        * 2-dimension
        * cell-based, edge-based, node-based
        * plane strain/stress
        * T3, T4 with bubble, Q4
'''
# IMPORT PACKAGES
import os, time
import numpy as np
from scipy.io import savemat

# IMPORT SRC FOLDER
from src import *
# 
class declareModel:
    pass
class declareClass:
    pass

# MAIN SCRIPT
def main(modelParam):
    # **************************************************************************************
    # PRE-PROCESSING
    # **************************************************************************************
    print("")
    print("*" * 60)
    print("PRE-PROCESSING")

    # DECLARE CLASS
    t0 = time.time()
    print("\tDeclare Class...")
    Class = declareClass()
    Class.Common = commonFunctions()
    Class.Mesh = modelMeshGeneration()
    if modelParam.elemType == "Q4": Class.Els = modelElement2D_Q4()
    elif modelParam.elemType == "T3": Class.Els = modelElement2D_T3()
    Class.BCs = modelBoundaryConditions()
    Class.SFEM = modelSFEM2D()
    Class.Bound = modelElement1D_L2()
    if modelParam.sfemType == "cell": 
        Class.SDs = modelSFEM2D_cell()
        if modelParam.elemType == "Q4": modelParam.numSub, modelParam.numSD = 1,4
        else: modelParam.numSub, modelParam.numSD = 1, 3
    elif modelParam.sfemType == "edge": Class.SDs = modelSFEM2D_edge()
    else: Class.SDs = modelSFEM2D_node()
    if modelParam.sfemType == "edge": Class.El = modelElement2D_T3()
    else: Class.El = Class.Els
    Class.Solve = solveSystem()

    # MESH GENERATION
    print("\tMesh Generation...")
    Class.Mesh.getModelMesh2D(Class,modelParam)

    # BOUNDARY CONDITIONS
    print("\tBoundary Conditions...")
    Class.BCs.getModelBCs2D(Class,modelParam)

    # GAUSS QUADRATURE
    print("\tGauss Quadrature...")
    Class.SFEM.getGaussQuadrature(modelParam)
    
    # CONSTITUTIVE MATRIX
    print("\tConstitutive Matrix...")
    Class.Common.getLinearConstitutive2D(modelParam)

    # **************************************************************************************
    # MAIN PROCESSING
    # **************************************************************************************
    print("MAIN PROCESSING")
    t1 = time.time()
    print("\tCompute Stiffness...")
    Class.SFEM.getLinearStiffness2D(Class,modelParam)
    print("\tSolve The System...")
    Class.Solve.solveLinearSystem(modelParam)
    t1_1 = time.time() - t1
    print("\t* {} secs.".format(t1_1))

    # **************************************************************************************
    # POST PROCESSING
    # **************************************************************************************
    print("POST PROCESSING")
    print("\tWrite VTK...")
    resPath = './res/' + modelParam.bcType
    if not os.path.exists(resPath): os.makedirs(resPath)
    saveName = resPath+'/LinearSFEM2D_'+modelParam.sfemType+'_'+modelParam.bcType+'_'+\
        modelParam.elemType+'_'+modelParam.numEls[0].__str__()+'x'+modelParam.numEls[1].__str__()
    Class.Common.writeVTK2D(modelParam,saveName+'.vtu')
    modelParam.comp_time = time.time() - t0

    return modelParam, saveName
    
if __name__ == '__main__':
    # **************************************************************************************
    # SET INPUT DATA
    # **************************************************************************************
    modelParam = declareModel()
    modelParam.bcType = "SS_DBC"
    modelParam.elemType = "T3"
    modelParam.stressType = "plane_stn"
    modelParam.sfemType = "node"
    modelParam.Lx = np.array([[0,1],[0,1]]).astype(float)
    modelParam.numEls = 20*np.ones((2,),dtype=int)
    modelParam.matParams = np.array([1e5,0.3]).astype(float)
    modelParam.P = 100
    if modelParam.elemType == "Q4": modelParam.ng, modelParam.NPE = 1, 4
    elif modelParam.elemType == "T3": modelParam.ng, modelParam.NPE = 1, 3
    elif modelParam.elemType == "T4": modelParam.ng, modelParam.NPE = 2, 4

    # MAIN SCRIPT
    modelParam, saveName = main(modelParam)

    # INFO
    saveData = {'modelParam':modelParam}
    savemat(saveName+".mat",saveData)
    print("*"*60)
    print("+ 2D planar problem - Linear Elasticity")
    print("+ SFEM type: {}-based".format(modelParam.sfemType))
    print("+ Problem type: {}".format(modelParam.bcType))
    print("+ Stress type: {}".format(modelParam.stressType))
    print("+ Element type: {}".format(modelParam.elemType))
    print("+ Num. of Elems.: {:d}x{:d} ({:d} DOFs)".format(modelParam.numEls[0],modelParam.numEls[1],modelParam.DOFs))
    print("+ Materials:")
    print("\tYoung's modulus: {}".format(modelParam.matParams[0]))
    print("\tPoisson's ratio: {}".format(modelParam.matParams[1]))
    print("+ Computational Time: {}".format(modelParam.comp_time))
    print("\nChangkye Lee, PhD")
    print("Nov 2021, changkyelee@gmail.com")