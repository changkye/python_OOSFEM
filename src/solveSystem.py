import numpy as np
import numpy.linalg as la
from dataclasses import *
# 
class solveSystem:
    def solveLinearSystem(self,modelParam):
        # INITIALISE
        K1, F1 = np.copy(modelParam.K), np.copy(modelParam.F)

        # APPLYING BCS
        numBC = len(modelParam.BCDof)
        for i in range(numBC):
            c = modelParam.BCDof[i].item(0)
            for j in range(modelParam.DOFs):
                K1[c,j] = 0.0
            K1[c,c], F1[c] = 1.0, modelParam.BCVal[i]

        # COMPUTE DISP
        modelParam.U = la.solve(K1,F1)
    # 
    def solveNonlinearSystem(self,modelParam,scale):
        # INITIALISE
        K1, R1, F1 = np.copy(modelParam.K), np.copy(modelParam.R), np.copy(modelParam.F)
        b = np.array(np.zeros(modelParam.DOFs)).astype(float)
        
        # APPLYING BCS
        b = scale*F1 - R1
        numBC = len(modelParam.BCDof)
        for i in range(numBC):
            c = modelParam.BCDof[i]
            for j in range(modelParam.DOFs):
                K1[c,j] = 0.0
            K1[c,c], b[c] = 1.0, scale*modelParam.BCVal[i] - modelParam.U[c]

        # COMPUTE DISP
        dU = la.solve(K1,b)
        modelParam.U += dU
        modelParam.dU, modelParam.b = dU, b