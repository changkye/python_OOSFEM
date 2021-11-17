import numpy as np
# 
class SS_DBC_2D:
    def getDirichletBCs(self,modelParam,ltNodes,rtNodes,btmNodes,topNodes):
        # FIXED NODES
        fixNodes = np.unique(np.r_[ltNodes,rtNodes,btmNodes,topNodes])
        nfix = len(fixNodes)
        X = np.array(np.zeros((2*nfix,2))).astype(float)
        X = modelParam.Nodes[fixNodes,:]

        modelParam.BCDof = np.array(np.zeros(2*nfix)).astype(int)
        modelParam.BCDof[0::2], modelParam.BCDof[1::2] = 2*fixNodes, 2*fixNodes+1
        modelParam.BCVal = np.array(np.zeros(2*nfix)).astype(float)
        modelParam.BCVal[0::2] = 1.0*X[:,1]
    # 
    def getNeumannBCs(self,modelParam):
        # INITIALISE
        modelParam.F = np.array(np.zeros(modelParam.DOFs)).astype(float)
        