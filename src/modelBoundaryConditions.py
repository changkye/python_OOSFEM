import numpy as np
from src.BCs import *
# 
class modelBoundaryConditions:
    def getModelBCs2D(self,Class,modelParam):
        # NODES ON EACH BOUNDARIES
        ltNodes,rtNodes,btmNodes,topNodes = self.boundaryNodes2D(Class,modelParam)

        if modelParam.bcType == "SS_DBC":
            BC = SS_DBC_2D()
            # DIRICHLET BCS
            BC.getDirichletBCs(modelParam,ltNodes,rtNodes,btmNodes,topNodes)
            # Neumann BCS
            BC.getNeumannBCs(modelParam)

    # 
    def boundaryNodes2D(self,Class,modelParam):
        # INITIALISE
        Common = Class.Common
        Nodes = modelParam.Nodes
        Lx = modelParam.Lx

        # GET BOUNDARY NODES
        ltNodes = Common.indices(Common.columns(Nodes,0), lambda x: x == Lx[0,0])
        rtNodes = Common.indices(Common.columns(Nodes,0), lambda x: x == Lx[0,1])
        btmNodes = Common.indices(Common.columns(Nodes,1), lambda x: x == Lx[1,0])
        topNodes = Common.indices(Common.columns(Nodes,1), lambda x: x == Lx[1,1])

        return ltNodes, rtNodes, btmNodes, topNodes
