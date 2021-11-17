import numpy as np
from scipy.sparse import coo_matrix
# 
class modelElement2D_T3:
    def getNodes(self,Class,modelParam,xc,yc):
        # INITIALISE
        modelParam.numNodes = (modelParam.numEls[0]+1)*(modelParam.numEls[1]+1)
        modelParam.Nodes = np.array(np.zeros((modelParam.numNodes,2))).astype(float)

        # NODAL COORDINATES
        modelParam.Nodes[:,0], modelParam.Nodes[:,1] = xc, yc
    # 
    def getElements(self,Class,modelParam,v1,v2,v3,v4):
        # INITIALISE
        modelParam.numElems = 2*modelParam.numEls[0]*modelParam.numEls[1]
        modelParam.Elements = np.array(np.zeros((modelParam.numElems,4))).astype(int)

        # ELEMENT CONNECTIVITIES
        modelParam.Elements = np.r_[np.c_[v1,v2,v3], np.c_[v2,v4,v3]]
    # 
    def getShapeFuncs(self,Pt):
        # INITIALISE
        N = np.array(np.zeros(3)).astype(float)
        dNdxi = np.array(np.zeros((3,2))).astype(float)
        
        # SHAPE FUNCS
        N = np.array([1-Pt[0]-Pt[1], Pt[0], Pt[1]])

        # SHAPE FUNCS DERIVS
        dNdxi[:,0] = np.array([-1, 1, 0])
        dNdxi[:,1] = np.array([-1, 0, 1])

        self.N, self.dNdxi = N, dNdxi
    # 
    def computeXiEta2XY(self,isc,N):
        # INITIALISE
        xi, eta = 0.0, 0.0

        # COMPUTE DN/DX
        if isc == 0:
            coord = np.array([0,1]).astype(float)
            xi, eta = np.dot(coord,N).item(0), 0.0
        elif isc == 1:
            coord1, coord2 = np.array([0,1]).astype(float), np.array([1,0]).astype(float)
            xi, eta = np.dot(coord1,N).item(0), np.dot(coord2,N).item(0)
        else:
            coord = np.array([0,1]).astype(float)
            xi, eta = 0.0, np.dot(coord,N).item(0)

        return xi, eta
    # 
    def getSubCellCoord_cell(self,modelParam,ivo,wkX):
        # INITIALISE
        xy = np.array(np.zeros((3,2))).astype(float)

        # SUBCELL
        if modelParam.numSub == 1:
            xy = np.copy(wkX)
        elif modelParam.numSub == 2:
            mm = np.mean(wkX[1:3,:],axis=0)
            if ivo == 0: xy = np.vstack((wkX[:2,:],mm))
            else: xy = np.vstack((wkX[0,:],mm,wkX[2,:]))
        else:
            mm = np.mean(wkX,axis=0)
            if ivo == 0: xy = np.vstack((wkX[:2,:],mm))
            elif ivo == 1: xy = np.vstack((wkX[1:3,:],mm))
            else: xy = np.vstack((wkX[2,:],wkX[0,:],mm))

        # SMOOTHING DOMAIN
        if modelParam.numSD == 1:
            scX, scInd = np.copy(wkX), np.array([0,1,2]).astype(int)
        elif modelParam.numSD == 2:
            scX = np.array(np.zeros((4,2))).astype(float)
            scInd = np.array(np.zeros((2,3))).astype(int)
            mm = np.mean(wkX[1:3,:],axis=0)
            # 
            scX = np.vstack((xy,mm))
            scInd = np.array([[0,1,3], [0,3,2]])
        else:
            scX = np.array(np.zeros((4,2))).astype(float)
            scInd = np.array(np.zeros((3,3))).astype(int)
            mm = np.mean(xy,axis=0)
            # 
            scX = np.vstack((xy,mm))
            scInd = np.array([[0,1,3], [1,2,3], [2,0,3]])
        
        return scX, scInd
    # 
    def getSubCellCoord_edge(self,wkInd,wkX,tgt_Edge):
        # INITIALISE
        gcoord = np.array(np.zeros((4,2))).astype(float)
        node_sc = np.array(np.zeros(3)).astype(int)
        subX = np.array(np.zeros((3,2))).astype(float)

        # SUBCELL COORDINATES
        gcoord[:-1,:] = np.copy(wkX)
        gcoord[-1,:] = np.mean(wkX,axis=0)

        # SUBCELL NODE NUMBERING
        if (sum(tgt_Edge == wkInd[:2]) > 1): node_sc = np.array([0,1,3])
        elif (sum(tgt_Edge == wkInd[1:3]) > 1): node_sc = np.array([1,2,3])
        else: node_sc = np.array([2,0,3])

        # CURRENT SUBCELL COORDINATES
        subX = gcoord[node_sc,:]

        return subX
    # 
    def getSubCellCoord_node(self,ivo,wkInd,wkX):
        # INITIALISE
        scX = np.array(np.zeros((7,2))).astype(float)
        scInd = np.array(np.zeros((2,3))).astype(int)
        m1, m2 = np.mean(wkX[:2,:],axis=0), np.mean(wkX[1:3,:],axis=0)
        m3, mm = np.mean(wkX[[0,2],:],axis=0), np.mean(wkX,axis=0)

        # SUBCELL COORDINATES
        scX[:3,:] = np.copy(wkX)
        scX[3:,:] = np.vstack((m1,m2,m3,mm))

        if ivo == wkInd[0]: scInd = np.array([[0,3,6],[0,6,5]])
        elif ivo == wkInd[1]: scInd = np.array([[1,4,6],[1,6,3]])
        else: scInd = np.array([[2,5,6],[2,6,4]])

        del m1, m2, m3, mm
        
        return scX, scInd

        
