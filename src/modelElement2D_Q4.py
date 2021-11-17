import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class modelElement2D_Q4:
    def getNodes(self,Class,modelParam,xc,yc):
        # INITIALISE
        modelParam.numNodes = (modelParam.numEls[0]+1)*(modelParam.numEls[1]+1)
        modelParam.Nodes = np.array(np.zeros((modelParam.numNodes,2))).astype(float)

        # NODAL COORDINATES
        modelParam.Nodes[:,0], modelParam.Nodes[:,1] = xc, yc
    # 
    def getElements(self,Class,modelParam,v1,v2,v3,v4):
        # INITIALISE
        modelParam.numElems = modelParam.numEls[0]*modelParam.numEls[1]
        modelParam.Elements = np.array(np.zeros((modelParam.numElems,4))).astype(int)

        # ELEMENT CONNECTIVITIES
        modelParam.Elements = np.column_stack((v1, v2, v4, v3))
    # 
    def getShapeFuncs(self,Pt):
        # INITIALISE
        N = np.array(np.zeros(4)).astype(float)
        dNdxi = np.array(np.zeros((4,2))).astype(float)
        Ix = np.array([1-Pt[0], 1+Pt[0]]).astype(float)
        Iy = np.array([1-Pt[1], 1+Pt[1]]).astype(float)

        # SHAPE FUNCS
        N = np.array([Ix[0]*Iy[0], Ix[1]*Iy[0], Ix[1]*Iy[1], Ix[0]*Iy[1]])/4

        # SHAPE FUNCS DERIVS
        dNdxi[:,0] = np.array([-Iy[0], Iy[0], Iy[1], -Iy[1]])/4
        dNdxi[:,1] = np.array([-Ix[0], -Ix[1], Ix[1], Ix[0]])/4

        self.N, self.dNdxi = N, dNdxi
    # 
    def computeXiEta2XY(self,isc,N):
        # INITIALISE
        xi, eta = 0.0, 0.0
        Idx = np.array([-1,1]).astype(float)

        # COMPUTE DN/DX
        if isc == 0: xi, eta = np.dot(Idx,N).item(0), -1.0
        elif isc == 1: xi, eta = 1.0, np.dot(Idx,N).item(0)
        elif isc == 2: xi, eta = np.dot(Idx,N).item(0), 1.0
        else: xi, eta = -1.0, np.dot(Idx,N).item(0)

        return xi, eta
    # 
    def getSubCellCoord_cell(self,modelParam,ivo,wkX):
        # INITIALISE
        xy = np.array(np.zeros((4,2))).astype(float)

        # SUBCELL
        if modelParam.numSub == 1:
            xy = np.copy(wkX)
        elif modelParam.numSub == 2:
            m1, m2 = np.mean(wkX[:2,:],axis=0), np.mean(wkX[2:,:],axis=0)
            if ivo == 0: xy = np.vstack((wkX[0,:],m1,m2,wkX[3,:]))
            else: xy = np.vstack((m1,wkX[1:3,:],m2))
        else:
            m1, m2 = np.mean(wkX[:2,:],axis=0), np.mean(wkX[1:3,:],axis=0)
            m3, m4 = np.mean(wkX[2:,:],axis=0), np.mean(wkX[[0,3],:],axis=0)
            mm = np.mean(wkX,axis=0)
            if ivo == 0: xy = np.vstack((wkX[0,:],m1,mm,m4))
            elif ivo == 1: xy = np.vstack((m1,wkX[1,:],m2,mm))
            elif ivo == 2: xy = np.vstack((mm,m2,wkX[2,:],m3))
            else: xy = np.vstack((m4,mm,m3,wkX[3,:]))

        # SMOOTHING DOMAIN
        if modelParam.numSD == 1:
            scX, scInd = np.copy(xy), np.array([0,1,2,3]).astype(int)
        elif modelParam.numSD == 2:
            scX = np.array(np.zeros((6,2))).astype(float)
            scInd = np.array(np.zeros((4,2))).astype(int)
            m1, m2 = np.mean(wkX[:2,:],axis=0), np.mean(wkX[2:,:],axis=0)
            # 
            scX = np.vstack((xy,m1,m2))
            scInd = np.array([[0,4,5,3], [4,1,2,5]])
        elif modelParam.numSD == 3:
            scX = np.array(np.zeros((8,2))).astype(float)
            scInd = np.array(np.zeros((3,4))).astype(int)
            m1, m2 = np.mean(xy[:2,:],axis=0), np.mean(xy[1:3,:],axis=0)
            m3, mm = np.mean(xy[2:,:],axis=0), np.mean(xy,axis=0)
            # 
            scX = np.vstack((xy,m1,mm,m2,m3))
            scInd = np.earray([[0,4,7,3], [4,1,6,5], [5,6,2,7]])
        elif modelParam.numSD == 4:
            scX = np.array(np.zeros((9,2))).astype(float)
            scInd = np.array(np.zeros((4,4))).astype(int)
            m1, m2 = np.mean(xy[:2,:],axis=0), np.mean(xy[1:3,:],axis=0)
            m3, m4 = np.mean(xy[2:,:],axis=0), np.mean(xy[[0,3],:],axis=0)
            mm = np.mean(xy,axis=0)
            # 
            scX = np.vstack((xy,m1,m2,m3,m4,mm))
            scInd = np.array([[0,4,8,7], [4,1,5,8], [8,5,2,6], [7,8,6,3]])
        else:
            scX = np.array(np.zeros((15,2))).astype(float)
            scInd = np.array(np.zeros((8,4))).astype(int)
            m1, m2 = (3*xy[0,:]+xy[1,:])/4, np.mean(xy[:2,:],axis=0)
            m3, m4 = (xy[0,:]+3*xy[1,:])/4, np.mean(xy[1:3,:],axis=0)
            m5, m6 = (3*xy[2,:]+xy[3,:])/4, np.mean(xy[2:,:],axis=0)
            m7, m8 = (xy[2,:]+3*xy[3,:])/4, np.mean(xy[[0,3],:],axis=0)
            m9, m10 = (3*xy[0:]+xy[1:]+xy[2:]+3*xy[3:])/8, np.mean(xy,axis=0)
            m11 = (xy[0,:]+3*xy[1,:]+3*xy[2,:]+xy[3,:])/8
            # 
            scX = np.vstack((xy,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11))
            scInd = np.array([[0,4,12,11], [4,5,13,12], [5,6,14,13], [6,1,7,14],\
                [11,12,10,3], [12,13,9,10], [13,14,8,9], [14,7,2,8]])
        
        return scX, scInd
    # 
    def getSubCellCoord_edge(self,wkInd,wkX,tgt_Edge):
        # INITIALISE
        gcoord = np.array(np.zeros((5,2))).astype(float)
        node_sc = np.array(np.zeros(3)).astype(int)
        subX = np.array(np.zeros((3,2))).astype(float)

        # SUBCELL COORDINATES
        gcoord[:-1,:] = np.copy(wkX)
        gcoord[-1,:] = np.mean(wkX,axis=0)
        
        # SUBCELL NODE NUMBERING
        if (sum(tgt_Edge == wkInd[:2]) > 1): node_sc = np.array([0,1,4])
        elif (sum(tgt_Edge == wkInd[1:3]) > 1): node_sc = np.array([1,2,4])
        elif (sum(tgt_Edge == wkInd[2:]) > 1): node_sc = np.array([2,3,4])
        else: node_sc = np.array([3,0,4])
        
        # CURRENT SUBCELL COORDINATES
        subX = gcoord[node_sc,:]

        return subX