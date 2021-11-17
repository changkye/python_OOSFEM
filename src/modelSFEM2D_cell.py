import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class modelSFEM2D_cell:
    def getLinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype=float).toarray()

        # ELEMENT LOOP
        for ivo in range(modelParam.numElems):
            # CURRENT ELEMENT NODE NUMBERING
            wkInd = np.array(np.zeros(modelParam.NPE)).astype(int)
            wkInd = modelParam.Elements[ivo,:]
            ndof = len(wkInd)

            # CURRENT ELEMENT COORDINATES
            wkX = np.array(np.zeros((modelParam.NPE,2))).astype(int)
            wkX = modelParam.Nodes[wkInd,:]

            # SMOOTHING DOMAIN STIFFNESS
            self.getLinearSmoothingDomainStiffness(Class,modelParam,wkInd,wkX,ndof)

            # ASSEMBLE GLOBAL STIFFNESS
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*wkInd, 2*wkInd+1
            K[np.ix_(edof,edof)] += self.Ksub

        self.K = K
    # 
    def getNonlinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype='float').toarray()
        R = np.array(np.zeros(modelParam.DOFs)).astype(float)
        modelParam.EE = 0.0

        # ELEMENT LOOP
        for ivo in range(modelParam.numElems):
            # CURRENT ELEMENT NODE NUMBERING
            wkInd = np.array(np.zeros(modelParam.NPE)).astype(int)
            wkInd = modelParam.Elements[ivo,:]
            ndof = len(wkInd)

            # CURRENT ELEMENT COORDINATES
            wkX = np.array(np.zeros((modelParam.NPE,2))).astype(int)
            wkX = modelParam.Nodes[wkInd,:]

            # CURRENT ELEMENT DISPLACEMENTS
            Uel = np.array(np.zeros((2,ndof))).astype(float)
            Uel[0,:], Uel[1,:] = modelParam.U[2*wkInd], modelParam.U[2*wkInd+1]

            # SMOOTHING DOMAIN STIFFNESS
            self.getNonlinearSmoothingDomainStiffness(Class,modelParam,wkInd,wkX,Uel,ndof)

            # ASSEMBLE GLOBAL STIFFNESS & RESIDUAL
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*wkInd, 2*wkInd+1
            K[np.ix_(edof,edof)] += self.Ksub
            R[edof] += self.Rsub

        modelParam.K, modelParam.R = K, R
        del K, R
    # 
    def getLinearSmoothingDomainStiffness(self,Class,modelParam,wkInd,wkX,ndof):
        # INITIALISE
        Ksub = coo_matrix((2*ndof,2*ndof),dtype=float).toarray()

        # SUBCELL LOOP
        for isc in range(modelParam.numSub):
            scX,scInd = Class.Els.getSubCellCoord_cell(modelParam,isc,wkX)
            # SMOOTHING DOMAIN LOOP
            for i in range(modelParam.numSD):
                bx, by = np.array(np.zeros(ndof)).astype(float), np.array(np.zeros(ndof)).astype(float)

                # CURRENT SMOOTHING DOMAIN NODE NUMBERING
                Ind = np.array(np.zeros(ndof)).astype(int)
                Ind = scInd[i,:]

                # CURRENT SMOOTHING DOMAIN COORDINATES
                subX = np.array(np.zeros((ndof,2))).astype(float)
                subX = scX[Ind,:]

                # SMOOTHING DOMAIN AREA
                subA = Class.Common.polyArea(subX)

                # SMOOTHED SHAPE FUNCS ON EACH SMOOTHING DOMAIN
                bx,by = Class.SFEM.getSmoothedShapeFuncs(Class,modelParam,wkX,subX,ndof,bx,by)
                bx /= subA; by /= subA

                # SMOOTHED LINEAR STRAIN-DISPLACEMENT MATRIX
                bxy = np.array(np.zeros((ndof,2))).astype(float)
                bxy[:,0], bxy[:,1] = bx, by
                Bmat = Class.SFEM.getLinearBmat(bxy,ndof)

                # STIFFNESS MATRIX
                BTC, BTCB = np.array(np.zeros((2*ndof,3))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
                BTC = np.dot(Bmat.T,modelParam.Cmat)
                BTCB = np.dot(BTC,Bmat)
                Ksub += BTCB*subA

        self.Ksub = Ksub
    # 
    def getNonlinearSmoothingDomainStiffness(self,Class,modelParam,wkInd,wkX,Uel,ndof):
        # INITIALISE
        Ksub = coo_matrix((2*ndof,2*ndof),dtype=float).toarray()
        Rsub = np.array(np.zeros(2*ndof)).astype(float)

        # SUBCELL LOOP
        for isc in range(modelParam.numSub):
            scX,scInd = Class.Els.getSubCellCoord_cell(modelParam,isc,wkX)
            # SMOOTHING DOMAIN LOOP
            for i in range(modelParam.numSD):
                bx,by = np.array(np.zeros(ndof)).astype(float),np.array(np.zeros(ndof)).astype(float)

                # CURRENT SMOOTHING DOMAIN NODE NUMBERING
                Ind = np.array(np.zeros(ndof)).astype(int)
                Ind = scInd[i,:]

                # CURRENT SMOOTHING DOMAIN COORDINATES
                subX = np.array(np.zeros((ndof,2))).astype(float)
                subX = scX[Ind,:]

                # SMOOTHING DOMAIN AREA
                subA = Class.Common.polyArea(subX)

                # SMOOTHED SHAPE FUNCS ON EACH SMOOTHING DOMAIN
                bx,by = Class.SFEM.getSmoothedShapeFuncs(Class,modelParam,wkX,subX,ndof,bx,by)
                bx /= subA; by /= subA

                # SMOOTHED NONLINEAR STRAIN-DISPLACEMENT MATRIX
                bxy = np.array(np.zeros((ndof,2))).astype(float)
                bxy[:,0], bxy[:,1] = bx, by
                Fmat,Bmat,Bgeo = Class.SFEM.getNonlinearBmat(bxy,Uel,ndof)

                # 4TH-ORDER CONSTITUTIVE MATRIX
                Cmat,Smat,W = Class.Common.getNonlinearConstitutive2D(modelParam,Fmat)

                # STIFFNESS MATRIX
                BTC, BTCB = np.array(np.zeros((2*ndof,3))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
                BTS, BTSB = np.array(np.zeros((2*ndof,4))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
                BTC, BTS = np.dot(Bmat.T,Cmat), np.dot(Bgeo.T,Smat)
                BTCB, BTSB = np.dot(BTC,Bmat), np.dot(BTS,Bgeo)
                Ksub += (BTCB+BTSB)*subA

                # RESIDUAL VECTOR
                BTS0, S0 = np.array(np.zeros(2*ndof)).astype(float), np.array(np.zeros(3)).astype(float)
                S0 = np.array([Smat[0,0], Smat[1,1], Smat[0,1]])
                BTS0 = np.dot(Bmat.T,S0)
                Rsub += BTS0*subA

                # STRAIN ENERGY
                modelParam.EE += W*subA
        
        self.Ksub, self.Rsub = Ksub, Rsub
        del Ksub, Rsub