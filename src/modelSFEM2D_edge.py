import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class modelSFEM2D_edge:
    def getLinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype=float).toarray()

        # TARGET EDGES & NEIGHBOURING ELEMENTS
        self.getTargetEdge(Class,modelParam)

        # TARGET EDGE LOOP
        for ivo in range(modelParam.numEdge):
            # NEIGHBOURING CELLS OF TARGET EDGE
            targets = modelParam.targetEdge[ivo,2:]
            neighbour = targets[np.where(targets >= 0)]
            nc = 0; nc = len(neighbour)
            tgt_Edge = modelParam.targetEdge[ivo,:2]

            # CURRENT SMOOTHING DOMAIN AREA
            subA = modelParam.subA[ivo]

            # SMOOTHING DOMAIN STIFFNESS
            Ksub,nodL = self.getLinearSmoothingDomainStiffness(Class,modelParam,neighbour,tgt_Edge,subA,)

            # ASSEMBLE GLOBAL STIFFNESS
            ndof = len(nodL)
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*nodL, 2*nodL+1
            K[np.ix_(edof,edof)] += Ksub
        
        self.K = K
    # 
    def getNonlinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype='float').toarray()
        R = np.array(np.zeros(modelParam.DOFs)).astype(float)
        modelParam.EE = 0.0

        # TARGET EDGES & NEIGHBOURING ELEMENTS
        self.getTargetEdge(Class,modelParam)

        # TARGET EDGE LOOP
        for ivo in range(modelParam.numEdge):
            # NEIGHBOURING CELLS OF TARGET EDGE
            targets = modelParam.targetEdge[ivo,2:]
            neighbour = targets[np.where(targets >= 0)]
            nc = 0; nc = len(neighbour)
            tgt_Edge = modelParam.targetEdge[ivo,:2]

            # CURRENT SMOOTHING DOMAIN AREA
            subA = modelParam.subA[ivo]

            # SMOOTHING DOMAIN STIFFNESS
            modelParam,Ksub,Rsub,nodL = self.getNonlinearSmoothingDomainStiffness(Class,modelParam,neighbour,tgt_Edge,subA)

            # ASSEMBLE GLOBAL STIFFNESS
            ndof = len(nodL)
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*nodL, 2*nodL+1
            K[np.ix_(edof,edof)] += Ksub
            R[edof] += Rsub
        
        modelParam.K, modelParam.R = K, R
        del K, R
    # 
    def getLinearSmoothingDomainStiffness(self,Class,modelParam,neighbour,tgt_Edge,subA):
        # COMPUTE SHAPE FUNCS ON BOUNDARIES OF SMOOTHING DOMAIN
        bxy,nodL = self.getSmoothingDomainShapeFuncs(Class,modelParam,neighbour,tgt_Edge,subA)
        ndof = len(nodL)

        # SMOOTHED STRAIN-DISPLACEMENT MATRIX
        Bmat = Class.SFEM.getLinearBmat(bxy,ndof)

        # SMOOTHING DOMAIN STIFFNESS
        Ksub = coo_matrix((2*ndof,2*ndof),dtype=float).toarray()
        BTC, BTCB = np.array(np.zeros((2*ndof,3))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
        BTC = np.dot(Bmat.T,modelParam.Cmat)
        BTCB = np.dot(BTC,Bmat)
        Ksub += BTCB*subA

        return Ksub,nodL
    # 
    def getNonlinearSmoothingDomainStiffness(self,Class,modelParam,neighbour,tgt_Edge,subA):
        # COMPUTE SHAPE FUNCS ON BOUNDARIES OF SMOOTHING DOMAIN
        bxy,nodL = self.getSmoothingDomainShapeFuncs(Class,modelParam,neighbour,tgt_Edge,subA)
        ndof = len(nodL)

        # CURRENT ELEMENT DISPLACEMENTS
        Uel = np.array(np.zeros((2,ndof))).astype(float)
        Uel[0,:], Uel[1,:] = modelParam.U[2*nodL], modelParam.U[2*nodL+1]

        # SMOOTHED NONLINEAR STRAIN-DISPLACEMENT MATRIX
        Fmat,Bmat,Bgeo = Class.SFEM.getNonlinearBmat(bxy,Uel,ndof)

        # 4TH-ORDER CONSTITUTIVE MATRIX
        Cmat,Smat,W = Class.Common.getNonlinearConstitutive2D(modelParam,Fmat)

        # STIFFNESS MATRIX
        Ksub = coo_matrix((2*ndof,2*ndof),dtype=float).toarray()
        BTC, BTCB = np.array(np.zeros((2*ndof,3))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
        BTS, BTSB = np.array(np.zeros((2*ndof,4))).astype(float), np.array(np.zeros((2*ndof,2*ndof))).astype(float)
        BTC, BTS = np.dot(Bmat.T,Cmat), np.dot(Bgeo.T,Smat)
        BTCB, BTSB = np.dot(BTC,Bmat), np.dot(BTS,Bgeo)
        Ksub += (BTCB+BTSB)*subA

        # RESIDUAL VECTOR
        Rsub = np.array(np.zeros(2*ndof)).astype(float)
        BTS0, S0 = np.array(np.zeros(2*ndof)).astype(float), np.array(np.zeros(3)).astype(float)
        S0 = np.array([Smat[0,0], Smat[1,1], Smat[0,1]])
        BTS0 = np.dot(Bmat.T,S0)
        Rsub += BTS0*subA

        # STRAIN ENERGY
        modelParam.EE += W*subA

        return modelParam, Ksub, Rsub, nodL
    # 
    def getSmoothingDomainShapeFuncs(self,Class,modelParam,neighbour,tgt_Edge,subA):
        # INITIALISE
        cnt, nn = 0, []

        for ic in neighbour:
            # CURRENT ELEMENT NODE NUMBERING
            wkInd = np.array(np.zeros(modelParam.NPE)).astype(int)
            wkInd = modelParam.Elements[ic,:]
            ndof = len(wkInd)

            # CURRENT ELEMENT COORDINATES
            wkX = np.array(np.zeros((modelParam.NPE,2))).astype(float)
            wkX = modelParam.Nodes[wkInd,:]

            # CURRENT SUBCELL COORDINATES
            subX = Class.Els.getSubCellCoord_edge(wkInd,wkX,tgt_Edge)

            # SMOOTHED SHAPE FUNCS ON EACH SMOOTHING DOMAIN
            bx, by = np.array(np.zeros(ndof)).astype(float), np.array(np.zeros(ndof)).astype(float)
            bx, by = Class.SFEM.getSmoothedShapeFuncs(Class,modelParam,wkX,subX,ndof,bx,by)
            bx /= subA; by /= subA

            # COMBINE SHAPE FUNCS ON SMOOTHING DOMAIN
            if cnt == 0:
                bxy = np.array(np.zeros((ndof,2))).astype(float)
                nodL = np.copy(wkInd)
            bxy,nodL,nn = Class.SFEM.combineSmoothedShapeFuncs(bx,by,bxy,wkInd,nodL,ndof,nn,cnt)

            cnt += 1

        return bxy, nodL
    # 
    def getTargetEdge(self,Class,modelParam):
        # TARGET EDGE
        targetEdge = self.getEdges(modelParam)

        # COMPUTE SMOOTHING DOMAIN AREA
        triA = np.array(np.zeros(modelParam.numElems)).astype(float)
        wkInd = list(map(lambda x: modelParam.Elements[x,:modelParam.NPE], range(modelParam.numElems)))
        wkX = list(map(lambda x: modelParam.Nodes[wkInd[x],:], range(modelParam.numElems)))
        triA = list(map(lambda x: Class.Common.polyArea(wkX[x]), range(modelParam.numElems)))
        # 
        numEdge = len(targetEdge)
        subA = np.array(np.zeros(numEdge)).astype(float)
        for i in range(numEdge):
            iedge = Class.Common.indices(targetEdge[i,2:], lambda x: x>= 0)
            for j in iedge:
                subA[i] += triA[j]/modelParam.NPE
        
        modelParam.targetEdge = targetEdge
        modelParam.numEdge = len(targetEdge)
        modelParam.subA = subA

        del targetEdge, triA, subA, wkInd, wkX
    #
    def getEdges(self,modelParam):
        # INITIALISE
        conn = modelParam.Elements

        if modelParam.elemType == "Q4":
            edges = np.array([[conn[0,0],conn[0,1],0,-1],[conn[0,1],conn[0,2],0,-1],[conn[0,2],conn[0,3],0,-1],[conn[0,0],conn[0,3],0,-1]]).astype(int)
            npe = 3
        else:
            edges = np.array([[conn[0,0],conn[0,1],0,-1],[conn[0,1],conn[0,2],0,-1],[conn[0,0],conn[0,2],0,-1]]).astype(int)
            npe = 2

        # GET EDGES
        for i in range(1,modelParam.numElems):
            for j in range(npe+1):
                n1 = j
                if n1 == npe: n2 = 0
                else: n2 = n1 + 1
                flag = 0
                for m in range(len(edges)):
                    if (conn[i,n1]==edges[m,0] and conn[i,n2]==edges[m,1]) or \
                        (conn[i,n2]==edges[m,0] and conn[i,n1]==edges[m,1]):
                        flag = 1
                        edges[m,3] = i
                        break
                if flag == 0:
                    idx = np.array([[conn[i,n1],conn[i,n2],i,-1]]).astype(int)
                    edges = np.concatenate((edges,idx),axis=0)

        targetEdge = np.copy(edges); del edges
        return targetEdge