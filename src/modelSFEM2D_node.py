import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class modelSFEM2D_node:
    def getLinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype=float).toarray()

        # TARGET NODES & NEIGHBOURING ELEMENTS
        self.getTargetNode(Class,modelParam)

        # TARGET NODE LOOP
        for ivo in range(modelParam.numNodes):
            # NEIGHBOURING CELLS OF TARGET NODE
            neighbour = modelParam.targetNode[ivo]
            
            # CURRENT SMOOTHING DOMAIN AREA
            subA = modelParam.subA[ivo]

            # SMOOTHING DOMAIN STIFFNESS
            Ksub,nodL = self.getLinearSmoothingDomainStiffness(Class,modelParam,neighbour,subA,ivo)

            # ASSEMBLE GLOBAL STIFFNESS
            ndof = len(nodL)
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*nodL, 2*nodL+1
            K[np.ix_(edof,edof)] += Ksub
        
        self.K = K
        del K,neighbour,subA,Ksub,nodL,edof
    # 
    def getNonlinearSmoothedStiffness(self,Class,modelParam):
        # INITIALISE
        K = coo_matrix((modelParam.DOFs,modelParam.DOFs),dtype=float).toarray()
        R = np.array(np.zeros(modelParam.DOFs)).astype(float)
        modelParam.EE = 0.0

        # TARGET NODES & NEIGHBOURING ELEMENTS
        self.getTargetNode(Class,modelParam)

        # TARGET NODE LOOP
        for ivo in range(modelParam.numNodes):
            # NEIGHBOURING CELLS OF TARGET NODE
            neighbour = modelParam.targetNode[ivo]
            
            # CURRENT SMOOTHING DOMAIN AREA
            subA = modelParam.subA[ivo]

            # SMOOTHING DOMAIN STIFFNESS
            modelParam,Ksub,Rsub,nodL = self.getNonlinearSmoothingDomainStiffness(Class,modelParam,neighbour,subA,ivo)

            # ASSEMBLE GLOBAL STIFFNESS
            ndof = len(nodL)
            edof = np.array(np.zeros(2*ndof)).astype(int)
            edof[0::2], edof[1::2] = 2*nodL, 2*nodL+1
            K[np.ix_(edof,edof)] += Ksub
            R[edof] += Rsub
        
        modelParam.K, modelParam.R = K, R
        del K,R,neighbour,subA,Ksub,Rsub,nodL,edof
    # 
    def getLinearSmoothingDomainStiffness(self,Class,modelParam,neighbour,subA,ivo):
        # COMPUTE SHAPE FUNCS ON BOUNDARIES OF SMOOTHING DOMAIN
        bxy,nodL = self.getSmoothingDomainShapeFuncs(Class,modelParam,neighbour,subA,ivo)
        ndof = len(nodL)

        # SMOOTHED STRAIN-DISPLACEMENT MATRIX
        Bmat = Class.SFEM.getLinearBmat(bxy,ndof)

        # SMOOTHING DOMAIN STIFFNESS
        Ksub = coo_matrix((2*ndof,2*ndof),dtype=float).toarray()
        BTC,BTCB = np.array(np.zeros((2*ndof,3))).astype(float),np.array(np.zeros((2*ndof,2*ndof))).astype(float)
        BTC = np.dot(Bmat.T,modelParam.Cmat)
        BTCB = np.dot(BTC,Bmat)
        Ksub += BTCB*subA

        return Ksub, nodL
    # 
    # 
    def getNonlinearSmoothingDomainStiffness(self,Class,modelParam,neighbour,subA,ivo):
        # COMPUTE SHAPE FUNCS ON BOUNDARIES OF SMOOTHING DOMAIN
        bxy,nodL = self.getSmoothingDomainShapeFuncs(Class,modelParam,neighbour,subA,ivo)
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
    def getSmoothingDomainShapeFuncs(self,Class,modelParam,neighbour,subA,ivo):
        # INITIALISE
        cnt, nn = 0, []

        for ic in neighbour:
            # CURRENT ELEMENT NODE NUMBERING
            wkInd = np.array(np.zeros(modelParam.NPE)).astype(int)
            wkInd = modelParam.Elements[ic,:]
            ndof = len(wkInd)

            # CURRENT ELEMENT COORIDNATES
            wkX = np.array(np.zeros((modelParam.NPE,2))).astype(float)
            wkX = modelParam.Nodes[wkInd,:]

            # CURRENT SUBCELL COORDINATES
            scX,scInd = Class.Els.getSubCellCoord_node(ivo,wkInd,wkX)

            # SMOOTHED SHAPE FUNCS ON EACH SMOOTHING DOMAIN
            bx,by = np.array(np.zeros(ndof)).astype(float), np.array(np.zeros(ndof)).astype(float)
            for icel in range(len(scInd)):
                subX = np.array(np.zeros((ndof,2))).astype(float)
                subX = scX[scInd[icel,:],:]
                bx,by = Class.SFEM.getSmoothedShapeFuncs(Class,modelParam,wkX,subX,ndof,bx,by)
            bx /= subA; by /= subA

            # COMBINE SHAPE FUNCS ON SMOOTHING DOMAIN
            if cnt == 0:
                bxy = np.array(np.zeros((ndof,2))).astype(float)
                nodL = np.copy(wkInd)
            bxy,nodL,nn = Class.SFEM.combineSmoothedShapeFuncs(bx,by,bxy,wkInd,nodL,ndof,nn,cnt)

            cnt += 1

        return bxy, nodL
    # 
    def getTargetNode(self,Class,modelParam):
        # TARGET NODE
        targetNode = self.getNodes(modelParam)

        # COMPUTE SMOOTHING DOMAIN AREA
        wkInd = list(map(lambda x: modelParam.Elements[x,:modelParam.NPE], range(modelParam.numElems)))
        wkX = list(map(lambda x: modelParam.Nodes[wkInd[x],:], range(modelParam.numElems)))
        triA = list(map(lambda x: Class.Common.polyArea(wkX[x]), range(modelParam.numElems)))
        # 
        subA = np.array(np.zeros(modelParam.numNodes)).astype(float)
        for i in range(modelParam.numNodes):
            for j in range(modelParam.numElems):
                if bool(np.any(i==modelParam.Elements[j,:])) and True:
                    subA[i] += triA[j]/modelParam.NPE

        modelParam.targetNode = targetNode
        modelParam.subA = subA

        del targetNode, triA, subA, wkInd, wkX
    # 
    def getNodes(self,modelParam):
        # INITIALISE
        nodes = np.array(np.zeros((modelParam.numNodes),dtype=object))

        for i in range(modelParam.numNodes):
            ind = -1
            nod = np.array([])
            for j in range(modelParam.numElems):
                if bool(np.any(i==modelParam.Elements[j,:])) and True:
                    ind += 1
                    nod = np.insert(nod,ind,j).astype(int)
                    nodes[i] = nod

        targetNode = np.copy(nodes); del nodes
        return targetNode
