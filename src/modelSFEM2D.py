import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class modelSFEM2D:
    def getLinearStiffness2D(self,Class,modelParam):
        # COMPUTE STIFFNESS
        Class.SDs.getLinearSmoothedStiffness(Class,modelParam)
        
        modelParam.K = Class.SDs.K
        del Class.SDs.K
    # 
    def getNonlinearStiffness2D(self,Class,modelParam):
        # INITIALISE
        numSteps, maxiter, Tol = 50, 80, 1e-9
        modelParam.U = np.array(np.zeros(modelParam.DOFs)).astype(float)
        modelParam.stnE = np.array(np.zeros(numSteps)).astype(float)

        # LOAD CONTROL
        for ist in range(numSteps):
            scale = np.divide(ist+1,numSteps,dtype='float')
            niter, condition = 0, 1.0
            print('\t\tStep {:d} ({:7.3f}%)'.format(ist+1,100*scale))

            # NEWTON-RAPSHON ITERATION
            while (condition > Tol) and (niter <= maxiter):
                niter += 1
                # COMPUTE STIFFNESS
                Class.SDs.getNonlinearSmoothedStiffness(Class,modelParam)

                # SOLVE NONLINEAR SYSTEM
                Class.Solve.solveNonlinearSystem(modelParam,scale)

                # CHECK CONDITIONS
                condition = np.sqrt(np.divide(np.vdot(modelParam.dU,modelParam.dU),np.vdot(modelParam.U,modelParam.U)))
                resi_cond = np.divide(np.sqrt(np.vdot(modelParam.b,modelParam.b)),modelParam.DOFs)
                print('\t\t\tIter. {:d}\tCond. {:f}\tResi. {:f}\tTol. {:f}'.format(niter,condition,resi_cond,Tol))
                del modelParam.dU, modelParam.b
            
            modelParam.stnE[ist] = modelParam.EE
    #
    def getLinearBmat(self,bxy,ndof):
        # INITIALISE
        Bmat = np.array(np.zeros((3,2*ndof))).astype(float)

        # LINEAR STRAIN-DISPLACEMENT MATRIX
        Bmat[0,0::2] = bxy[:,0]
        Bmat[1,1::2] = bxy[:,1]
        Bmat[2,0::2] = bxy[:,1]
        Bmat[2,1::2] = bxy[:,0]

        return Bmat
    # 
    def getNonlinearBmat(self,bxy,U,ndof):
        # INITIALISE
        dxdX = np.array(np.zeros((2,2))).astype(float)
        Fmat = np.array(np.eye(3,dtype=float))
        Bmat = np.array(np.zeros((3,2*ndof))).astype(float)
        Bgeo = np.array(np.zeros((4,2*ndof))).astype(float)

        # DEFORMATION GRADIENT
        dxdX = np.dot(U,bxy)
        Fmat[:2,:2] += dxdX

        # STRAIN-DISPLACEMENT MATRIX: MATERIAL PART
        Bmat[0,0::2] = bxy[:,0]*Fmat[0,0]
        Bmat[0,1::2] = bxy[:,0]*Fmat[1,0]
        Bmat[1,0::2] = bxy[:,1]*Fmat[0,1]
        Bmat[1,1::2] = bxy[:,1]*Fmat[1,1]
        Bmat[2,0::2] = bxy[:,1]*Fmat[0,0] + bxy[:,0]*Fmat[0,1]
        Bmat[2,1::2] = bxy[:,0]*Fmat[1,1] + bxy[:,1]*Fmat[1,0]

        # STRAIN-DISPLACEMENT MATRIX: GEOMETRY PART
        Bgeo[0,0::2] = Bgeo[2,1::2] = bxy[:,0]
        Bgeo[1,0::2] = Bgeo[3,1::2] = bxy[:,1]

        return Fmat, Bmat, Bgeo
    # 
    def getSmoothedShapeFuncs(self,Class,modelParam,wkX,subX,ndof,bx,by):
        # OUTWARD NORMAL VECTORS
        self.getOutwardNormals(modelParam,subX[:,0],subX[:,1],ndof)

        # SMOOTHED SHAPE FUNCS
        bx,by = self.getShapeFuncs(Class,modelParam,wkX,subX,ndof,bx,by)

        return bx, by
    # 
    def getShapeFuncs(self,Class,modelParam,wkX,subX,ndof,bx,by):
        # INITIALISE
        ns = len(subX)
        
        # SMOOTHING DOMAIN BOUNDARY LOOP
        for isc in range(ndof):
            # GAUSS POINT LOOP
            for ig in range(modelParam.ng):
                # SHAPE FUNCS ON EACH BOUDNARY
                Class.Bound.getShapeFuncs(modelParam.Q)

                # GAUSS POINT IN LOCAL COORD
                xi,eta = Class.El.computeXiEta2XY(isc,Class.Bound.N)

                # MAP GLOBAL GAUSS PTS TO LOCAL
                ksi = np.array(np.zeros(2)).astype(float)
                ksi = np.hstack((xi,eta))
                Class.El.getShapeFuncs(ksi)
                xy_g = np.array(np.zeros(2)).astype(float)
                xy_g = np.dot(Class.El.N.T,subX)
                xieta = Class.SFEM.computeXiEta(Class,modelParam,wkX,xy_g)

                # SHAPE FUNCS
                Class.Els.getShapeFuncs(xieta)

                # JACOBIAN
                J = self.side[isc]/2

                # SHAPE FUNCS SET
                bx += (self.nxy[isc,0]*Class.Els.N*J*modelParam.W[ig])
                by += (self.nxy[isc,1]*Class.Els.N*J*modelParam.W[ig])

        return bx, by
    # 
    def combineSmoothedShapeFuncs(self,bx,by,bxy,wkInd,nodL,ndof,nn,cnt):
        # INITIALISE

        if cnt == 0:
            nn = ndof
            bxy[:,0], bxy[:,1] = bx, by
        else:
            i0 = -1
            for jj in range(ndof):
                nod = wkInd[jj]
                flag = 0
                for j in range(nn):
                    if nodL[j] == nod:
                        bxy[j,0] += bx[jj]
                        bxy[j,1] += by[jj]
                        flag = 1
                        break
                if flag == 0:
                    i0 += 1
                    nodL = np.insert(nodL,nn+i0,nod)
                    bxy = np.concatenate((bxy,np.array([[bx[jj],by[jj]]])),axis=0)
            nn = len(nodL)

        return bxy, nodL, nn
    # 
    def computeXiEta(self,Class,modelParam,wkX,xy_g):
        # INITIALISE
        nodes = np.copy(wkX)
        xieta = np.array(np.zeros(2)).astype(float)
        it, inc = 10, 0

        # COMPUTE COORD
        while inc < it:
            # INITIALISE
            xy = np.array(np.zeros(2)).astype(float)
            dxdxi = np.array(np.zeros((2,2))).astype(float)
            delta = np.array(np.zeros(2)).astype(float)
            F = np.array(np.zeros((2,2))).astype(float)

            # SHAPE FUNCS
            Class.Els.getShapeFuncs(xieta)

            xy = np.dot(Class.Els.N.T,nodes)
            dxdxi = np.dot(Class.Els.dNdxi.T,nodes)
            delta = xy - xy_g
            F = dxdxi.T
            invF = la.inv(F)
            xieta -= np.dot(delta,invF.T)
            inc += 1

        return xieta
    # 
    def getOutwardNormals(self,modelParam,x,y,ndof):
        # INITIALISE
        side = np.array(np.zeros(ndof)).astype(float)
        nxy = np.array(np.zeros((ndof,2))).astype(float)
        if len(x) == 4: ind = np.array([[0,1],[1,2],[2,3],[3,0]]).astype(int)
        else: ind = np.array([[0,1],[1,2],[2,0]]).astype(int)

        # EACH BOUNDARY LENGTH & NORMAL VECTOR
        for ie in range(len(x)):
            side[ie] = np.sqrt((x[ind[ie,0]]-x[ind[ie,1]])**2 + \
                (y[ind[ie,0]]-y[ind[ie,1]])**2)
            # 
            nxy[ie,0] =  (y[ind[ie,1]]-y[ind[ie,0]])/side[ie]
            nxy[ie,1] = -(x[ind[ie,1]]-x[ind[ie,0]])/side[ie]
        
        self.side, self.nxy = side, nxy
    # 
    def getGaussQuadrature(self,modelParam):
        # INITIALISE
        modelParam.Q = np.array(np.zeros(modelParam.ng)).astype(float)
        modelParam.W = np.array(np.zeros(modelParam.ng)).astype(float)

        if modelParam.ng == 1: 
            modelParam.Q[0], modelParam.W[0] = 0.0, 2.0
        else: 
            modelParam.Q[0], modelParam.Q[1] = 0.577350269189626, -0.577350269189626
            modelParam.W[0] = modelParam.W[1] = 1.0