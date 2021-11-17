import numpy as np
# 
class modelMeshGeneration:
    def getModelMesh2D(self,Class,modelParam):
        # LINEARLY SPACED NODES
        xc,yc = self.nodeSpace2D(modelParam.numEls,modelParam.Lx)
        
        # NODE NUMBERING
        v1,v2,v3,v4 = self.nodeNumbering2D(modelParam.numEls)

        # NODAL COORDINATES
        Class.Els.getNodes(Class,modelParam,xc,yc)

        # ELEMENT CONNECTIVITY
        Class.Els.getElements(Class,modelParam,v1,v2,v3,v4)

        modelParam.DOFs = 2*modelParam.numNodes
    # 
    def nodeSpace2D(self,numEls,Lx):
        nnode = numEls + 1
        xc,yc = np.meshgrid(np.linspace(Lx[0,0],Lx[0,1],nnode[0]),\
            np.linspace(Lx[1,0],Lx[1,1],nnode[1]))
        xc,yc = np.ravel(xc), np.ravel(yc)
        return xc, yc
    # 
    def nodeNumbering2D(self,numEls):
        nnode = numEls + 1
        idx = np.reshape(np.arange(0,np.prod([nnode[1],nnode[0]])),[nnode[1],nnode[0]]).astype(int)
        v1, v2, v3, v4 = idx[:-1,:-1], idx[:-1,1:], idx[1:,:-1], idx[1:,1:]
        v1, v2, v3, v4 = np.ravel(v1), np.ravel(v2), np.ravel(v3), np.ravel(v4)
        return v1, v2, v3, v4
