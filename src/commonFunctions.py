import numpy as np
import numpy.linalg as la
from scipy.sparse import coo_matrix
# 
class commonFunctions:
    def getLinearConstitutive2D(self,modelParam):
        # INITIALISE
        E, nu = modelParam.matParams[0], modelParam.matParams[1]
        modelParam.Cmat = np.array(np.zeros((3,3))).astype(float)

        # CONSTITUTIVE MODEL
        if modelParam.stressType == "plane_stn":
            D = E/(1+nu)/(1-2*nu)
            modelParam.Cmat[0,0] = modelParam.Cmat[1,1] = 1.0 - nu
            modelParam.Cmat[0,1] = modelParam.Cmat[1,0] = nu
            modelParam.Cmat[2,2] = 0.5*(1.0-2*nu)
            modelParam.Cmat *= D
        else:
            D = E/(1-nu*nu)
            modelParam.Cmat[0,0] = modelParam.Cmat[1,1] = 1.0
            modelParam.Cmat[0,1] = modelParam.Cmat[1,0] = nu
            modelParam.Cmat[2,2] = 0.5*(1-nu)
            modelParam.Cmat *= D
    # 
    def getNonlinearConstitutive2D(self,modelParam,Fmat):
        # RIGHT CAUCHY-GREEN STRAIN TENSOR
        C = np.array(np.zeros((3,3))).astype(float)
        C = np.dot(Fmat.T,Fmat)
        invC = la.inv(C)

        # INVARIANTS
        I1, I2, I3 = np.trace(C), 0.5*(np.trace(C)**2-np.trace(C**2)), la.det(C)

        # CONSTITUTIVE TENSOR
        if modelParam.matType == "NH":
            Cmat,PK2,W = self.matNH(modelParam.matParams,I1,I3,invC)
            Smat = np.array(np.zeros((4,4))).astype(float)
            Smat[:2,:2] = Smat[2:,2:] = PK2[:2,:2]

        return Cmat, Smat, W
    # 
    def matNH(self,matParams,I1,I3,invC):
        # INITIALISE
        PK2, Cmat = 0.0, np.array(np.zeros((3,3))).astype(float)

        # LAMÉ'S PARAMETERS
        lmbda = matParams[1] - 2/3*matParams[0]
        mu = matParams[0] - lmbda/2*np.log(I3)
        Id = np.eye(3,dtype='float')

        # PK2 STRESS
        PK2 = np.array(np.zeros((2,2))).astype(float)
        PK2 = lmbda/2*np.log(I3)*invC + matParams[0]*(Id - invC)

        # 4TH-ORDER CONSTITUTIVE TENSOR
        Cmat[0,0] = lmbda*invC[0,0]*invC[0,0] + mu*(invC[0,0]*invC[0,0]+invC[0,0]*invC[0,0])
        Cmat[0,1] = Cmat[1,0] = lmbda*invC[0,0]*invC[1,1] + mu*(invC[0,1]*invC[0,1]+invC[0,1]*invC[0,1])
        Cmat[0,2] = Cmat[2,0] = lmbda*invC[0,0]*invC[0,1] + mu*(invC[0,0]*invC[0,1]+invC[0,1]*invC[0,0])
        Cmat[1,1] = lmbda*invC[1,1]*invC[1,1] + mu*(invC[1,1]*invC[1,1]+invC[1,1]*invC[1,1])
        Cmat[1,2] = Cmat[2,1] = lmbda*invC[1,1]*invC[0,1] + mu*(invC[1,0]*invC[1,1]+invC[1,1]*invC[1,0])
        Cmat[2,2] = lmbda*invC[0,1]*invC[0,1] + mu*(invC[0,0]*invC[1,1]+invC[0,1]*invC[1,0])

        # STRAIN ENERGY
        W = lmbda/8.0*(np.log(I3))**2 - matParams[0]/2.0*np.log(I3) + matParams[0]/2.0*(I1-3.0)

        return Cmat, PK2, W
    # 
    def writeVTK2D(self,modelParam,fname):
        # INITIALISE
        _U = modelParam.U
        Nodes = np.insert(modelParam.Nodes,2,0.0,axis=1)
        Elements = np.copy(modelParam.Elements)
        
        out = open(fname,'w')
        if modelParam.elemType == "Q4": numVertexPerCell, vtkCellCode = 4, 9
        else: numVertexPerCell, vtkCellCode = 3, 5
        dof_per_vertex = 2

        # WRITE HEAD
        out.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        out.write('\t<UnstructuredGrid>\n')
        out.write('\t\t<Piece NumberOfPoints="{:d}" NumberOfCells="{:d}">\n'.format(modelParam.numNodes,modelParam.numElems))

        # WRITE POINT DATA
        out.write('\t\t\t<Points>\n')
        out.write('\t\t\t\t<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for i in range(modelParam.numNodes):
            out.write('\t\t\t\t\t{:f}\t{:f}\t{:f}\n'.format(Nodes[i,0],Nodes[i,1],Nodes[i,2]))
        out.write('\t\t\t\t</DataArray>\n')
        out.write('\t\t\t</Points>\n')
        del Nodes

        # WRITE CELL DATA
        out.write('\t\t\t<Cells>\n')
        out.write('\t\t\t\t<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        if modelParam.elemType == "Q4":
            for i in range(modelParam.numElems):
                out.write('\t\t\t\t\t{:d}\t{:d}\t{:d}\t{:d}\n'.format(Elements[i,0],Elements[i,1],Elements[i,2],Elements[i,3]))
        else:
            for i in range(modelParam.numElems):
                out.write('\t\t\t\t\t{:d}\t{:d}\t{:d}\n'.format(Elements[i,0],Elements[i,1],Elements[i,2]))
        out.write('\t\t\t\t</DataArray>\n')
        del Elements

        # WRITE CELL OFFSETS
        out.write('\t\t\t\t<DataArray type="Int32" Name="offsets" format="ascii">\n')
        offsets = 0
        for i in range(modelParam.numElems):
            offsets += numVertexPerCell
            out.write('\t\t\t\t\t{:d}\n'.format(offsets))
        out.write('\t\t\t\t</DataArray>\n')
        del offsets

        # WRITE CELL TYPE
        out.write('\t\t\t\t<DataArray type="UInt8" Name="types" format="ascii">\n')
        for i in range(modelParam.numElems):
            out.write('\t\t\t\t\t{:d}\n'.format(vtkCellCode))
        out.write('\t\t\t\t</DataArray>\n')
        out.write('\t\t\t</Cells>\n')
        del vtkCellCode

        # WRITE DISPLACEMENTS
        out.write('\t\t\t<PointData Vectors="Res">\n')
        if hasattr(modelParam,'U'):
            out.write('\t\t\t\t<DataArray type="Float64" Name="U" NumberOfComponents="3" format="ascii">\n')
            for i in range(modelParam.numNodes):
                out.write('\t\t\t\t\t{:f}\t{:f}\t{:f}\n'.format(_U.item(2*i),_U.item(2*i+1),0.0))
            out.write('\t\t\t\t</DataArray>\n')
        del _U

        # WRITE STRAINS
        if hasattr(modelParam,'E'):
            out.write('\t\t\t\t<DataArray type="Float64" Name="E" NumberOfComponents="3" format="ascii">\n')
            for i in range(modelParam.numNodes):
                out.write('\t\t\t\t\t{:f}\t{:f}\t{:f}\n'.format(modelParam.E[i,0],modelParam.E[i,1],modelParam.E[i,2]))
            out.write('\t\t\t\t</DataArray>\n')
        
        # WRITE STRESSES
        if hasattr(modelParam,'S'):
            out.write('\t\t\t\t<DataArray type="Float64" Name="S" NumberOfComponents="3" format="ascii">\n')
            for i in range(modelParam.numNodes):
                out.write('\t\t\t\t\t{:f}\t{:f}\t{:f}\n'.format(modelParam.S[i,0],modelParam.S[i,1],modelParam.S[i,2]))
            out.write('\t\t\t\t</DataArray>\n')
        
        # WRITE VON MISES STRESS
        if hasattr(modelParam,'Svm'):
            out.write('\t\t\t\t<DataArray type="Float64" Name="Svm" NumberOfComponents="1" format="ascii">\n')
            for i in range(modelParam.numNodes):
                out.write('\t\t\t\t\t{:f}\n'.format(modelParam.Svm[i]))
            out.write('\t\t\t\t</DataArray>\n')
        out.write('\t\t\t</PointData>\n')

        # CLOSE VTK FILE
        out.write('\t\t</Piece>\n')
        out.write('\t</UnstructuredGrid>\n')
        out.write('</VTKFile>')
        out.close()
    # 
    def polyArea(self,x):
        n, area = len(x), 0.0
        j = list(map(lambda i: (i+1)%n, range(n)))
        a0 = list(map(lambda i: x[i,0]*x[j[i],1], range(n)))
        area += np.sum(a0)
        a1 = list(map(lambda i: x[j[i],0]*x[i,1], range(n)))
        area -= np.sum(a1)
        area = np.absolute(area)/2.0
        self.area = area
        return self.area
    # 
    def indices(self,a,func):
        self.aInd = [i for (i, val) in enumerate(a) if func(val)]
        return self.aInd
    # 
    def columns(self,matrix,i):
        self.aCol = [row[i] for row in matrix]
        return self.aCol