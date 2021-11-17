import numpy as np
# 
class modelElement1D_L2:
    def getShapeFuncs(self,Pt):
        # INITIALISE
        N = np.array(np.zeros(2)).astype(float)
        dNdxi = np.array(np.zeros(2)).astype(float)

        # SHAPE FUNCS
        N[0], N[1] = 0.5*(1-Pt), 0.5*(1+Pt)

        # SHAPE FUNCS DERIVS
        dNdxi[0], dNdxi[1] = -0.5, 0.5

        self.N, self.dNdxi = N, dNdxi