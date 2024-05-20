import sys
from itertools import combinations
import numpy as np
import math as m

class T_plane:
    def __init__(self, a,b,c):
        self.triple = [a,b,c]
        self.T3 = self.generateTPlan(a,b,c)

    def getTriple(self):
        return self.triple

    def getT3(self):
        return self.T3
    
    def isEqualTo(self, Tplane):
        triple = Tplane.getTriple()
        esito_point_a = self.triple[0][0] == triple[0][0] and self.triple[0][1] == triple[0][1] and self.triple[0][2] == triple[0][2]
        esito_point_b = self.triple[1][0] == triple[1][0] and self.triple[1][1] == triple[1][1] and self.triple[1][2] == triple[1][2]
        esito_point_c = self.triple[2][0] == triple[2][0] and self.triple[2][1] == triple[2][1] and self.triple[2][2] == triple[2][2]
        return (esito_point_a and esito_point_b and esito_point_c)

    def __getNormalVec__(self, a, b, c):
        v = np.cross(b-a, c-a) # uscente dall'oggetto
        v = np.cross(c-a, b-a) # entrante nell'oggetto
        n = v / np.linalg.norm(v)
        return n

    def generateTPlan(self, a,b,c): # restituisce la terna di quel piano specifico, l'origine e' il punto a
        y = self.__getNormalVec__(a,b,c) # normal vector
        congiungVec = b - a
        z = congiungVec/np.linalg.norm(congiungVec)
        v = np.cross(y,z)
        x = v / np.linalg.norm(v)
        T = np.array([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
        T[0:3,0] = x
        T[0:3,1] = y
        T[0:3,2] = z
        T[0:3,3] = a
        return T

class G_transform:
    def __init__(self, a,b,c, nominalT):
        self.plane = T_plane(a,b,c)
        self.computeTransformation(self.plane, nominalT)

    def getPlane(self):
        return self.plane

    def getTg(self):
        return self.Tg

    def computeTransformation(self, Tplane, nominalT):
        self.Tg = np.linalg.inv(Tplane.getT3()).dot(nominalT)


class GraspingPoint:

    def __init__(self, gp, n3D, nK): # graspingPoint = gp, nominal3D = n3D, numKeypoints = nK
        #gp --> GRASPING POSE
        self.graspingPoint = gp
        self.transformations = []
        self.nominal_3D = n3D
        self.numKeypoints = nK

        self.__generatePlans__()

    def getGraspingPoint(self):
        return self.graspingPoint


    def getTransformations(self):
        return self.transformations
    
    # Returns the G_tranform Tg related to a plane that has a specific triplet (the triplet is specified through the keypoint indexes)
    def getGTranformWithSpecificTripleIdxes(self, ia,ib,ic):
        plane_temp = T_plane(self.nominal_3D[ia,:], self.nominal_3D[ib,:], self.nominal_3D[ic,:])
        for gTranform in self.transformations:
            if gTranform.getPlane().isEqualTo(plane_temp):
                return gTranform
        return -1
    
    def __generatePlans__(self):
        l = range(0,self.numKeypoints, 1)
        comb = combinations(l,3)
        for triple in list(comb):
            gt = G_transform(self.nominal_3D[triple[0],:], self.nominal_3D[triple[1],:], self.nominal_3D[triple[2],:], self.graspingPoint)
            self.transformations.append(gt)


class ObjX:

    def __init__(self, obj = ""):
        self.obj = obj
        self.graspingPoints = []
        self.__inizializeNominalSkeleton__()
        self.__generateGraspingPoints__()


    def __inizializeNominalSkeleton__(self):
        if "castiron" in self.obj: 
            self.nominal_2D = np.array([[350, 245],
                                        [250, 290],
                                        [350, 130],
                                        [445, 305],
                                        [415, 390]])
            self.nominal_3D = np.array([[0.014153, 0.003794, 0.225641],
                                        [-0.036255, 0.026773, 0.284667],
                                        [0.021589, -0.044881, 0.289716],
                                        [0.057400, 0.038809, 0.285552],
                                        [ 0.03727766, 0.06337749, 0.25]])
            self.numKeypoints = 5  
            if "back" in self.obj:
                self.gpIndexes = [1,2,3]
            else:
                self.gpIndexes = [0,1,2,3] #[0,1,2,3]
        elif "air" in self.obj:
            self.nominal_2D = np.array([[535, 105],
                                        [525,  65],
                                        [505, 225],
                                        [505, 290],
                                        [270, 190],
                                        [130, 385]])
            self.nominal_3D = np.array([[ 0.10121389, -0.06424604,  0.294     ],
                                        [ 0.09215661, -0.0797553 ,  0.281     ], 
                                        [ 0.07708595, -0.0059027 ,  0.261     ],
                                        [ 0.08380000,  0.023275  ,  0.297389  ],
                                        [-0.030869  , -0.028323  ,  0.298163  ],
                                        [-0.07333688,  0.05535409,  0.232     ]])
            self.numKeypoints = 6
            self.gpIndexes = [3,4]
        elif "plastic" in self.obj:
            self.nominal_2D = np.array([[490, 290],
                                        [410, 150],
                                        [325, 285],
                                        [400, 355],
                                        [435, 370]])
            self.nominal_3D = np.array([[0.084423, 0.038249, 0.302517],  
                                        [0.052219, -0.039410, 0.308961], 
                                        [-0.000356, 0.025207, 0.296671], 
                                        [0.024710, 0.039892, 0.230699], 
                                        [ 0.04947035,  0.05845113,  0.273]])
            self.numKeypoints = 5
            if "back" in self.obj:
                self.gpIndexes = [0,1,2]
            else:
                self.gpIndexes = [0,1,2,3] 
            


    def getGpIndexes(self):
        return self.gpIndexes
         
         
    def __getGraspingPointOrientation__(self, idx):
        if "castiron" in self.obj:
            if idx == 0:
                R = np.array([[0.011585, 0.999845, 0.013331],
                            [-0.999831, 0.011774, -0.014103],
                            [-0.014258, -0.013166, 0.999812]])
            elif idx == 1:
                R = np.array([[-0.441743, 0.012298, 0.897066],
                            [-0.897099, -0.015671, -0.441554],
                            [0.008627, -0.999792, 0.017955]])
            elif idx == 2:
                R = np.array([[0.999708, 0.007517, -0.022989],
                            [0.023162, -0.023920, 0.999454],
                            [0.006963, -0.999676, -0.024087]])
            elif idx == 3:
                R = np.array([[-0.444775, 0.014825, -0.895528],
                            [0.895427, -0.014543, -0.444974],
                            [-0.019620, -0.999776, -0.006806]])
        elif "air" in self.obj:
            if idx == 3:
                R = np.array([[0.071686, 0.956707, -0.282089],
                            [-0.290391, -0.250543, -0.923535],
                            [-0.954211, 0.148118, 0.259858]])
            elif idx == 4:
                R = np.array([[0.422622, 0.717489, 0.553721],
                            [-0.441649, -0.370468, 0.817136],
                            [0.791407, -0.589878, 0.160310]])
        elif "plastic" in self.obj:
            if idx == 0:
                R = np.array([[-0.451717, 0.024750, -0.891826],
                            [0.889982, -0.057322, -0.452382],
                            [-0.062317, -0.998039, 0.003866]])
            elif idx == 1:
                R = np.array([[0.996440, 0.075954, -0.036587],
                            [0.040780, -0.054414, 0.997695],
                            [0.073787, -0.995616, -0.057318]])
            elif idx == 2:
                R = np.array([[-0.325595, 0.017964, 0.945348],
                            [-0.945467, 0.003275, -0.325705],
                            [-0.008947, -0.999823, 0.015918]])
            elif idx == 3:
                R = np.array([[0.288881, 0.957267, -0.013708],
                            [-0.956426, 0.289202, 0.040138],
                            [0.042386, 0.001516, 0.999100]])

        return R

    def __rotElemX__(self, theta):
        return np.matrix([[1, 0, 0],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
    
    
    def __rotElemY__(self, theta):
        return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [0, 1, 0],
                   [-m.sin(theta), 0, m.cos(theta)]])
    

    def __rotElemZ__(self, theta):
        return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [0, 0, 1]])

    def __generateGraspingPoints__(self):
        for idx in self.gpIndexes:
            T_gp = np.array([[1., 0., 0., self.nominal_3D[idx,0]],
                          [0., 1., 0., self.nominal_3D[idx,1]],
                          [0., 0., 1., self.nominal_3D[idx,2]],
                          [0., 0., 0., 1.]])
            T_gp[0:3,0:3] = self.__getGraspingPointOrientation__(idx)
            print("--")
            print(idx)
            print(T_gp)
            print("--")
            gp = GraspingPoint(T_gp, self.nominal_3D, self.numKeypoints)  # graspingPoint = gp, nominal3D = n3D, numKeypoints = nK
            self.graspingPoints.append(gp)

    
    def selectGraspingPoint(self, gp_curr_idx):
        gp_curr = self.nominal_3D[gp_curr_idx,:]
        for gpi in self.graspingPoints:
            Tpoint = gpi.getGraspingPoint()
            print(Tpoint)
            if gp_curr[0] == Tpoint[0,3] and gp_curr[1] == Tpoint[1,3] and gp_curr[2] == Tpoint[2,3]:
                return gpi
        return -1