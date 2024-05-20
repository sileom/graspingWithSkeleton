import numpy as np
import math as m
from graspingPoints import *

class GeomUtility:    
    @staticmethod
    def Rx(theta):
        return np.array([[1, 0, 0],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
    
    
    @staticmethod
    def Ry(theta):
        return np.array([[ m.cos(theta), 0, m.sin(theta)],
                   [0, 1, 0],
                   [-m.sin(theta), 0, m.cos(theta)]])
    
    
    @staticmethod
    def Rz(theta):
        return np.array([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [0, 0, 1]])


    @staticmethod
    def getNormalVec(a, b, c):
        v = np.cross(b-a, c-a) # uscente dall'oggetto
        v = np.cross(c-a, b-a) # entrante nell'oggetto
        n = v / np.linalg.norm(v)
        return n

    @staticmethod
    def getHigherPoint(a, b, c):
        idx = 0
        if (a[2] >= b[2]) and (a[2] >= c[2]):
            hp = a
            idx = 1
        elif (b[2] >= a[2]) and (b[2] >= c[2]):
            hp = b
            idx = 2
        else:
            hp = c
            idx = 3
        return [hp, idx]
         

    @staticmethod
    def getLyingVec(a, b, c):
        [p_higher, idx_p] = GeomUtility.getHigherPoint(a,b,c)
        p_med = np.array([0.,0.,0.])
        p1 = []
        p2 = []
        pp = []
        if idx_p == 1: #a
            pp = a
            p1 = b
            p2 = c
        elif idx_p == 2: #b
            pp = b
            p1 = a
            p2 = c
        elif idx_p == 3: #c
            pp = c
            p1 = a
            p2 = b
        p_med[0] = (p1[0]+p2[0])/2
        p_med[1] = (p1[1]+p2[1])/2
        p_med[2] = (p1[2]+p2[2])/2
        #print("punto medio")
        #print(p_med)
        v = p_med - pp #pp - p_med
        #print(v)
        g = v / np.linalg.norm(v)
        return g

    @staticmethod
    def computeGraspingPoint(detectedBB, keypoints3D, R_curr):
        print()

        normal_vec = GeomUtility.getNormalVec(keypoints3D[1,:], keypoints3D[2,:], keypoints3D[3,:])
        print("normale")
        print(normal_vec)
        [high_p, idx_p] = GeomUtility.getHigherPoint(keypoints3D[1,:], keypoints3D[2,:], keypoints3D[3,:])
        print("p alto")
        print(high_p)
        lying_vec = GeomUtility.getLyingVec(keypoints3D[1,:], keypoints3D[2,:], keypoints3D[3,:])
        print("giacente")
        print(lying_vec)

        T= np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
        
        R1 = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
        
        R2 = np.array([[1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
        
        x,y,z = [[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]

        if 'back' in detectedBB.classe.lower():
            T[:3,3] = high_p
            print(T[:3,3])
            y = normal_vec
            z = lying_vec
            x = np.cross(y,z)
        else:
            # first matrix 
            z = normal_vec
            y = lying_vec
            x = np.cross(y,z)
        R1[:,0] = x
        R1[:,1] = y
        R1[:,2] = z
        # second matrix 
        y = -np.array(y)
        x = np.cross(y,z)
        R2[:,0] = x
        R2[:,1] = y
        R2[:,2] = z
        err1=R_curr.T.dot(R1)
        err2=R_curr.T.dot(R2)
        asseA1 = GeomUtility.r2asseangolo(err1)
        asseA2 = GeomUtility.r2asseangolo(err2)
        print('DECISIONE ASSE')
        print('R1')
        print(R1)
        print('R2')
        print(R2)
        print(' ang1 ---- ang2')
        print(str(asseA1[-1]) + ' ---- ' + str(asseA2[-1]))
        if asseA1[-1] < asseA2[-1]:
            T[:3,:3] = R1
        else:
            T[:3,:3] = R2
        if 'back' in detectedBB.classe.lower():
            T[:3,3] = high_p
        else:
            print("vecchio")
            print(keypoints3D[0,:])
            print("nuovo")
            print(keypoints3D[0,:] +np.array(T[:3,:3]).dot([0,0,-0.0]))
            T[:3,3] = keypoints3D[0,:] + T[:3,:3].dot([0,0,-0.01])# -1 is index for last element, -2 penultimo
        print(T)
        return T


    @staticmethod
    def getHigherPointInCameraFrame(keypoints,keyIdxes):
        higherPoint = np.array([10,10,10])
        higherPointIdx = 0
        for i in range(keypoints.shape[0]):
            point = keypoints[i]
            norm_curr = np.linalg.norm(point)
            if norm_curr < np.linalg.norm(higherPoint) and i in keyIdxes:
                higherPoint = point
                higherPointIdx = i
        return [higherPoint, higherPointIdx]

    @staticmethod
    def isVisible(a): #a e' un punto 3D
        return a[0] != -1 and a[1] != -1 and a[2] != -1

    @staticmethod
    def getVisibleTripleIdxes(keypoints, obj_name):
        l = range(0,keypoints.shape[0], 1)
        comb = combinations(l,3)
        listaComb = list(comb)
        dimListaComb = len(listaComb)
        for i in (range(dimListaComb)):
            triple = listaComb[i]
            print(triple)
            print(triple[0])
            print("+++++")
            if 'castiron' in obj_name:
                if 'back' in obj_name:
                    esito = not(triple[0]==0) and GeomUtility.isVisible(keypoints[triple[0],:]) and GeomUtility.isVisible(keypoints[triple[1],:]) and GeomUtility.isVisible(keypoints[triple[2],:])
                else:
                    esito = GeomUtility.isVisible(keypoints[triple[0],:]) and GeomUtility.isVisible(keypoints[triple[1],:]) and GeomUtility.isVisible(keypoints[triple[2],:])
            elif 'plastic' in obj_name:
                if 'back' in obj_name:
                    esito = (triple[0]!=3 and triple[1]!=3 and triple[2]!=3) and GeomUtility.isVisible(keypoints[triple[0],:]) and GeomUtility.isVisible(keypoints[triple[1],:]) and GeomUtility.isVisible(keypoints[triple[2],:])
                else:
                    esito = GeomUtility.isVisible(keypoints[triple[0],:]) and GeomUtility.isVisible(keypoints[triple[1],:]) and GeomUtility.isVisible(keypoints[triple[2],:])
            elif 'air' in obj_name:
                esito = (triple[0]!=1 and triple[1]!=1 and triple[2]!=1) and (triple[0]!=2 and triple[1]!=2 and triple[2]!=2) and (triple[0]!=0 and triple[1]!=0 and triple[2]!=0) and GeomUtility.isVisible(keypoints[triple[0],:]) and GeomUtility.isVisible(keypoints[triple[1],:]) and GeomUtility.isVisible(keypoints[triple[2],:])
            if esito == True:
                return [triple[0],triple[1],triple[2]]
        return [-1,-1,-1]


    @staticmethod
    def computeGraspingPointWithTriple(detectedBB, keypoints3D, R_curr):
        print("[computeGraspingPointWithTriple]")
        if 'oil' in detectedBB.classe.lower() and 'castiron' in detectedBB.classe.lower():
            obj_name = "oil_separator_crankcase_castiron"
        elif 'oil' in detectedBB.classe.lower() and 'plastic' in detectedBB.classe.lower():
            obj_name = "oil_separator_crankcase_plastic"
        elif 'air' in detectedBB.classe.lower():
            obj_name = "air_pipe"
        
        objOil = ObjX(detectedBB.classe.lower())
        [higherKeyPoint, higherKeyPointIdx] = GeomUtility.getHigherPointInCameraFrame(keypoints3D, objOil.getGpIndexes()) # punto più in alto e relativo indice nella lista dei keypoint
        print("")
        print(higherKeyPoint)
        print(higherKeyPointIdx)
        nominalGraspingPoint = objOil.selectGraspingPoint(higherKeyPointIdx)
        print(nominalGraspingPoint.getGraspingPoint())
        if nominalGraspingPoint == -1:
            print("ERROR: nominalGraspingPoint not found. Abort")

        [ia,ib,ic] = GeomUtility.getVisibleTripleIdxes(keypoints3D, detectedBB.classe.lower()) # indici dei primi keypoint visibili, usare le combinazioni così manteniamo gli indici delle triplette
        print()
        print(ia)
        print(ib)
        print(ic)
        print()
        if ia == -1 and ib == -1 and ic == -1:
            print("ERROR: Tripla not found. Abort")
        plane_curr = T_plane(keypoints3D[ia,:],keypoints3D[ib,:],keypoints3D[ic,:]) # --> T_3^c
        tg_curr = nominalGraspingPoint.getGTranformWithSpecificTripleIdxes(ia,ib,ic)
        if tg_curr == -1:
            print("ERROR: Tg not found. Abort")

        T_grasp_c = plane_curr.getT3().dot(tg_curr.getTg()) # GRASPING POINT CAMERA FRAME


        if 'castiron' in obj_name:
            if higherKeyPointIdx == 0:
                T_grasp_c[0:3,3] = T_grasp_c[0:3,3] + T_grasp_c[0:3,0:3].dot([0,0,0.0015])
            else:
                T_grasp_c[0:3,3] = T_grasp_c[0:3,3] + T_grasp_c[0:3,0:3].dot([0,0,0.003])
        elif 'plastic' in obj_name:
            if higherKeyPointIdx == 3:
                T_grasp_c[0:3,3] = T_grasp_c[0:3,3] + T_grasp_c[0:3,0:3].dot([0,0,0.005])
        elif 'air' in obj_name:
            if higherKeyPointIdx == 3:
                T_grasp_c[0:3,3] = T_grasp_c[0:3,3] + T_grasp_c[0:3,0:3].dot([0,0,0.005])

        return T_grasp_c
        
    
    @staticmethod
    def isIn(bb, pixel):
        if bb.x <= pixel[0] and pixel[0] <= (bb.x + bb.w):
            if bb.y <= pixel[1] and pixel[1] <= (bb.y + bb.h):
                return True
        return False       



    @staticmethod
    def quat2r(quat):
        #quat = [w, x, y, z]
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]
        r00 = 1-2*(y**2)-2*(z**2)
        r01 = 2*x*y-2*w*z
        r02 = 2*x*z+2*w*y
        r10 = 2*x*y+2*w*z
        r11 = 1-2*(x**2)-2*(z**2)
        r12 = 2*y*z-2*w*x
        r20 = 2*x*z-2*w*y
        r21 = 2*y*z+2*w*x
        r22 = 1-2*(x**2)-2*(y**2)
        R = np.array([[r00, r01, r02],
                      [r10, r11, r12],
                      [r20, r21, r22]])
        return R


    @staticmethod
    def r2quat(R):
        import numpy as np
        #q(0) e` l'elemento "scalare" in uscita
        q = [0, 0, 0, 0]
        r1 = R[0,0]
        r2 = R[1,1]
        r3 = R[2,2]
        r4 = r1 + r2 + r3
        j = 1
        rj = r1
        if r2>rj:
            j = 2
            rj = r2
        if r3>rj:
            j = 3
            rj = r3
        if r4>rj:
            j = 4
            rj = r4
        pj = 2* np.sqrt(1+2*rj-r4)
        if j == 1:
            p1 = pj/4
            p2 = (R[1,0]+R[0,1])/pj
            p3 = (R[0,2]+R[2,0])/pj
            p4 = (R[2,1]-R[1,2])/pj
        elif j == 2:
            p1 = (R[1,0]+R[0,1])/pj
            p2 = pj/4;
            p3 = (R[2,1]+R[1,2])/pj
            p4 = (R[0,2]-R[2,0])/pj
        elif j == 3:
            p1 = (R[0,2]+R[2,0])/pj
            p2 = (R[2,1]+R[1,2])/pj
            p3 = pj/4;
            p4 = (R[1,0]-R[0,1])/pj
        else:
            p1 = (R[2,1]-R[1,2])/pj
            p2 = (R[0,2]-R[2,0])/pj
            p3 = (R[1,0]-R[0,1])/pj
            p4 = pj/4
        if p4 > 0:
            q[1] =  p1
            q[2] =  p2
            q[3] =  p3
            q[0] =  p4
        else:
            q[1] =  -p1
            q[2] =  -p2
            q[3] =  -p3
            q[0] =  -p4
        return q
    

    @staticmethod
    def r2asseangolo(R):
        # r_x, r_y, r_z, theta
        val = (R[0,0]+R[1,1]+R[2,2] - 1)*0.5
        theta = np.arccos( min(max(val,-1),1) )

        r = np.zeros(3)

        if abs(theta-np.pi) <= 0.00001:
            r[0] = -1* np.sqrt( (R[0,0]+1)*0.5 )
            r[1] = np.sqrt( (R[1,1]+1)*0.5 )
            r[2] = np.sqrt( 1- r[0]**2 - r[1]**2)
        else:
            if theta >= 0.00001:
                r[0] = (R[2,1]-R[1,2]) / (2*np.sin(theta))
                r[1] = (R[0,2]-R[2,0]) / (2*np.sin(theta))
                r[2] = (R[1,0]-R[0,1]) / (2*np.sin(theta))
        result = [r[0], r[1], r[2], theta]
        return result


    @staticmethod
    def asseangolo2r(asseAngolo):
        # r_x, r_y, r_z, theta
        rx = asseAngolo[0]
        ry = asseAngolo[1]
        rz = asseAngolo[2]
        angle = asseAngolo[3]
        a = np.cos(angle)
        b = np.sin(angle)
        R= np.zeros((3,3))
        R[0,0] = (rx**2) * (1-a) + a
        R[0,1] = (rx*ry)*(1-a)-rz*b
        R[0,2] = (rx*rz)*(1-a)+ry*b
        R[1,0] = (rx*ry)*(1-a)+rz*b
        R[1,1] = (ry**2) * (1-a) + a
        R[1,2] = (ry*rz)*(1-a)-rx*b
        R[2,0] = (rx*rz)*(1-a)-ry*b
        R[2,1] = (ry*rz)*(1-a)+rx*b
        R[2,2] = (rz**2) * (1-a) + a
        return R


    @staticmethod
    def deproject_pixel_to_point(ppixel, depth_array, K):
        v = ppixel[1] #x 640
        u = ppixel[0] #y 480
        d=0

        depth_array = np.array(depth_array)

        if v != -1:
            d1 = depth_array[v,u] 
            print('d1')
            print(d1)

            r = 10 #10
            vi = v-r 
            vf = v+r+1
            ui = u-r 
            uf = u+r+1
            
            if vi < 0: 
                vi = 0
            if ui < 0:
                ui = 0
            if vf > depth_array.shape[0]:
                vf = depth_array.shape[0]
            if uf > depth_array.shape[1]:
                uf = depth_array.shape[1]


            if d1 != 0.:
                d = d1
            else:
                intornoMIN = np.array(depth_array[vi:vf, ui:uf])
                nonZeroValuesMIN = intornoMIN[np.nonzero(intornoMIN)]
                d = np.min(nonZeroValuesMIN)
            r = 0
            d2 = 0.
            x_ = (u - K[0,2])/K[0,0]
            y_ = (v - K[1,2])/K[1,1]
            z = (float(np.array(d))) / 1000
            x = x_ * z
            y = y_ * z
            return np.array([x, y, z])
        else: 
            return np.array([-1, -1, -1])


    @staticmethod
    def computeNewLookPose(po, Ae_curr, angleForLook):
        pRob = Ae_curr[:3,3]

        center = np.array([po[0], po[1], pRob[2]])
        r = np.linalg.norm(center - pRob)

        angle_curr = np.arcsin((pRob[1]-center[1])/r)
        angle_curr2 = np.arcsin((center[1]-pRob[1])/r)
        angle_curr3 = np.arcsin((center[0]-pRob[0])/r)

        px_new = r*np.cos(angle_curr+angleForLook) + center[0]
        py_new = r*np.sin(angle_curr+angleForLook) + center[1]
        pz_new = center[2]
        p_new = np.array([px_new, py_new, pz_new])
        x = (center-p_new)/r
        z = (po - center) / np.linalg.norm(po-center)
        y = np.cross(z,x)

        T= np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
        T[:3,0] = x
        T[:3,1] = y
        T[:3,2] = z
        T[0,3] = px_new
        T[1,3] = py_new
        T[2,3] = pz_new
        T_new = np.array([[-0.795977, 0.603738, 0.0436101, -0.112803],
                            [0.597776, 0.795358, -0.100255, 0.467724],
                            [-0.0952133, -0.0537317, -0.994006, 0.430012],
                            [0, 0, 0, 1]])
        return T_new
        #return T

    
    @staticmethod
    def getCenterBB(detection):
        xr = detection.x + detection.w
        yb = detection.y + detection.h
        a = (detection.x + xr)/2
        b = (detection.y + yb)/2
        center = [int(a), int(b)]
        return center
        

        

