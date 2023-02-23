import numpy as np

class GeomUtility:

    #@staticmethod
    #def planeWith3Points(P):
    #    points = np.array(P)
    #    p1 = points[0, 0:3] #np.array([1, 2, 3])
    #    p2 = points[1, 0:3] #np.array([4, 6, 9])
    #    p3 = points[2, 0:3] #np.array([12, 11, 9])
    #    # These two vectors are in the plane
    #    v1 = p3 - p1
    #    v2 = p2 - p1
    #    # the cross product is a vector normal to the plane
    #    cp = np.cross(v1, v2)
    #    a, b, c = cp
    #    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    #    d = np.dot(cp, p3)
    #    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))


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
        print("punto medio")
        print(p_med)
        v = p_med - pp #pp - p_med
        print(v)
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

        if 'back' in detectedBB.classe.lower():
            T[:3,3] = high_p
            print(T[:3,3])
            y = normal_vec
            z = lying_vec
            x = np.cross(y,z)
            print(x)
            print(y)
            print(z)
            T[:3,0] = x
            T[:3,1] = y
            T[:3,2] = z
        else:
            T[:3,3] = keypoints3D[0,:] # -1 is index for last element, -2 penultimo
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

            asseA1 = GeomUtility.r2asseangolo(R1)
            asseA2 = GeomUtility.r2asseangolo(R2)

            if asseA1[-1] < asseA2[-1]:
                T[:3,:3] = R1
            else:
                T[:3,:3] = R2
            #T[:3,0] = x
            #T[:3,1] = y
            #T[:3,2] = z
        print(T)
        return T
        

            



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
            if vf > depth_array.shape[1]:
                vf = depth_array.shape[1]
            if uf > depth_array.shape[0]:
                uf = depth_array.shape[0]


            if d1 != 0.:
                d = d1
            else:
                intornoMIN = np.array(depth_array[vi:vf, ui:uf])
                print('DIMENSIONE INTORNO')
                print(intornoMIN.shape)
                print("vi: " + str(vi) + " vf: " + str(vf) + " ui: " + str(ui) + " uf: " + str(uf))
                nonZeroValuesMIN = intornoMIN[np.nonzero(intornoMIN)]
                #print("non zero min")
                #print(nonZeroValuesMIN)
                d = np.min(nonZeroValuesMIN)
                print('d3')
                print(d)


            r = 0
            d2 = 0.

            #if(v-60 > 0 and u-60 > 0):
            #    if(v+60 < 480 and u+60 < 640):
            #        intornoMIN = depth_array[v-60:v+60, u-60:u+60]
            #        nonZeroValuesMIN = intornoMIN[np.nonzero(intornoMIN)]
            #        d3 = np.min(nonZeroValuesMIN)
            #        print('d3')
            #        print(d3)
            

            #nonZeroValues = depth_array[v,u] 
            #while d2 == 0.:
            #    if not np.all((nonZeroValues == 0)):
            #        d2 = np.mean(nonZeroValues)
            #        print('d2')
            #        print(d2)
            #    else:
            #        r = r + 1
            #        intorno = depth_array[v-r:v+r, u-r:u+r]
            #        nonZeroValues = intorno[np.nonzero(intorno)]
            #        print(intorno)
            
            x_ = (u - K[0,2])/K[0,0]
            y_ = (v - K[1,2])/K[1,1]
            z = (float(np.array(d))) / 1000
            x = x_ * z
            y = y_ * z
            return np.array([x, y, z])
        else: 
            return np.array([-1, -1, -1])

    @staticmethod
    def get_matrix(normal, y_w):
        R = np.array([[-0.094378, 0.995485, -0.009164],
                    [0.995355, 0.094186, -0.019453],
                    [-0.018502, -0.010957, -0.999769]])
        v_ = R.dot(normal)
        x = np.cross(normal,y_w)
        #x = np.cross(np.array([1, 0, 0]), normal)
        x = x/np.linalg.norm(x)
        y = np.cross(normal, x)
        y = y/np.linalg.norm(y)
        return np.array([x, y, normal]).T 