import numpy as numpy

class GeomUtility:

    @staticmethod
    def planeWith3Points(P):
        points = np.array(P)
        p1 = points[0, 0:3] #np.array([1, 2, 3])
        p2 = points[1, 0:3] #np.array([4, 6, 9])
        p3 = points[2, 0:3] #np.array([12, 11, 9])
        # These two vectors are in the plane
        v1 = p3 - p1
        v2 = p2 - p1
        # the cross product is a vector normal to the plane
        cp = np.cross(v1, v2)
        a, b, c = cp
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        d = np.dot(cp, p3)
        print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))


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