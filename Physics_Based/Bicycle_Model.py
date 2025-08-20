import numpy as np

class Bicycle_Model():

    def bicycle_model(heading, y, xvel,yvel,T_s,L):
        vel=np.sqrt(xvel**2+yvel**2)
        orientation = np.arctan2(np.sin(heading), np.cos(heading))
        xpos_new=y[:,0]+(vel*np.cos(orientation))*T_s
        ypos_new=y[:,1]+(vel*np.sin(orientation))*T_s

        heading_new=orientation+((vel/L)*np.tan(heading))*T_s
        heading_new=np.degrees(heading_new)
        return xpos_new, ypos_new, heading_new