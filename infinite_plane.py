import numpy as np
from ray import Ray
class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index
    def intersect(self,ray):
        t = -(np.dot(ray.origin, self.normal)-self.offset)/np.dot(ray.direction, self.normal)
        if(t>0):
            hit_point = ray.at(t)
            if(np.dot(self.normal, ray.direction)<0):
                normal = self.normal
            else:
                normal = - self.normal
            return [True, t, hit_point, normal]
        else:
            return [False, None,None, None]