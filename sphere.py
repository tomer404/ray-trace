import numpy as np
from ray import Ray

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

class Sphere:
    def __init__(self, position, radius, material_index):
        self.position = position
        self.radius = radius
        self.material_index = material_index

    def intersect(self,ray):
        oc = ray.origin - self.position
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            t = min(t1, t2)
            if t > 0:
                hit_point = ray.at(t)
                normal = normalize(hit_point - self.position)
                return [True, t, hit_point, normal]
        return [False, None, None, None]
    