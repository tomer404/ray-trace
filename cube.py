import numpy as np
from ray import Ray
class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index
    def intersect(self,ray):
        pass