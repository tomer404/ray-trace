import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere
from ray import Ray


class Intersection:
    def __init__(self, t, obj, hit_point, normal):
        self.t = t
        self.obj = obj
        self.hit_point = hit_point
        self.normal = normal

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def calc_P0(camera, w, h):
    direction= normalize(np.array(camera.look_at)-np.array(camera.position))
    mid_screen = camera.position+direction*camera.screen_distance
    Vy = normalize(np.array(camera.up_vector))
    Vx = normalize(np.cross(direction, Vy))
    screen_width = camera.screen_width
    screen_height = camera.screen_width*h/w
    P0= mid_screen-screen_width/2*Vx-screen_height/2*Vy
    return P0




def find_nearest_intersection(ray, surfaces):
    closest_t = float('inf')
    closest_obj = None
    normal = None
    hit_point = None
    for obj in surfaces:
        result = obj.intersect(ray)
        if result[0] and result[1] < closest_t:
            closest_t = result[1]
            hit_point = result[2]
            normal = result[3]
            closest_obj = obj
                
    return Intersection(closest_t, closest_obj, hit_point, normal)


def light_intersect(light_ray, surfaces):
    epsilon = 1e8
    for obj in surfaces:
        result = obj.intersect(light_ray)
        if result[0] and result[1]<1:
            return True
    return False

def get_color(hit, ray, scene_settings, lights, surfaces, material_list):
    if(hit.obj==None):
        return scene_settings.background_color
    color_sum = np.zeros(3)
    if(hit.obj.material_index>len(material_list)):
        raise ValueError(f"Material with index {hit.obj.material_index} not found!")
    hit_object_material = material_list[hit.obj.material_index-1]
    hit_object_diffuse = hit_object_material.diffuse_color
    hit_object_specular = hit_object_material.specular_color
    alpha = hit_object_material.shininess
    for object in lights:
        Ld=object.position-hit.hit_point
        if(light_intersect(Ray(hit.hit_point, Ld), surfaces)):
            continue
        Ld = normalize(Ld)
        reflected = normalize(2*np.dot(hit.normal, Ld)*np.array(hit.normal)-Ld)
        diffuse_color = np.dot(Ld, hit.normal)*np.array(object.color)*np.array(hit_object_diffuse)
        specular_color = object.specular_intensity*np.power(np.dot(reflected, -ray.direction), alpha)*np.array(object.color)*np.array(hit_object_specular)
        color_sum +=diffuse_color+specular_color
    return np.clip(color_sum, 0, 1)


def soft_shadows():
    pass

def construct_reflection_ray():
    pass

def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()


    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    w = args.width
    h = args.height
    image = np.zeros((w, h, 3), dtype=np.float32)
    direction= normalize(np.array(camera.look_at)-np.array(camera.position))
    perpendicular_vec = np.cross(camera.up_vector, direction)
    camera.up_vector = np.cross(direction, perpendicular_vec)
    Vy = normalize(np.array(camera.up_vector))
    Vx = normalize(np.cross(direction, Vy))
    screen_width = camera.screen_width
    screen_height = camera.screen_width*h/w
    P0 = calc_P0(camera, w, h)
    materials = []
    lights = []
    surfaces = []
    for object in objects:
        if(isinstance(object, Material)):
            #TODO check if this is correct usage of materials list
            materials.append(object)
        elif(isinstance(object, Light)):
            lights.append(object)
        elif isinstance(object, Sphere) or isinstance(object, InfinitePlane) or isinstance(object, Cube):
            surfaces.append(object)
    for x in range(w):
        for y in range(h):
            screen_pos = P0+ (x/w)*screen_width*Vx + (y/h)*screen_height*Vy
            ray = Ray(screen_pos, normalize(screen_pos-camera.position))
            hit = find_nearest_intersection(ray, surfaces)
            image[x,y]=get_color(hit, ray, scene_settings, lights, surfaces, materials)
    image = image*255

    # Save the output image
    save_image(image)


if __name__ == '__main__':
    main()
