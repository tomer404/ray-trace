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
from tqdm import tqdm

def reflect(v, n):
    n = np.array(n)
    v = np.array(v)
    return 2 * np.dot(v, n) * n - v

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
    
    def at(self, t):
        return self.origin + t * self.direction


class Intersection:
    def __init__(self, t, obj, hit_point, normal):
        self.t = t
        self.obj = obj
        self.hit_point = hit_point
        self.normal = normal

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, shininess, transparency):
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.reflection_color = reflection_color
        self.shininess = shininess
        self.transparency = transparency

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, radius):
        self.position = position
        self.color = color
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

class SceneSettings:
    def __init__(self, background_color, root_number_shadow_rays, max_recursions):
        self.background_color = background_color
        self.root_number_shadow_rays = root_number_shadow_rays
        self.max_recursions = max_recursions

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        self.position = position
        self.look_at = look_at
        self.up_vector = up_vector
        self.screen_distance = screen_distance
        self.screen_width = screen_width

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
    
class Cube:
    def __init__(self, position, scale, material_index):
        self.position = position
        self.scale = scale
        self.material_index = material_index

    def intersect(self,ray):
        #TODO implement method
        pass


class InfinitePlane:
    def __init__(self, normal, offset, material_index):
        self.normal = normal
        self.offset = offset
        self.material_index = material_index

    def intersect(self,ray):
        denom = np.dot(ray.direction, self.normal)
        if(abs(denom)>1e-6):
            t = -(np.dot(ray.origin, self.normal)-self.offset)/denom
            if(t>0):
                hit_point = ray.at(t)
                if(np.dot(self.normal, ray.direction)<0):
                    normal = np.array(self.normal)
                else:
                    normal = - np.array(self.normal)
                return [True, t, hit_point, normal]    
        return [False, None, None, None]
        

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
    '''
    This function recieves as input a light ray from the hit point to the light source
    and returns True if the light source gets blocked by an object
    '''
    
    epsilon = 1e-3
    for obj in surfaces:
        result = obj.intersect(light_ray)
        if result[0] and epsilon < result[1] < 1:
            return True
    return False

def get_color(hit, ray, scene_settings, lights, surfaces, material_list):
    if(hit.obj==None):
        return scene_settings.background_color
    color_sum = np.zeros(3)
    if(hit.obj.material_index>len(material_list)):
        raise ValueError(f"Material with index {hit.obj.material_index} not found!")
    #get necessary material parameters
    hit_object_material = material_list[hit.obj.material_index-1]
    hit_object_diffuse = hit_object_material.diffuse_color
    hit_object_specular = hit_object_material.specular_color
    alpha = hit_object_material.shininess
    transparency = hit_object_material.transparency
    #compute lighting for each light
    for light in lights:
        color_sum += compute_lighting(hit, ray, light, surfaces, hit_object_diffuse, hit_object_specular, alpha, scene_settings.root_number_shadow_rays)
    color_sum = np.array(scene_settings.background_color) * transparency+ color_sum*(1-transparency)
    return np.clip(color_sum, 0, 1)


def ray_trace(hit, ray, scene_settings, lights, surfaces, material_list, level):
    if(level == 0):
        return scene_settings.background_color
    if(hit.obj==None):
        return scene_settings.background_color
    color = get_color(hit, ray, scene_settings, lights, surfaces, material_list)
    reflection_dir = normalize(reflect(-ray.direction, hit.normal))
    reflection_hit = find_nearest_intersection(Ray(hit.hit_point, reflection_dir), surfaces)
    reflection_color = ray_trace(reflection_hit, Ray(hit.hit_point, reflection_dir), scene_settings, lights, surfaces, material_list, level-1)      
    color += np.array(reflection_color)*material_list[hit.obj.material_index-1].reflection_color
    return np.clip(color, 0, 1)


def calc_light_intensity(hit, ray, light, surfaces, root_number_shadow_rays):
    root_number_shadow_rays = int(root_number_shadow_rays)
    num_shadow_rays = root_number_shadow_rays ** 2

    Ld=light.position-hit.hit_point
    plane_normal = normalize(Ld)

    #TODO check what happens if parallel
    #These are the vectors of the plane
    u = normalize(np.cross(np.array([0, 0, 1]), plane_normal))*light.radius/root_number_shadow_rays
    v = normalize(np.cross(u, plane_normal))*light.radius/root_number_shadow_rays

    #calculating the starting corner of the plane
    rectangle_corner = light.position - (root_number_shadow_rays/2)*u - (root_number_shadow_rays/2)*v
    rng = np.random.default_rng()
    cnt_intersection = 0
    for x in range(root_number_shadow_rays):
        for y in range(root_number_shadow_rays):
            sample_position = rectangle_corner + u*(x+rng.random()) + v*(y+rng.random())
            light_ray = Ray(hit.hit_point, sample_position-hit.hit_point)
            if(light_intersect(light_ray, surfaces)):
                cnt_intersection += 1
    return 1-cnt_intersection/num_shadow_rays

def compute_lighting(hit, ray, light, surfaces, hit_object_diffuse, hit_object_specular, alpha, root_number_shadow_rays):

    light_intensity = calc_light_intensity(hit, ray, light, surfaces, root_number_shadow_rays)
    
    Ld=light.position-hit.hit_point
    Ld = normalize(Ld)
    reflected = normalize(reflect(Ld, hit.normal))
    diffuse_color = np.dot(Ld, hit.normal)*np.array(light.color)*np.array(hit_object_diffuse)

    # These are 2 auxilliary values necessary for the specular color calculation
    reflected_dot = np.dot(reflected, -ray.direction)
    phong_factor = np.sign(reflected_dot)*np.power(np.abs(reflected_dot), alpha)

    specular_color = light.specular_intensity*phong_factor*np.array(light.color)*np.array(hit_object_specular)
    return (diffuse_color+specular_color)*light_intensity





def compute_shadow(hit_point, light, root_number_shadow_rays, scene_objects):
    num_shadow_rays = int(root_number_shadow_rays ** 2)  # Ensure it's an integer
    light_hits = 0

    for _ in range(num_shadow_rays):
        # Generate random shadow ray within light radius
        light_dir = light.position - hit_point
        light_distance = np.linalg.norm(light_dir)
        light_dir = light_dir / light_distance

        # Create a shadow ray
        shadow_ray = Ray(hit_point + light_dir * 1e-3, light_dir)

        # Check for intersections
        shadow_intersects = False
        for obj in scene_objects:
            if isinstance(obj, Sphere):
                result = obj.intersect(shadow_ray, obj)
            elif isinstance(obj, InfinitePlane):
                result = obj.intersect(shadow_ray, obj)
            else:
                continue

            if result and 0 < result[0] < light_distance:
                shadow_intersects = True
                break

        if not shadow_intersects:
            light_hits += 1

    return light_hits / num_shadow_rays

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
    image = np.zeros((h, w, 3), dtype=np.float32)
    direction= normalize(np.array(camera.look_at)-np.array(camera.position))
    perpendicular_vec = np.cross(camera.up_vector, direction)
    camera.up_vector = np.cross(perpendicular_vec, direction)
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
    with tqdm(total=args.height * args.width, desc="Rendering", unit="pixel") as pbar:
        for x in range(w):
            for y in range(h):
                screen_pos = P0+ (x/w)*screen_width*Vx + (y/h)*screen_height*Vy
                ray = Ray(screen_pos, normalize(screen_pos-camera.position))
                hit = find_nearest_intersection(ray, surfaces)
                image[y,x]=ray_trace(hit, ray, scene_settings, lights, surfaces, materials, scene_settings.max_recursions)
                pbar.update(1)
    image = image*255
    Image.fromarray((image).astype(np.uint8)).save(args.output_image)


    # Save the output image
    save_image(image)


if __name__ == '__main__':
    main()
