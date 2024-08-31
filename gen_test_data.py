import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random
import numpy as np
from PIL import Image

def create_sphere(radius, slices, stacks):
    sphere = gluNewQuadric()
    gluQuadricDrawStyle(sphere, GLU_FILL)
    gluQuadricNormals(sphere, GLU_SMOOTH)  # Enable smooth normals
    gluSphere(sphere, radius, slices, stacks)
    gluDeleteQuadric(sphere)

def random_color():
    # Generate vibrant colors
    r = random.random()
    g = random.random()
    b = random.random()
    max_component = max(r, g, b)
    r /= max_component
    g /= max_component
    b /= max_component
    return (r, g, b, 1.0)

def create_scene(num_spheres=15):
    spheres = []
    for _ in range(num_spheres):
        x = random.uniform(-5, 5)
        y = random.uniform(-1, 5)
        z = random.uniform(-5, -50)
        radius = random.uniform(0.01, 1.0)
        color = random_color()
        spheres.append((x, y, z, radius, color))

    # spheres.append((0, 0, -5, radius, random_color()))
    return spheres

def render_scene(spheres):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0)

    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (5.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))

    # Render ground plane
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)  # Normal pointing upwards
    glColor3f(0.5, 0.5, 0.5)
    glVertex3f(-10, -1, -50)
    glVertex3f(10, -1, -50)
    glVertex3f(10, -1, 0)
    glVertex3f(-10, -1, 0)
    glEnd()

    # Render spheres
    for x, y, z, radius, color in spheres:
        glPushMatrix()
        glTranslatef(x, y, z)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color)
        glMaterialfv(GL_FRONT, GL_SPECULAR, (1.0, 1.0, 1.0, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
        create_sphere(radius, 32, 32)
        glPopMatrix()

def generate_scene(width=800, height=600, num_spheres=15):
    pygame.init()
    display = (width, height)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

    spheres = create_scene(num_spheres)
    render_scene(spheres)

    # Capture color image
    color_buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    color_image = Image.frombytes("RGB", (width, height), color_buffer)
    color_image = color_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Capture depth map
    depth_buffer = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    depth_array = np.frombuffer(depth_buffer, dtype=np.float32).reshape(height, width)
    
    # Normalize depth values
    zNear, zFar = 0.1, 50.0
    depth_array = (2.0 * zNear) / (zFar + zNear - depth_array * (zFar - zNear))

    print(depth_array.shape)
    print(depth_array.dtype)
    print(np.min(depth_array))
    print(np.max(depth_array))
    
    # Scale to 0-255 and convert to uint8
    depth_array = (depth_array * 255).astype(np.uint8)
    
    # Flip the depth map vertically and create an image
    depth_map = Image.fromarray(np.flipud(depth_array))

    pygame.quit()

    return color_image, depth_map

def save_images(color_image, depth_map, color_path='images/color_scene.png', depth_path='images/depth_map.png'):
    color_image.save(color_path)
    depth_map.save(depth_path)
    print(f"Color image saved as {color_path}")
    print(f"Depth map saved as {depth_path}")

def load_images(color_path='images/color_scene.png', depth_path='images/depth_map.png'):
    color_image = np.array(Image.open(color_path))
    depth_map = np.array(Image.open(depth_path).convert('L'))  # Convert to grayscale
    return color_image, depth_map

if __name__ == "__main__":
    color_image, depth_map = generate_scene(width=500, height=500, num_spheres=200)
    save_images(color_image, depth_map)