# represent cube as center + 2 rotation axes, write function which converts this to coordinates of the 8 vertices
# use coordinates x, y, z, where z is the distance from the observer.
# normalize these vectors (i.e. project them onto unit sphere), and then project the vector from the center to each
# vertex onto the plane, then display the points (and eventually lines between them)

import numpy as np
import math
import matplotlib.pyplot as plt
from pynput import keyboard
import time
import copy
import stl
import sys

class Triangle:
    def __init__(self, v1 : np.array, v2 : np.array, v3 : np.array):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.v1_2d = None
        self.v2_2d = None
        self.v3_2d = None

        self.color1 = np.array([255, 0, 0])
        self.color2 = np.array([0, 255, 0])
        self.color3 = np.array([255, 0, 255])

    def _edge_func(self, x, v1, v2):
        return (x[0] - v1[0]) * (v2[1] - v1[1]) - (x[1] - v1[1]) * (v2[0] - v1[0])

    def edge_func(self, x : np.array, xdim, ydim): # x [0 - 400, 0 - 400], xdim = 400, ydim = 400
        x = 2 * x / np.array([xdim, ydim]) - 1

        a = self._edge_func(self.v2_2d, self.v3_2d, x)
        b = self._edge_func(self.v3_2d, self.v1_2d, x)
        c = self._edge_func(self.v1_2d, self.v2_2d, x)

        return a, b, c

    def normal(self):
        return np.cross(self.v2 - self.v1, self.v3 - self.v1)

    # def distance(self, x : np.array):

class Camera:
    def __init__(self, pos : np.array, looks_at : np.array, up : np.array, FOV : float):
        self.pos = pos
        self.looks_at = looks_at
        self.up = up
        self.FOV = FOV
        self.near = 1 / np.tan(self.FOV / 2)

        v = looks_at - pos
        v = v / np.linalg.norm(v)

        w = np.cross(v, up)
        w = w / np.linalg.norm(w)

        s = np.cross(w, v)

        self.transform = np.array([np.concatenate((w, - np.array([np.dot(w, pos)]))),
                                   np.concatenate((s, -np.array([np.dot(s, pos)]))),
                                   np.concatenate((v, -np.array([np.dot(v, pos)])))])

    def clip_triangle(self, triangle : Triangle):
        triangle.v1 = self.get_position(triangle.v1)
        triangle.v2 = self.get_position(triangle.v2)
        triangle.v3 = self.get_position(triangle.v3)


    def get_position(self, x : np.array):
         return np.matmul(self.transform, np.concatenate((x, np.array([1]))))

    def project(self, x : np.array):
        return (self.near / x[2]) * x[0:2]

    def project_triangle(self, triangle : Triangle):
        triangle.v1_2d = self.project(triangle.v1)
        triangle.v2_2d = self.project(triangle.v2)
        triangle.v3_2d = self.project(triangle.v3)
        return triangle

    def get_near_intersection(self, v1, v2):
        t = (v1[2] - self.near) / (v2[2] - v1[2])
        return np.array([v1[0] + (v2[0] - v1[0])* t, v1[1] + (v2[1] - v1[1])* t, self.near])

class Renderer:
    def __init__(self, pos : np.array, looks_at : np.array, up : np.array, FOV : float, x_dim = 800, y_dim = 800):
        self.triangles = []
        self.to_render = []
        self.camera = Camera(pos, looks_at, up, FOV)

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.canvas = 255*np.ones(shape=(self.x_dim, self.y_dim, 3)).astype(np.uint8)
        self.distances = 1000000 * np.ones(shape=(self.x_dim, self.y_dim))

        self.count = 0

    def addTriangle(self, v1 : np.array, v2 : np.array, v3 : np.array):
        self.triangles.append(Triangle(v1, v2, v3))

    def clipTriangles(self):
        for triangle in self.triangles:
            self.camera.clip_triangle(triangle)

            if triangle.normal()[2] < 0:
                continue

            # a = (triangle.v1[2] >= self.camera.near)
            # b = (triangle.v2[2] >= self.camera.near)
            # c = (triangle.v3[2] >= self.camera.near)

            # if a and b and c:

            self.camera.project_triangle(triangle)
            self.to_render.append(triangle)
            self.count += 1

            # elif not a and not b and not c:
            #     continue
            #
            # elif not a and b and c:
            #     t1 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v2), triangle.v2, triangle.v3)
            #     t2 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v2), triangle.v3, self.camera.get_near_intersection(triangle.v1, triangle.v3))
            #     self.to_render.append(self.camera.project_triangle(t1))
            #     self.to_render.append(self.camera.project_triangle(t2))
            # elif not a and not b and c:
            #     t1 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v3),
            #                   self.camera.get_near_intersection(triangle.v2, triangle.v3),
            #                   triangle.v3)
            #     self.to_render.append(self.camera.project_triangle(t1))
            # elif not a and b and not c:
            #     t1 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v2),
            #                   triangle.v2,
            #                   self.camera.get_near_intersection(triangle.v3, triangle.v2))
            #     self.to_render.append(self.camera.project_triangle(t1))
            # elif a and not b and c:
            #     t1 = Triangle(triangle.v1, self.camera.get_near_intersection(triangle.v2, triangle.v1), triangle.v3)
            #     t2 = Triangle(self.camera.get_near_intersection(triangle.v2, triangle.v1), self.camera.get_near_intersection(triangle.v2, triangle.v3), triangle.v3)
            #     self.to_render.append(self.camera.project_triangle(t1))
            #     self.to_render.append(self.camera.project_triangle(t2))
            # elif a and b and not c:
            #     t1 = Triangle(triangle.v1, triangle.v2, self.camera.get_near_intersection(triangle.v1, triangle.v2))
            #     t2 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v3), triangle.v1, self.camera.get_near_intersection(triangle.v2, triangle.v3))
            #     self.to_render.append(self.camera.project_triangle(t1))
            #     self.to_render.append(self.camera.project_triangle(t2))
            # elif a and not b and not c:
            #     t1 = Triangle(self.camera.get_near_intersection(triangle.v1, triangle.v2), self.camera.get_near_intersection(triangle.v1, triangle.v3), triangle.v1)
            #     self.to_render.append(self.camera.project_triangle(t1))

        print("{} polygons clipped.".format(len(self.to_render) - self.count))

    def draw_triangles(self):
        print("rendering ", len(self.to_render), " triangles")
        for i, triangle in enumerate(self.to_render):
            bounding_box = np.array([self.x_dim / 2 * max(min(triangle.v1_2d[0], triangle.v2_2d[0], triangle.v3_2d[0]), -1) + self.x_dim / 2,
                                     self.x_dim / 2 * min(1, max(triangle.v1_2d[0], triangle.v2_2d[0], triangle.v3_2d[0])) + self.x_dim / 2,
                                     self.y_dim / 2 * max(-1, min(triangle.v1_2d[1], triangle.v2_2d[1], triangle.v3_2d[1])) + self.y_dim / 2,
                                     self.y_dim / 2 * min(1, max(triangle.v1_2d[1], triangle.v2_2d[1], triangle.v3_2d[1])) + self.y_dim / 2]).astype(np.int64)

            unit = triangle.normal() / np.linalg.norm(triangle.normal())
            color = 255 * (unit[1] + 1) / 2

            if (i % 1000 == 0):
                print ("{} of {} polygons rendered...".format(i, len(self.to_render)))

                # print(i, bounding_box, triangle.v1_2d, triangle.v2_2d, triangle.v3_2d, color)

            # area = triangle._edge_func(triangle.v1, triangle.v2, triangle.v3)
            for x in range(bounding_box[0], bounding_box[1] + 1):
                for y in range(bounding_box[2], bounding_box[3] + 1):
                    pos = np.array([x, y])
                    a, b, c = triangle.edge_func(pos, self.x_dim, self.y_dim)

                    if a <= 0 and b <= 0 and c <= 0:

                        area = a + b + c
                        a /= area
                        b /= area
                        c /= area

                        dist = 1 / (a / triangle.v1[2] + b / triangle.v2[2] + c / triangle.v3[2])

                        if (dist < self.distances[self.y_dim - y - 1, x -1]):
                            self.distances[self.y_dim - y - 1, x -1] = dist
                            self.canvas[self.y_dim - y - 1, x - 1] = np.asarray([color, color, color], dtype=np.uint8)

        print("Rendering image.")
        plt.imshow(self.canvas, extent=[-self.x_dim / 2, self.x_dim / 2, -self.y_dim/2,self.y_dim/2])
        plt.savefig("render.png")
        plt.show()

def render(file):
    mesh = stl.mesh.Mesh.from_file(file)
    avg = mesh.vectors.mean(axis=1)

    vals, vecs = np.linalg.eigh(np.matmul(avg.T, avg)) # get principle components

    # indices = np.argsort(vals)
    # vecs = vecs[indices]
    # vecs /= np.linalg.norm(vecs, axis=1)

    # print(vecs.shape, vals.shape)
    # print("vecs:", vecs, "vals:", vals, "norms:", np.linalg.norm(vecs, axis=0))

    # min = mesh.vectors.min(axis=0).min(axis=0)
    # max = mesh.vectors.max(axis=0).max(axis=0)

    #r = Renderer(min - (max - min) / 4, mean, np.array([0, 1, 0]), np.pi / 2)
    #r = Renderer(np.array([15, min[1] - (max[1] - min[1]) / 2, 4*mean[2]]), mean, np.array([0, 0, 1]), np.pi / 2)

    indices = np.random.choice(range(mesh.vectors.shape[0]), size = np.min([1000, mesh.vectors.shape[0]]), replace=False)
    sample = avg[indices]

    max = np.matmul(sample, vecs[:,0]).max()
    maxz = np.matmul(sample, vecs[:,2]).max()

    mean = avg.mean(axis=0)

    r = Renderer(vecs[:,0] * 1.5 * max, mean, np.array([0, 0, 1]), np.pi / 2)

    for polygon in mesh.vectors:
        r.addTriangle(polygon[0], polygon[1], polygon[2])

    r.clipTriangles()
    r.draw_triangles()

if __name__ == "__main__":
    render(sys.argv[1])
