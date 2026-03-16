import OpenGL.GL as g1
import OpenGL.GLU as g2
import numpy as np
import threading
import point_in_mesh as mpl

g = (1 + 5 ** .5) / 2


def initialize(width, height, cam, orientation=(0., 0., 0., 0.)):
    g2.gluPerspective(45, width / height, .1, 10.)
    g1.glTranslate(0.0, .0, -cam)
    g1.glRotatef(orientation[0], orientation[1], orientation[2], orientation[3])


def make_display():
    g1.glClear(g1.GL_COLOR_BUFFER_BIT | g1.GL_DEPTH_BUFFER_BIT)


def rotate(zoom, rx, ry, rz):
    g1.glRotatef(zoom, rx, ry, rz)


class Cube:
    def __init__(self):
        self.volume = 8
        self.status = 'cube'
        self.radius_outer_circle = 3 ** .5
        self.vertices = np.array((
            (1., -1., -1.),
            (1., 1., -1.),
            (-1., 1., -1.),
            (-1., -1., -1.),
            (1., -1., 1.),
            (1., 1., 1.),
            (-1., 1., 1.),
            (-1., -1., 1.),
        ))

        self.surfaces = ((0, 1, 2, 3),
                         (2, 3, 6, 7),
                         (6, 7, 4, 5),
                         (4, 5, 1, 0),
                         (5, 1, 2, 6),
                         (4, 0, 3, 7))
        self.edges = (
            (0, 1),
            (0, 3),
            (0, 4),
            (2, 1),
            (2, 3),
            (2, 6),
            (5, 1),
            (5, 4),
            (5, 6),
            (7, 4),
            (7, 3),
            (7, 6)
        )
        self.points = []
        for edge in self.edges:
            for vertex in edge:
                self.points.append(self.vertices[vertex])

    def draw(self):
        g1.glColor3d(0., 1., 1.)
        g1.glBegin(g1.GL_LINES)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()


class Tetrahedron:
    def __init__(self):
        self.volume = 8 / 3
        self.status = 'tetrahedron'
        self.radius_outer_circle = 3 ** .5
        self.vertices = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        self.surfaces = [(0, 1, 2),
                         (1, 2, 3),
                         (0, 2, 3),
                         (0, 1, 3)
                         ]
        self.edges = [(0, 1),
                      (0, 2),
                      (0, 3),
                      (1, 3),
                      (1, 2),
                      (2, 3)]
        self.points = []
        for edge in self.edges:
            for vertex in edge:
                self.points.append(self.vertices[vertex])

    def draw(self):
        g1.glColor3d(0., 1., 1.)
        g1.glBegin(g1.GL_LINES)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()


class Octahedron:
    def __init__(self):
        self.volume = 4 / 3
        self.status = 'octahedron'
        self.radius_outer_circle = 1
        self.vertices = [(0, 1, 0),
                         (0, -1, 0),
                         (1, 0, 0),
                         (-1, 0, 0),
                         (0, 0, -1),
                         (0, 0, 1),
                         ]
        self.edges = [(0, 2),
                      (0, 4),
                      (0, 5),
                      (0, 3),
                      (1, 2),
                      (1, 3),
                      (1, 4),
                      (1, 5),
                      (2, 5),
                      (5, 3),
                      (3, 4),
                      (4, 2)
                      ]
        self.surfaces = [(0, 3, 5),
                         (0, 3, 4),
                         (0, 2, 4),
                         (0, 2, 5),
                         (1, 3, 5),
                         (1, 5, 2),
                         (1, 2, 4),
                         (1, 3, 4)
                         ]
        self.points = []
        for edge in self.edges:
            for vertex in edge:
                self.points.append(self.vertices[vertex])

    def draw(self):
        g1.glColor3d(0., 1., 1.)
        g1.glBegin(g1.GL_LINES)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()


class Icosahedron:
    def __init__(self):
        self.status = 'icosahedron'
        self.volume = 10 * (3 + 5 ** .5) / 3
        self.radius_outer_circle = (g * g + 1) ** .5
        self.vertices = [(0, g, 1),
                         (0, g, -1),
                         (g, 1, 0),
                         (-g, 1, 0),
                         (0, -g, 1),
                         (0, -g, -1),
                         (g, -1, 0),
                         (-g, -1, 0),
                         (-1, 0, g),
                         (1, 0, g),
                         (1, 0, -g),
                         (-1, 0, -g)
                         ]
        self.edges = [(0, 1),
                      (0, 3),
                      (0, 2),
                      (0, 8),
                      (0, 9),
                      (1, 3),
                      (1, 2),
                      (1, 11),
                      (1, 10),
                      (3, 8),
                      (3, 7),
                      (3, 11),
                      (2, 9),
                      (2, 10),
                      (2, 6),
                      (4, 5),
                      (4, 6),
                      (4, 7),
                      (4, 8),
                      (4, 9),
                      (5, 6),
                      (5, 7),
                      (5, 10),
                      (5, 11),
                      (6, 9),
                      (6, 10),
                      (7, 8),
                      (7, 11),
                      (8, 9),
                      (10, 11)]
        self.surfaces = [(0, 1, 3),
                         (0, 1, 2),
                         (0, 3, 8),
                         (0, 8, 9),
                         (0, 2, 9),
                         (1, 3, 11),
                         (1, 2, 10),
                         (1, 11, 10),
                         (2, 9, 6),
                         (2, 6, 10),
                         (3, 7, 8),
                         (3, 7, 11),
                         (4, 5, 6),
                         (4, 6, 9),
                         (4, 7, 8),
                         (4, 8, 9),
                         (4, 5, 7),
                         (5, 10, 11),
                         (5, 7, 11),
                         (5, 6, 10)
                         ]
        self.points = []
        for edge in self.edges:
            for vertex in edge:
                self.points.append(self.vertices[vertex])

    def draw(self):
        g1.glColor3d(0., 1., 1.)
        g1.glBegin(g1.GL_LINES)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()


class Dodecahedron():
    def __init__(self):
        self.status = 'dodecahedron'
        self.volume = (15 + 7 * np.sqrt(5)) * 2 / (g ** 3)
        self.radius_outer_circle = (3 ** .5) * .5 * (1 + (5 ** .5)) / g
        self.vertices = [(1., 1., 1.),
                         (g, 1 / g, 0.),
                         (g, -1 / g, 0.),
                         (1., -1., 1.),
                         (1 / g, 0., g),
                         (-1 / g, 0., g),
                         (-1., -1., 1.),
                         (0., -g, 1 / g),
                         (0., -g, -1 / g),
                         (-1., -1., -1.),
                         (-g, -1 / g, 0.),
                         (1 / g, 0., -g),
                         (1., 1., -1.),
                         (1., -1., -1.),
                         (-1., 1., 1.),
                         (-g, 1 / g, 0.),
                         (-1., 1., -1.),
                         (0., g, -1 / g),
                         (0., g, 1 / g),
                         (-1 / g, 0, -g)]
        self.edges = [(0, 1),
                      (1, 2),
                      (2, 3),
                      (3, 4),
                      (4, 0),
                      (4, 5),
                      (5, 6),
                      (6, 7),
                      (3, 7),
                      (7, 8),
                      (8, 9),
                      (9, 10),
                      (10, 6),
                      (8, 13),
                      (2, 13),
                      (9, 19),
                      (11, 19),
                      (11, 13),
                      (11, 12),
                      (1, 12),
                      (10, 15),
                      (15, 14),
                      (5, 14),
                      (14, 18),
                      (18, 0),
                      (17, 18),
                      (17, 12),
                      (17, 16),
                      (16, 19),
                      (15, 16)]
        self.surfaces = [(0, 1, 2, 3, 4),
                         (3, 4, 5, 6, 7),
                         (6, 7, 8, 9, 10),
                         (2, 3, 7, 8, 13),
                         (8, 9, 19, 11, 13),
                         (1, 2, 13, 11, 12),
                         (5, 6, 10, 15, 14),
                         (15, 10, 9, 19, 16),
                         (19, 16, 17, 12, 11),
                         (12, 17, 18, 0, 1),
                         (0, 18, 14, 5, 4),
                         (14, 15, 16, 17, 18)]
        self.points = []
        for edge in self.edges:
            for vertex in edge:
                self.points.append(self.vertices[vertex])

    def draw(self):
        g1.glColor3d(0., 1., 1.)
        g1.glBegin(g1.GL_LINES)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()


class Points:
    def __init__(self, n, shape, com):
        self.status = 'point'
        self.length = n
        self.points = np.array([[None, None, None], ])
        self.points_inside_mesh = 0
        self.com = np.array(com)
        self.timer = threading.Timer(0.0, self.generate, args=(shape, n))
        self.timer.start()

    def draw(self):
        g1.glEnable(g1.GL_DEPTH_TEST)
        g1.glPointSize(2)
        g1.glBlendFunc(g1.GL_SRC_ALPHA, g1.GL_ONE_MINUS_SRC_ALPHA)
        g1.glEnable(g1.GL_BLEND)
        g1.glColor4f(1., 1., 1., 0.4)
        g1.glBegin(g1.GL_POINTS)
        for j in self.points:
            g1.glVertex3fv(j)
        g1.glEnd()
        g1.glDisable(g1.GL_BLEND)

    def generate(self, shape, n):
        for i in range(n):
            r, t1, t2 = shape.radius_outer_circle * np.random.default_rng().uniform(0, 1) ** (1 / 3), \
                        np.random.default_rng().uniform(0, 360) * np.pi / 180, \
                        np.arccos(2 * np.random.default_rng().uniform(0, 1) - 1)
            x, y, z = np.array([r * np.sin(t2) * np.cos(t1), r * np.sin(t2) * np.sin(t1), r * np.cos(t2)])
            if self.com[0] > 0:
                if x > 0:
                    x *= -1
            if self.com[0] < 0:
                if x < 0:
                    x *= -1
            if self.com[1] > 0:
                if y > 0:
                    y *= -1
            if self.com[1] < 0:
                if y < 0:
                    y *= -1
            if self.com[2] > 0:
                if z > 0:
                    z *= -1
            if self.com[2] < 0:
                if z < 0:
                    z *= -1
            v = np.array([x, y, z])
            if self.points[0][0] == None:
                self.points[0][0], self.points[0][0], self.points[0][0] = v[0], v[1], v[2]
            else:
                self.points = np.append(self.points, [v], axis=0)
            self.com = (self.com * len(self.points) + v) / (len(self.points) + 1)
            if mpl.check_point_inside_solid(shape.surfaces, shape.vertices, v, shape.status):
                self.points_inside_mesh += 1
        self.timer.cancel()


