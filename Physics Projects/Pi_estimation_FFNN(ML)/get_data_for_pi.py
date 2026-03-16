import numpy as np
import point_in_mesh as mpl
import csv
import time

g = (1 + 5 ** .5) / 2


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


class Tetrahedron:
    def __init__(self):
        self.volume = 8 / 3
        self.status = 'tetrahedron'
        self.radius_outer_circle = 3 ** .5
        self.vertices = np.array([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)])
        self.surfaces = [(0, 1, 2),
                         (1, 2, 3),
                         (0, 2, 3),
                         (0, 1, 3)
                         ]
        self.edges = [(0, 1),
                      (0, 2),
                      (0, 2),
                      (0, 3),
                      (1, 3),
                      (1, 2),
                      (2, 3)]


class Octahedron:
    def __init__(self):
        self.volume = 4 / 3
        self.status = 'octahedron'
        self.radius_outer_circle = 1
        self.vertices = np.array([(0, 1, 0),
                                  (0, -1, 0),
                                  (1, 0, 0),
                                  (-1, 0, 0),
                                  (0, 0, -1),
                                  (0, 0, 1),
                                  ])
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


class Dodecahedron:
    def __init__(self):
        self.status = 'dodecahedron'
        self.volume = (15 + 7 * np.sqrt(5)) * 2 / (g ** 3)
        self.radius_outer_circle = (3 ** .5) * .5 * (1 + (5 ** .5)) / g
        self.vertices = np.array([(1., 1., 1.),
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
                                  (-1 / g, 0, -g)])
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


class Icosahedron:
    def __init__(self):
        self.status = 'icosahedron'
        self.volume = 10 * (3 + 5 ** .5) / 3
        self.radius_outer_circle = (g * g + 1) ** .5
        self.vertices = np.array([(0, g, 1),
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
                                  ])
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


class Sphere:
    def __init__(self, r):
        self.status = 'sphere'
        self.radius = r
        self.volume = 4 * np.pi * r * r * r / 3


def generate_points(shape, number_of_points):
    global com
    data = []
    file_name = shape.status + '_data_' + str(number_of_points) + '.csv'
    for i in range(number_of_points):
        rng = np.random.default_rng()
        r = shape.radius_outer_circle * rng.uniform(0, 1) ** (1 / 3)
        t1 = rng.uniform(0, 360) * np.pi / 180
        t2 = np.arccos(2 * rng.uniform(0, 1) - 1)
        x = r * np.sin(t2) * np.cos(t1)
        y = r * np.sin(t2) * np.sin(t1)
        z = r * np.cos(t2)
        if com[0] > 0:
            if x > 0:
                x *= -1
        if com[0] < 0:
            if x < 0:
                x *= -1
        if com[1] > 0:
            if y > 0:
                y *= -1
        if com[1] < 0:
            if y < 0:
                y *= -1
        if com[2] > 0:
            if z > 0:
                z *= -1
        if com[2] < 0:
            if z < 0:
                z *= -1
        v = np.array([x, y, z])
        com = ((com * (i + 1)) + v) / (i + 1)

        inside = 0
        if shape.status != 'cube':
            if mpl.check_point_inside_solid(shape.surfaces, shape.vertices.get(), v.get()):
                inside = 1
        elif shape.status == 'cube':
            if -1. <= v[0] <= 1. and -1. <= v[1] <= 1. and -1. <= v[2] <= 1.:
                inside = 1
        if i % 100 == 0:
            print(f'{i} / {number_of_points} points generated for {shape.status}.')

        data.append([x, y, z, inside])

    write_data_to_csv(f'{path}/' + file_name, data, time.perf_counter() - t)


def write_data_to_csv(file_path, data, t):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['X', 'Y', 'Z', 'Label'])
        writer.writerows(data)
        writer.writerow(['t', t])


com = np.array([0, 0, 0])
shape_array = [Tetrahedron(), Icosahedron(), Octahedron(), Cube(), Dodecahedron()]
print('Select shape')
for i in range(0, len(shape_array)):
    print(str(i) + ". " + shape_array[i].status)

path = int(input('Enter 1 if you want to generate training data. For generating testing data, press 0.'))
if n:
    path = 'training_data'
else:
    path = 'testing_data'
shape_index = input('Enter shape number: ')
shape = shape_array[int(shape_index)]
total_points = int(input("Enter total points to iterate over: "))
t = time.perf_counter()
generate_points(shape, total_points)
print('COM:', com)
print('Time taken:', time.perf_counter() - t)