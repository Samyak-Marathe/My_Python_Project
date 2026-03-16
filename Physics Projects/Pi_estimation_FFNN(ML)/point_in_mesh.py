import numpy as np

limit = 10


def check_point_intersects_face(p, plane, limit):
    r = np.array([1, 0, 0])
    s = plane[0]
    n = np.cross(plane[0] - plane[1], plane[1] - plane[2])
    n = n / np.linalg.norm(n)

    try:
        x, y, z = (r * (np.dot((s - p), n) / np.dot(r, n)) + p)
    except RuntimeWarning:
        return False

    x, y, z = round(x, 6), round(y, 6), round(z, 6)
    r1, r2, r3 = plane[0] - np.array([x, y, z]), plane[1] - np.array([x, y, z]), plane[2] - np.array([x, y, z])
    if len(plane) > 3:
        r4, r5 = plane[3] - np.array([x, y, z]), plane[4] - np.array([x, y, z])
    if (x, y, z) == (p[0], p[1], p[2]) or \
            ((r1[0], r1[1], r1[2]) == (0, 0, 0) or
             (r2[0], r2[1], r2[2]) == (0, 0, 0) or
             (r3[0], r3[1], r3[2]) == (0, 0, 0)):
        return True
    if len(plane) > 3 and ((r4[0], r4[1], r4[2]) == (0, 0, 0) or (r5[0], r5[1], r5[2]) == (0, 0, 0)):
        return True
    if len(plane) == 3:
        angle = np.array([
            [(r1[0] * r2[0]) + (r1[1] * r2[1]) + (r1[2] * r2[2]), np.linalg.norm(r1), np.linalg.norm(r2)],
            [(r2[0] * r3[0]) + (r2[1] * r3[1]) + (r2[2] * r3[2]), np.linalg.norm(r2), np.linalg.norm(r3)],
            [(r3[0] * r1[0]) + (r3[1] * r1[1]) + (r3[2] * r1[2]), np.linalg.norm(r1), np.linalg.norm(r3)]
        ])
    else:
        angle = np.array([
            [(r1[0] * r2[0]) + (r1[1] * r2[1]) + (r1[2] * r2[2]), np.linalg.norm(r1), np.linalg.norm(r2)],
            [(r2[0] * r3[0]) + (r2[1] * r3[1]) + (r2[2] * r3[2]), np.linalg.norm(r2), np.linalg.norm(r3)],
            [(r3[0] * r4[0]) + (r3[1] * r4[1]) + (r3[2] * r4[2]), np.linalg.norm(r3), np.linalg.norm(r4)],
            [(r4[0] * r5[0]) + (r4[1] * r5[1]) + (r4[2] * r5[2]), np.linalg.norm(r4), np.linalg.norm(r5)],
            [(r5[0] * r1[0]) + (r5[1] * r1[1]) + (r5[2] * r1[2]), np.linalg.norm(r5), np.linalg.norm(r1)]
        ])
    t = np.arccos(round(angle[0][0] / (angle[0][1] * angle[0][2]), 13)) + np.arccos(
        round(angle[1][0] / (angle[1][1] * angle[1][2]), 13)) + np.arccos(
        round(angle[2][0] / (angle[2][1] * angle[2][2]), 13))
    if len(plane) > 3:
        t += np.arccos(round(angle[3][0] / (angle[3][1] * angle[3][2]), 13)) + np.arccos(
            round(angle[4][0] / (angle[4][1] * angle[4][2]), 13))
    if round(t * 180 / np.pi, 6) == 360 and ((p[0] <= x <= limit) or (limit <= x <= p[0])):
        return True

    return False


def check_point_inside_solid(face_indices, points, point, shape):
    if shape != 'cube':
        counter_ray1, counter_ray2 = 0, 0
        for i in face_indices:
            if len(i) > 3:
                faces = np.array([points[i[0]], points[i[1]], points[i[2]], points[i[3]], points[i[4]]])
            else:
                faces = np.array([points[i[0]], points[i[1]], points[i[2]]])
            if not counter_ray1:
                counter_ray1 = check_point_intersects_face(point, faces, limit)
            if not counter_ray2:
                counter_ray2 = check_point_intersects_face(point, faces, -limit)
            if counter_ray1 and counter_ray2:
                return 1
        return 0
    else:
        if -1. <= point[0] <= 1. and -1. <= point[1] <= 1. and -1. <= point[2] <= 1.:
            return 1
        return 0
