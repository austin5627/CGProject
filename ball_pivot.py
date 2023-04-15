import open3d as o3d
import numpy as np

from enum import Enum


class VertexType(Enum):
    ORPHAN = 0
    FRONT = 1
    INNER = 2


class EdgeType(Enum):
    BORDER = 0
    FRONT = 1
    INNER = 2


class Vertex:
    def __init__(self, idx, point, normal):
        self.idx = idx
        self.point = point
        self.normal = normal
        self.edges = []
        self.type = VertexType.ORPHAN

    def update_type(self):
        if len(self.edges) == 0:
            self.type = VertexType.ORPHAN
        for e in self.edges:
            if e.type == 0:
                self.type = VertexType.FRONT
                return
        self.type = VertexType.INNER


class Edge:
    def __init__(self, v0, v1):
        self.source = v0
        self.target = v1
        self.triangle0 = None
        self.triangle1 = None
        self.type = EdgeType.FRONT

    def add_triangle(self, t):
        if t != self.triangle0 and t != self.triangle1:
            if self.triangle0 is None:
                self.triangle0 = t
                self.type = EdgeType.FRONT
                opp = self.get_opposite_vertex()
                tr_norm = np.cross(self.target.point - self.source.point, opp.point - self.source.point)
                pt_norm = self.source.normal + self.target.normal + opp.normal
                if np.dot(tr_norm, pt_norm) < 0:
                    self.target, self.source = self.source, self.target
            elif self.triangle1 is None:
                self.triangle1 = t
                self.type = EdgeType.INNER
            else:
                raise Exception("Edge already has two triangles")

    def get_opposite_vertex(self):
        if self.triangle0 is None:
            return None
        if self.triangle0.v0.idx != self.source.idx and self.triangle0.v0.idx != self.target.idx:
            return self.triangle0.v0
        elif self.triangle0.v1.idx != self.source.idx and self.triangle0.v1.idx != self.target.idx:
            return self.triangle0.v1
        else:
            return self.triangle0.v2


class Triangle:
    def __init__(self, v0, v1, v2, ball_center):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.ball_center = ball_center


def get_center_of_ball(v0: Vertex, v1: Vertex, v2: Vertex, radius: float) -> np.ndarray or None:
    p0, p1, p2 = v0.point, v1.point, v2.point
    p10 = p1 - p0
    p20 = p2 - p0
    n = np.cross(p10, p20)
    if np.dot(n, v0.normal) < 0:
        n *= -1

    circum_center = np.cross(
        np.dot(p10, p10) * p20 - np.dot(p20, p20) * p10,
        n
    ) / 2 / np.dot(n, n) + p0

    # print(f"The circum_center is {circum_center} and the point p1 is {p1}, the dot product of the difference is {
    # np.dot(circum_center-p1, circum_center-p1)}")
    if (radius ** 2 - np.dot(circum_center - p0, circum_center - p0)) < 0:
        return None
    t1 = np.sqrt((radius ** 2 - np.dot(circum_center - p0, circum_center - p0)))
    center_of_ball = circum_center + n * t1
    return center_of_ball


def get_triangle_norm(v0: Vertex, v1: Vertex, v2: Vertex) -> np.ndarray:
    p0, p1, p2 = v0.point, v1.point, v2.point
    n = np.cross(p1 - p0, p2 - p0)
    return n / np.linalg.norm(n)


# is the triangle norm consistent with the point normals?
def test_triangle_norm(v0: Vertex, v1: Vertex, v2: Vertex) -> bool:
    triangle_norm = get_triangle_norm(v0, v1, v2)
    if np.dot(triangle_norm, v0.normal) < 0:
        triangle_norm *= -1
    if np.dot(triangle_norm, v1.normal) < 0 or np.dot(triangle_norm, v2.normal) < 0:
        return False
    return True


def test_triangle_ball(v0, v1, v2, radius):
    center_of_ball = get_center_of_ball(v0, v1, v2, radius)
    if center_of_ball is None:
        return False
    num_results, _, _ = kd.search_radius_vector_3d(center_of_ball, radius)
    if num_results > 0:
        return False
    return True


def try_seed(v0, radius):
    p1 = v0.point
    num_results, index_vec, distance2_vec = kd.search_radius_vector_3d(np.array(p1), radius * 2)
    if num_results < 3:
        print("Not enough points to seed")
        return None, None
    # consider all sigma_a, sigma_b pairs
    # they just can't be the same, and they can't be orphans
    for i, idx_1 in enumerate(index_vec):
        if (idx_1 == v0.idx) or (idx_1 in orphans):
            continue
        v1 = vertices[idx_1]
        for j in range(i + 1, num_results):
            idx_2 = index_vec[j]
            if (idx_2 == idx_1) or (idx_2 == v0.idx) or (idx_2 in orphans):
                continue
            v2 = vertices[idx_2]
            # is the triangle norm consistent with the point normals?
            if test_triangle_norm(v0, v1, v2):
                if test_triangle_ball(v0, v1, v2, radius):
                    # good seed
                    return idx_1, idx_2
                else:
                    print(f"Bad ball: {v0.idx}, {idx_1}, {idx_2}")
            else:
                print(f"Bad triangle: {v0.idx}, {idx_1}, {idx_2}")
    return None, None


def find_seed_triangle(radius):
    for p_idx in possible_seed:
        if p_idx not in orphans:
            print(f"Trying seed {p_idx}...")
            p2_idx, p3_idx = try_seed(vertices[p_idx], radius)
            if p2_idx is not None:
                return p_idx, p2_idx, p3_idx
    # no good seed, terminate algorithm
    return None, None, None


def get_linking_edge(v0: Vertex, v1: Vertex) -> Edge or None:
    for e in v0.edges:
        for e2 in v1.edges:
            if e.source.idx == e2.source.idx and e.target.idx == e2.target.idx:
                return e
    return None


def create_triangle(v0: Vertex, v1: Vertex, v2: Vertex, ball_center: np.ndarray):
    t = Triangle(v0, v1, v2, ball_center)
    e0 = get_linking_edge(v0, v1)
    if e0 is None:
        e0 = Edge(v0, v1)
        v0.edges.append(e0)
        v1.edges.append(e0)
    e0.add_triangle(t)

    e1 = get_linking_edge(v1, v2)
    if e1 is None:
        e1 = Edge(v1, v2)
        v1.edges.append(e1)
        v2.edges.append(e1)
    e1.add_triangle(t)

    e2 = get_linking_edge(v2, v0)
    if e2 is None:
        e2 = Edge(v2, v0)
        v2.edges.append(e2)
        v0.edges.append(e2)
    e2.add_triangle(t)

    v0.update_type()
    v1.update_type()
    v2.update_type()

    norm = get_triangle_norm(v0, v1, v2)


def main():
    radius = 0.01
    p1_idx, p2_idx, p3_idx = find_seed_triangle(radius)
    if p1_idx is None:
        print("No seed triangle found, terminating algorithm")
        o3d.visualization.draw_geometries([point_cloud])
        exit(0)
    print(f"Found seed triangle with points {p1_idx}, {p2_idx}, {p3_idx}")
    v1, v2, v3 = vertices[p1_idx], vertices[p2_idx], vertices[p3_idx]
    center = get_center_of_ball(v1, v2, v3, radius)

    # make a mesh with the seed triangle
    tri_mesh = o3d.geometry.TriangleMesh()
    tri_mesh.vertices = o3d.utility.Vector3dVector(np.array([v1.point, v2.point, v3.point]))
    tri_mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    tri_mesh.paint_uniform_color([1, 0.706, 0])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(center)
    o3d.visualization.draw_geometries([sphere, tri_mesh, point_cloud], mesh_show_back_face=True)


mesh = o3d.io.read_triangle_mesh('famous_original/03_meshes/dragon.ply')
mesh.compute_vertex_normals()
point_cloud = mesh.sample_points_uniformly(number_of_points=500)
vertices = [Vertex(i, p, n) for i, (p, n) in enumerate(zip(point_cloud.points, point_cloud.normals))]
orphans = set()
possible_seed = set(i for i in range(len(vertices)))
kd = o3d.geometry.KDTreeFlann(point_cloud)

if __name__ == "__main__":
    main()