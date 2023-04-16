import os
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
            if e.type != EdgeType.INNER:
                self.type = VertexType.FRONT
                return
        self.type = VertexType.INNER

    def __repr__(self):
        return f"{self.idx}, {self.point}, {self.type}"

    def __str__(self):
        return self.__repr__()


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
                if np.dot(tr_norm, pt_norm) < -1e-12:
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

    def __repr__(self):
        return f"({self.source.idx}, {self.target.idx}, {self.type})"

    def __str__(self):
        return self.__repr__()


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

    circum_center = np.cross(
        np.dot(p10, p10) * p20 - np.dot(p20, p20) * p10,
        n
    ) / 2 / np.dot(n, n) + p0

    # print(f"The circum_center is {circum_center} and the point p1 is {p1}, the dot product of the difference is {
    # np.dot(circum_center-p1, circum_center-p1)}")
    if (radius ** 2 - np.dot(circum_center - p0, circum_center - p0)) < 0:
        return None
    t1 = np.sqrt((radius ** 2 - np.dot(circum_center - p0, circum_center - p0)))
    if np.dot(n, v0.normal) < 0:
        n *= -1
    n /= np.linalg.norm(n)
    center_of_ball = circum_center + n * t1
    return center_of_ball


def get_triangle_norm(v0: Vertex, v1: Vertex, v2: Vertex) -> np.ndarray:
    p0, p1, p2 = v0.point, v1.point, v2.point
    n = np.cross(p1 - p0, p2 - p0)
    return n / np.linalg.norm(n)


def is_coplanar(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    n = np.cross(p1 - p0, p2 - p0)
    return np.dot(n, p3 - p0) == 0


def segment_intersection(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
    if not is_coplanar(p0, p1, p2, p3):
        return False
    da = p1 - p0
    db = p3 - p2
    dp = p0 - p2
    dap = np.cross(da, np.cross(da, db))
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    if denom == 0:
        return False
    return 0 <= num / denom <= 1


def get_linking_edge(v0: Vertex, v1: Vertex) -> Edge or None:
    for e in v0.edges:
        for e2 in v1.edges:
            if e.source.idx == e2.source.idx and e.target.idx == e2.target.idx:
                return e
    return None


# is the triangle norm consistent with the point normals?
def test_triangle_norm(v0: Vertex, v1: Vertex, v2: Vertex) -> bool:
    triangle_norm = get_triangle_norm(v0, v1, v2)
    if np.dot(triangle_norm, v0.normal) < 0:
        triangle_norm *= -1
    if np.dot(triangle_norm, v1.normal) < 0 or np.dot(triangle_norm, v2.normal) < 0:
        return False
    return True


class BallPivot:
    def __init__(self, point_cloud: o3d.geometry.PointCloud):
        self.kd = o3d.geometry.KDTreeFlann(point_cloud)

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = point_cloud.points
        self.mesh.vertex_normals = point_cloud.normals
        self.mesh.vertex_colors = point_cloud.colors

        self.vertices = [Vertex(i, p, n) for i, (p, n) in enumerate(zip(point_cloud.points, point_cloud.normals))]
        self.edge_front = set()
        self.border_edges = []
        self.possible_seed = set(i for i in range(len(self.vertices)))

    def test_triangle_ball(self, v0, v1, v2, radius):
        center_of_ball = get_center_of_ball(v0, v1, v2, radius)
        if center_of_ball is None:
            return False
        num_results, idxs, _ = self.kd.search_radius_vector_3d(center_of_ball, radius)
        if num_results > 3:
            return False
        elif num_results <= 3:
            for idx in idxs:
                if idx not in [v0.idx, v1.idx, v2.idx]:
                    return False
        return True

    def test_triangle(self, v0, v1, v2, radius):
        if not test_triangle_norm(v0, v1, v2):
            return False
        e0 = get_linking_edge(v0, v1)
        e1 = get_linking_edge(v1, v2)
        if e0 is not None and e0.type == EdgeType.INNER:
            return False
        if e1 is not None and e1.type == EdgeType.INNER:
            return False

        if not self.test_triangle_ball(v0, v1, v2, radius):
            return False
        return True

    def try_seed(self, v0, radius):
        p1 = v0.point
        num_results, index_vec, distance2_vec = self.kd.search_radius_vector_3d(np.array(p1), radius * 2)
        if num_results < 3:
            return False
        # consider all sigma_a, sigma_b pairs
        # they just can't be the same, and they can't be orphans
        for i, idx_1 in enumerate(index_vec):
            v1 = self.vertices[idx_1]
            if (idx_1 == v0.idx) or (v1.type != VertexType.ORPHAN):
                continue
            candidate = None
            for j in range(i + 1, num_results):
                idx_2 = index_vec[j]
                v2 = self.vertices[idx_2]
                if (idx_2 == idx_1) or (idx_2 == v0.idx) or (v2.type != VertexType.ORPHAN):
                    continue
                # is the triangle norm consistent with the point normals?
                if self.test_triangle(v0, v1, v2, radius):
                    # good seed
                    candidate = v2
                    break
            if candidate is not None:
                e0 = get_linking_edge(v0, candidate)
                e1 = get_linking_edge(v1, candidate)
                e2 = get_linking_edge(v0, v1)
                if e0 is not None and e0.type != EdgeType.FRONT:
                    continue
                if e1 is not None and e1.type != EdgeType.FRONT:
                    continue
                if e2 is not None and e2.type != EdgeType.FRONT:
                    continue
                center = get_center_of_ball(v0, v1, candidate, radius)
                self.create_triangle(v0, v1, candidate, center)
                e0 = get_linking_edge(v0, candidate)
                e1 = get_linking_edge(v1, candidate)
                e2 = get_linking_edge(v0, v1)
                if e0.type == EdgeType.FRONT:
                    self.edge_front.add(e0)
                if e1.type == EdgeType.FRONT:
                    self.edge_front.add(e1)
                if e2.type == EdgeType.FRONT:
                    self.edge_front.add(e2)
                if len(self.edge_front) > 0:
                    return True
        return False

    def find_seed_triangle(self, radius):
        for i, p_idx in enumerate(self.possible_seed):
            print(f"\r fs {i:5d} | {'#'*int(i/len(self.possible_seed)*50):50s} | {i/len(self.possible_seed)*100:5.2f}%", end="")
            if self.try_seed(self.vertices[p_idx], radius):
                self.expand_triangulation(radius)
        self.possible_seed = set(i for i in range(len(self.vertices)) if self.vertices[i].type == VertexType.ORPHAN)

    def create_triangle(self, v0: Vertex, v1: Vertex, v2: Vertex, ball_center: np.ndarray):
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
        if np.dot(norm, v0.normal) < 0:
            norm *= -1

        if np.dot(norm, v1.normal) > 0:
            self.mesh.triangles.append([v0.idx, v1.idx, v2.idx])
        else:
            self.mesh.triangles.append([v0.idx, v2.idx, v1.idx])
        self.mesh.triangle_normals.append(norm)

    def find_candidate_vertex(self, pivot_edge: Edge, radius) -> tuple[Vertex or None, np.ndarray or None]:
        src = pivot_edge.source
        target = pivot_edge.target
        opposite = pivot_edge.get_opposite_vertex()
        if opposite is None:
            raise Exception("pivot edge has no opposite vertex")

        midpoint = 0.5 * (src.point + target.point)

        triangle = pivot_edge.triangle0
        center = triangle.ball_center
        v = target.point - src.point
        v /= np.linalg.norm(v)
        a = center - midpoint
        a /= np.linalg.norm(a)

        _, idxs, dst2 = self.kd.search_radius_vector_3d(midpoint, radius * 2)
        min_candidate = None
        candidate_center = None
        min_angle = 2 * np.pi + 1

        pts = np.array([self.vertices[i].point for i in idxs])
        debug = False
        for idx in idxs:
            candidate = self.vertices[idx]
            if idx == src.idx or idx == target.idx or idx == opposite.idx:
                continue
            if is_coplanar(src.point, target.point, opposite.point, candidate.point) and (
                    segment_intersection(midpoint.point, candidate.point, src.point, opposite.point) or
                    segment_intersection(midpoint.point, candidate.point, target.point, opposite.point)
            ):
                print("coplanar and intersecting")
                continue
            new_center = get_center_of_ball(src, target, candidate, radius)
            if new_center is None:
                continue

            b = new_center - midpoint
            b /= np.linalg.norm(b)

            cos_angle = np.dot(a, b).clip(-1, 1)
            angle = np.arccos(cos_angle)

            c = np.cross(a, b)
            if np.dot(c, v) < -1e-12:
                angle = 2 * np.pi - angle
            if angle >= min_angle:
                continue

            if not self.test_triangle_ball(src, target, candidate, radius):
                continue

            min_angle = angle
            min_candidate = candidate
            candidate_center = new_center

        return min_candidate, candidate_center

    def expand_triangulation(self, radius):
        l = len(self.edge_front)
        while self.edge_front:
            l = max(l, len(self.edge_front))
            print(f"\r ex {l-len(self.edge_front):5d} | {'#'*int((l-len(self.edge_front))/l*50):50s} | {(l-len(self.edge_front))/l*100:5.2f}%", end="")
            edge = self.edge_front.pop()
            if edge.type != EdgeType.FRONT:
                continue

            candidate, candidate_center = self.find_candidate_vertex(edge, radius)
            if candidate is None or \
                    candidate.type == VertexType.INNER or \
                    not test_triangle_norm(candidate, edge.source, edge.target):
                edge.type = EdgeType.BORDER
                self.border_edges.append(edge)
                continue

            e0 = get_linking_edge(candidate, edge.source)
            e1 = get_linking_edge(candidate, edge.target)

            if (e0 is not None and e0.type != EdgeType.FRONT) or (e1 is not None and e1.type != EdgeType.FRONT):
                edge.type = EdgeType.BORDER
                self.border_edges.append(edge)
                continue

            self.create_triangle(edge.source, candidate, edge.target, candidate_center)
            # print(f'new triangle: {edge.source.idx} {candidate.idx} {edge.target.idx}')

            e0 = get_linking_edge(candidate, edge.source)
            e1 = get_linking_edge(candidate, edge.target)
            if e0.type == EdgeType.FRONT:
                self.edge_front.add(e0)
            if e1.type == EdgeType.FRONT:
                self.edge_front.add(e1)

    def run(self, radii: list[float]):
        self.mesh.triangles.clear()
        for radius in radii:
            print(f"radius: {radius}")
            i = 0
            while i < len(self.border_edges):
                edge = self.border_edges[i]
                triangle = edge.triangle0
                empty_ball = self.test_triangle_ball(triangle.v0, triangle.v1, triangle.v2, radius)
                if empty_ball:
                    edge.type = EdgeType.FRONT
                    self.edge_front.add(edge)
                    self.border_edges.remove(edge)
                    continue
                i += 1
            if not self.edge_front:
                self.find_seed_triangle(radius)
            else:
                self.expand_triangulation(radius)
            print()
        return self.mesh


if __name__ == "__main__":
    mesh = o3d.io.read_triangle_mesh('famous_original/03_meshes/dragon.ply')
    # bunny = o3d.data.BunnyMesh()
    # mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=50000)

    bpa = BallPivot(point_cloud)
    radii = [0.005, 0.01]
    mesh = bpa.run(radii)
    print(f'number of triangles: {len(mesh.triangles)}')
    o3d.visualization.draw_geometries([mesh, point_cloud], mesh_show_back_face=True)
    # main()