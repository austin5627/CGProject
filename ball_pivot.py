import open3d as o3d
import numpy as np


def get_center_of_ball(triangle_norm, p1, p2, p3, radius):
    p21 = p2 - p1
    p31 = p3 - p1
    n = np.cross(p21, p31)

    circum_center = np.cross(
        np.dot(p21, p21) * p31 - np.dot(p31, p31) * p21,
        n
    ) / 2 / np.dot(n, n) + p1

    # print(f"The circum_center is {circum_center} and the point p1 is {p1}, the dot product of the difference is {
    # np.dot(circum_center-p1, circum_center-p1)}")
    if (radius ** 2 - np.dot(circum_center - p1, circum_center - p1)) < 0:
        return None
    t1 = np.sqrt((radius ** 2 - np.dot(circum_center - p1, circum_center - p1)))
    center_of_ball = circum_center + triangle_norm * t1
    return center_of_ball


# is the triangle norm consistent with the point normals?
def test_triangle_norm(sigma_idx, s_a, s_b):
    triangle_norm = np.cross(PointsNP[s_a] - PointsNP[sigma_idx], PointsNP[s_b] - PointsNP[sigma_idx])
    if np.dot(triangle_norm, NormalsNP[sigma_idx]) < 0:
        triangle_norm *= -1
    if (np.dot(triangle_norm, NormalsNP[s_a]) > 0) and (np.dot(triangle_norm, NormalsNP[s_b]) > 0):
        # print(f"found valid triangle norm for points p1 {PointsNP[sigma_idx]}|{NormalsNP[sigma_idx]}, p2 {PointsNP[
        # s_a]}|{NormalsNP[s_a]}, p3{PointsNP[s_b]}|{NormalsNP[s_b]}")
        triangle_norm /= np.linalg.norm(triangle_norm)
        return True, triangle_norm
    return False, None


def test_triangle_ball(triangle_norm, p1, p2, p3, radius):
    center_of_ball = get_center_of_ball(triangle_norm, p1, p2, p3, radius)
    if center_of_ball is None:
        return False
    num_results, _, _ = kd.search_radius_vector_3d(center_of_ball, radius)
    if num_results > 0:
        return False
    return True


def try_seed(sigma_idx, radius):
    p1 = PointsNP[sigma_idx]
    num_results, index_vec, distance2_vec = kd.search_radius_vector_3d(np.array(p1), radius * 2)
    if num_results < 3:
        return None, None
    # consider all sigma_a, sigma_b pairs
    # they just can't be the same, and they can't be orphans
    for i, s_a in enumerate(index_vec):
        if (s_a == sigma_idx) or (s_a in orphans):
            continue
        p2 = PointsNP[s_a]
        for j in range(i + 1, num_results):
            s_b = index_vec[j]
            if (s_b == s_a) or (s_b == sigma_idx) or (s_b in orphans):
                continue
            p3 = PointsNP[s_b]
            # is the triangle norm consistent with the point normals?
            tnorm_is_valid, triangle_norm = test_triangle_norm(sigma_idx, s_a, s_b)
            if tnorm_is_valid and test_triangle_ball(triangle_norm, p1, p2, p3, radius):
                # good seed
                return s_a, s_b
    return None, None


def find_seed_triangle(radius):
    for p_idx in possible_seed:
        if p_idx not in orphans:
            p2_idx, p3_idx = try_seed(p_idx, radius)
            if p2_idx is not None:
                return p_idx, p2_idx, p3_idx
    # no good seed, terminate algorithm
    return None, None, None


def main():
    radius = 0.01
    p1_idx, p2_idx, p3_idx = find_seed_triangle(radius)
    if p1_idx is None:
        print("No seed triangle found, terminating algorithm")
        exit(0)
    print(f"Found seed triangle with points {p1_idx}, {p2_idx}, {p3_idx}")
    _, normal = test_triangle_norm(p1_idx, p2_idx, p3_idx)
    p1 = PointsNP[p1_idx]
    p2 = PointsNP[p2_idx]
    p3 = PointsNP[p3_idx]
    center = get_center_of_ball(normal, p1, p2, p3, radius)

    # make a mesh with the seed triangle
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([p1, p2, p3]))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    mesh.paint_uniform_color([1, 0.706, 0])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(center)
    o3d.visualization.draw_geometries([sphere, mesh, point_cloud], mesh_show_back_face=True)


mesh = o3d.io.read_triangle_mesh('famous_original/03_meshes/dragon.ply')
mesh.compute_vertex_normals()
point_cloud = mesh.sample_points_uniformly(number_of_points=100000)
PointsNP = np.asarray(point_cloud.points)
NormalsNP = np.asarray(point_cloud.normals)
orphans = []
possible_seed = set(i for i in range(len(PointsNP)))
kd = o3d.geometry.KDTreeFlann(point_cloud)

if __name__ == "__main__":
    main()