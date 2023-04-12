import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
pcd = o3d.geometry.PointCloud()

mesh = o3d.io.read_triangle_mesh('famous_original/03_meshes/dragon.ply')
# print(mesh)
# print("no vertex normals makes visualization not good")
# o3d.visualization.draw_geometries([mesh])


mesh.compute_vertex_normals()
# print("having vertex normals gives depth")
# o3d.visualization.draw_geometries([mesh])


# sample a point cloud from the mesh to work with
pointcloud = mesh.sample_points_uniformly(number_of_points=2500)
# o3d.visualization.draw_geometries([pointcloud], point_show_normal=True)

PointsNP = np.asarray(pointcloud.points)
NormalsNP = np.asarray(pointcloud.normals)
orphans = []

kd = o3d.geometry.KDTreeFlann(pointcloud)

seed_point_bool_dict = dict.fromkeys(range(0, len(PointsNP)), True)


def GetCenterOfBall(triangle_norm, sigma, s_a, s_b, radius):
    # triangle_norm is np.cross(p21, p31)
    # p1 is sigma
    # p2 is s_a
    # p3 is s_b
    p21 = PointsNP[s_a] - sigma
    p31 = PointsNP[s_b] - sigma

    circumcenter = np.cross(
        np.dot(p21, p21) * p31 - np.dot(p31, p31) * p21,
        triangle_norm
        ) / 2 / np.dot(triangle_norm, triangle_norm) + sigma

    t1 = np.sqrt((radius**2 - np.dot(circumcenter-sigma, circumcenter-sigma))/ np.dot(triangle_norm, triangle_norm))
    center_of_ball = circumcenter + triangle_norm * t1
    return center_of_ball

# is the triangle norm consistent with the point normals?
def TestTriangleNorm(sigma, s_a, s_b):
    triangle_norm = np.cross(PointsNP[s_a] - sigma, PointsNP[s_b] - sigma)
    if (np.dot(triangle_norm, sigma) < 0):
        triangle_norm *= -1
    if (np.dot(triangle_norm, PointsNP[s_a]) > 0) and (np.dot(triangle_norm, PointsNP[s_b]) > 0):
        return True, triangle_norm
    return False, None

def TestTriangleBall(triangle_norm, p1, p2, p3, radius):
    center_of_ball = GetCenterOfBall(triangle_norm, p1, p2, p3, radius)
    num_results, _, _ = kd.search_radius_vector_3d(center_of_ball, radius)
    if(num_results > 3):
        return False
    return True

def TrySeed(sigma_idx, sigma, radius):
    num_results, index_vec, distance2_vec = kd.search_radius_vector_3d(np.array(sigma), radius * 2)
    if (num_results < 3):
        return False
    # consider all sigma_a, sigma_b pairs
    # they just can't be the same, and they can't be orphans
    for s_a in index_vec:
        if (s_a == sigma_idx) or (s_a in orphans):
            continue
        for s_b in index_vec:
            if (s_b == s_a) or (s_b == sigma_idx) or (s_b in orphans):
                continue
            # is the triangle norm consistent with the point normals?
            tnorm_is_valid, triangle_norm = TestTriangleNorm(sigma, s_a, s_b)
            if tnorm_is_valid and TestTriangleBall(triangle_norm, sigma, s_a, s_b, radius):
                # good seed
                return True
    return False


def FindSeedTriangle(radius):
    possible_seed_indices = [k for k, v in seed_point_bool_dict.items() if v]
    for pidx in possible_seed_indices:
        if pidx not in orphans:
            if TrySeed(pidx, PointsNP[pidx], radius):
                return pidx
    # no good seed, terminate algorithm
    return False


# if __name__ == "__main__":