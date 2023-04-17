import open3d as o3d
import numpy as np


def alpha_shape(point_cloud, alpha):
    t = o3d.geometry.TetraMesh.create_from_point_cloud(point_cloud)
    tetra_mesh: o3d.geometry.TetraMesh = t[0]
    point_map = t[1]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = tetra_mesh.vertices
    if point_cloud.has_normals():
        mesh.vertex_normals = point_cloud.normals
        for i in range(len(point_map)):
            mesh.vertex_normals[i] = point_cloud.normals[point_map[i]]
    if point_cloud.has_colors():
        mesh.vertex_colors = point_cloud.colors
        for i in range(len(point_map)):
            mesh.vertex_colors[i] = point_cloud.colors[point_map[i]]

    vsqn = np.zeros(len(tetra_mesh.vertices))
    for i in range(len(tetra_mesh.vertices)):
        vsqn[i] = tetra_mesh.vertices[i].dot(tetra_mesh.vertices[i])

    v = tetra_mesh.vertices
    for i, tetra in enumerate(tetra_mesh.tetras):
        print(f"\r {i:10}/{len(tetra_mesh.tetras)}"
              f" |{'#' * int((i / len(tetra_mesh.tetras)) * 50):<50}|"
              f" {i/len(tetra_mesh.tetras):3.0%}", end="")

        points = np.array([v[tetra[0]], v[tetra[1]], v[tetra[2]], v[tetra[3]]])
        A = np.bmat([[2 * np.dot(points, points.T), np.ones((points.shape[0], 1))],
                     [np.ones((1, points.shape[0])), np.zeros((1, 1))]])
        b = np.hstack((np.sum(points * points, axis=1), np.ones(1)))
        center = np.linalg.solve(A, b)[:-1]
        r = np.linalg.norm(points[0, :] - np.dot(center, points))

        if r < alpha:
            mesh.triangles.append([tetra[0], tetra[1], tetra[2]])
            mesh.triangles.append([tetra[0], tetra[1], tetra[3]])
            mesh.triangles.append([tetra[0], tetra[2], tetra[3]])
            mesh.triangles.append([tetra[1], tetra[2], tetra[3]])

    triangle_count = {}
    for t in mesh.triangles:
        triangle = t.tobytes()
        if triangle not in triangle_count:
            triangle_count[triangle] = 1
        else:
            triangle_count[triangle] += 1
    idx = 0
    for triangle in mesh.triangles:
        if triangle_count[triangle.tobytes()] == 1:
            mesh.triangles[idx] = triangle
            idx += 1
    mesh.triangles = mesh.triangles[:idx]
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()

    return mesh


if __name__ == "__main__":
    # npy = np.load("famous_original/04_pts/dragon.xyz.npy")
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(npy)
    dragon = o3d.io.read_triangle_mesh("famous_original/03_meshes/dragon.ply")
    dragon.compute_vertex_normals()
    point_cloud = dragon.sample_points_uniformly(number_of_points=10000)
    a = 0.1
    a_shape = alpha_shape(point_cloud, a)
    a_shape.compute_vertex_normals()
    a_shape.paint_uniform_color([1, 0.706, 0])
    a_shape2 = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, a)
    a_shape2.compute_vertex_normals()
    a_shape2.translate([0, 0, 1])
    print(len(a_shape.triangles), len(a_shape2.triangles))

    o3d.visualization.draw_geometries([a_shape, a_shape2], mesh_show_back_face=True)