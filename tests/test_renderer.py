"""Tests for BlackRoad 3D Renderer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import tempfile
import pytest
from renderer import (
    Vec3, Vec4, Matrix4x4, Camera, Mesh, Scene,
    project_vertex, bresenham, load_obj,
)
EPSILON = 1e-9


# ── Vec3 tests ────────────────────────────────────────────────────────────────

class TestVec3:
    def test_add(self):
        v = Vec3(1, 2, 3) + Vec3(4, 5, 6)
        assert v.x == 5 and v.y == 7 and v.z == 9

    def test_sub(self):
        v = Vec3(5, 5, 5) - Vec3(3, 2, 1)
        assert v.x == 2 and v.y == 3 and v.z == 4

    def test_mul(self):
        v = Vec3(1, 2, 3) * 3
        assert v.x == 3 and v.y == 6 and v.z == 9

    def test_neg(self):
        v = -Vec3(1, -2, 3)
        assert v.x == -1 and v.y == 2 and v.z == -3

    def test_dot(self):
        d = Vec3(1, 0, 0).dot(Vec3(0, 1, 0))
        assert math.isclose(d, 0.0)

    def test_cross(self):
        c = Vec3(1, 0, 0).cross(Vec3(0, 1, 0))
        assert math.isclose(c.z, 1.0)

    def test_length(self):
        assert math.isclose(Vec3(0, 3, 4).length, 5.0)

    def test_normalize_unit(self):
        n = Vec3(1, 2, 3).normalize()
        assert math.isclose(n.length, 1.0)

    def test_normalize_zero(self):
        n = Vec3(0, 0, 0).normalize()
        assert n.length < EPSILON

    def test_to_vec4(self):
        v4 = Vec3(1, 2, 3).to_vec4()
        assert v4.w == 1.0

    def test_repr(self):
        assert "Vec3" in repr(Vec3(1, 2, 3))


# ── Vec4 tests ────────────────────────────────────────────────────────────────

class TestVec4:
    def test_to_vec3_divides_w(self):
        v4 = Vec4(2, 4, 6, 2)
        v3 = v4.to_vec3()
        assert math.isclose(v3.x, 1) and math.isclose(v3.y, 2) and math.isclose(v3.z, 3)

    def test_to_vec3_zero_w(self):
        v4 = Vec4(1, 2, 3, 0)
        v3 = v4.to_vec3()
        assert v3.x == 1.0


# ── Matrix4x4 tests ───────────────────────────────────────────────────────────

class TestMatrix4x4:
    def test_identity(self):
        m = Matrix4x4.identity()
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert math.isclose(m[i, j], expected)

    def test_multiply_identity(self):
        m = Matrix4x4.identity()
        result = m @ m
        for i in range(4):
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert math.isclose(result[i, j], expected)

    def test_translate(self):
        m = Matrix4x4.translate(5, 3, 1)
        v = m.transform_point(Vec3(0, 0, 0))
        assert math.isclose(v.x, 5) and math.isclose(v.y, 3) and math.isclose(v.z, 1)

    def test_scale(self):
        m = Matrix4x4.scale(2, 3, 4)
        v = m.transform_point(Vec3(1, 1, 1))
        assert math.isclose(v.x, 2) and math.isclose(v.y, 3) and math.isclose(v.z, 4)

    def test_rotate_x_90(self):
        m = Matrix4x4.rotate_x(math.pi / 2)
        v = m.transform_point(Vec3(0, 1, 0))
        assert math.isclose(v.y, 0.0, abs_tol=1e-9)
        assert math.isclose(v.z, 1.0, abs_tol=1e-9)

    def test_rotate_y_90(self):
        m = Matrix4x4.rotate_y(math.pi / 2)
        v = m.transform_point(Vec3(1, 0, 0))
        assert math.isclose(v.x, 0.0, abs_tol=1e-9)
        assert math.isclose(v.z, -1.0, abs_tol=1e-9)

    def test_perspective_not_identity(self):
        m = Matrix4x4.perspective(math.radians(60), 1.0, 0.1, 100.0)
        assert not math.isclose(m[0, 0], 1.0)

    def test_look_at(self):
        m = Matrix4x4.look_at(Vec3(0, 0, -5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        # Origin translated by the eye position — should be non-zero
        v = m.transform_point(Vec3(0, 0, 0))
        assert abs(v.z) > 0

    def test_transpose(self):
        m = Matrix4x4.identity()
        m[0, 1] = 7
        t = m.transpose()
        assert math.isclose(t[1, 0], 7.0)

    def test_compose_translate_scale(self):
        t = Matrix4x4.translate(10, 0, 0)
        s = Matrix4x4.scale(2, 2, 2)
        m = t @ s
        v = m.transform_point(Vec3(1, 0, 0))
        assert math.isclose(v.x, 12.0)

    def test_repr(self):
        assert "Matrix4x4" in repr(Matrix4x4())


# ── Camera tests ──────────────────────────────────────────────────────────────

class TestCamera:
    def test_vp_matrix_is_matrix(self):
        cam = Camera()
        vp = cam.vp_matrix()
        assert isinstance(vp, Matrix4x4)

    def test_view_matrix_moves_origin(self):
        cam = Camera(pos=Vec3(0, 0, -5), target=Vec3(0, 0, 0))
        v = cam.view_matrix().transform_point(Vec3(0, 0, 0))
        assert abs(v.z) > 0  # origin is displaced from camera position


# ── Mesh tests ────────────────────────────────────────────────────────────────

class TestMesh:
    def test_cube_vertex_count(self):
        m = Mesh.cube()
        assert len(m.vertices) == 8

    def test_cube_face_count(self):
        m = Mesh.cube()
        assert len(m.faces) == 6

    def test_pyramid_face_count(self):
        m = Mesh.pyramid()
        assert len(m.faces) == 5

    def test_get_edges_unique(self):
        m = Mesh.cube()
        edges = list(m.get_edges())
        assert len(edges) == len(set(edges))  # all unique

    def test_cube_edges_count(self):
        m = Mesh.cube()
        assert len(list(m.get_edges())) == 12


# ── OBJ parser tests ──────────────────────────────────────────────────────────

class TestOBJLoader:
    def test_load_simple_obj(self):
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as f:
            f.write(content)
            path = f.name
        m = load_obj(path)
        assert len(m.vertices) == 3
        assert len(m.faces) == 1
        os.unlink(path)

    def test_load_obj_slash_format(self):
        content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as f:
            f.write(content)
            path = f.name
        m = load_obj(path)
        assert len(m.faces) == 1
        os.unlink(path)


# ── Projection tests ──────────────────────────────────────────────────────────

class TestProjectVertex:
    def test_projects_front_face(self):
        cam = Camera(pos=Vec3(0, 0, -5), target=Vec3(0, 0, 0))
        mvp = cam.vp_matrix()
        result = project_vertex(Vec3(0, 0, 0), mvp, 80, 40)
        assert result is not None
        sx, sy, depth = result
        assert 0 <= sx < 80 and 0 <= sy < 40

    def test_behind_camera_returns_none(self):
        cam = Camera(pos=Vec3(0, 0, -5), target=Vec3(0, 0, 0))
        mvp = cam.vp_matrix()
        # Far behind camera
        result = project_vertex(Vec3(0, 0, -1000), mvp, 80, 40)
        assert result is None


# ── Bresenham tests ───────────────────────────────────────────────────────────

class TestBresenham:
    def test_straight_line(self):
        pts = list(bresenham(0, 0, 4, 0))
        assert (0, 0) in pts and (4, 0) in pts
        assert len(pts) == 5

    def test_diagonal(self):
        pts = list(bresenham(0, 0, 2, 2))
        assert (0, 0) in pts and (2, 2) in pts

    def test_single_point(self):
        pts = list(bresenham(3, 3, 3, 3))
        assert pts == [(3, 3)]


# ── Scene tests ───────────────────────────────────────────────────────────────

class TestScene:
    def test_add_remove_mesh(self):
        s = Scene(width=40, height=20)
        m = Mesh.cube("test_cube")
        s.add_mesh(m)
        assert len(s.meshes) == 1
        s.remove_mesh("test_cube")
        assert len(s.meshes) == 0

    def test_render_frame_returns_rows(self):
        s = Scene(width=40, height=20)
        s.add_mesh(Mesh.cube("cube"))
        s.camera = Camera(pos=Vec3(0, 0, -5), target=Vec3(0, 0, 0), aspect=40/20)
        frame = s.render_frame()
        assert len(frame) == 20
        assert all(len(row) == 40 for row in frame)

    def test_render_marks_pixels(self):
        s = Scene(width=80, height=40)
        s.add_mesh(Mesh.cube("c"))
        s.camera = Camera(pos=Vec3(0, 2, -5), target=Vec3(0, 0, 0), aspect=2.0)
        frame = s.render_frame()
        non_empty = sum(ch != " " for row in frame for ch in row)
        assert non_empty > 0  # something should be drawn
