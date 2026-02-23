"""
BlackRoad 3D Renderer — Software wireframe renderer with ASCII terminal output.
Matrix math, perspective projection, OBJ parser, scene management.
"""
from __future__ import annotations
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple


# ─── ANSI helpers ────────────────────────────────────────────────────────────
RESET = "\033[0m"
_DEPTH_CHARS = " .:-=+*#%@"


# ─── Vec3 / Vec4 ─────────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    @property
    def length(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> "Vec3":
        l = self.length
        if l < 1e-9:
            return Vec3(0, 0, 0)
        return Vec3(self.x / l, self.y / l, self.z / l)

    def to_vec4(self, w: float = 1.0) -> "Vec4":
        return Vec4(self.x, self.y, self.z, w)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class Vec4:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_vec3(self) -> Vec3:
        if abs(self.w) < 1e-9:
            return Vec3(self.x, self.y, self.z)
        return Vec3(self.x / self.w, self.y / self.w, self.z / self.w)

    def __repr__(self) -> str:
        return f"Vec4({self.x:.3f}, {self.y:.3f}, {self.z:.3f}, {self.w:.3f})"


# ─── Matrix4x4 ───────────────────────────────────────────────────────────────

class Matrix4x4:
    """Row-major 4×4 matrix used for all 3D transforms."""

    __slots__ = ("_m",)

    def __init__(self, data: Optional[List[List[float]]] = None) -> None:
        if data is not None:
            self._m: List[List[float]] = [list(row) for row in data]
        else:
            self._m = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    @classmethod
    def identity(cls) -> "Matrix4x4":
        return cls()

    @classmethod
    def zeros(cls) -> "Matrix4x4":
        return cls([[0] * 4 for _ in range(4)])

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        r, c = idx
        return self._m[r][c]

    def __setitem__(self, idx: Tuple[int, int], val: float) -> None:
        r, c = idx
        self._m[r][c] = val

    def multiply(self, other: "Matrix4x4") -> "Matrix4x4":
        result = Matrix4x4.zeros()
        for i in range(4):
            for j in range(4):
                result[i, j] = sum(self._m[i][k] * other._m[k][j] for k in range(4))
        return result

    def __matmul__(self, other: "Matrix4x4") -> "Matrix4x4":
        return self.multiply(other)

    def transform_vec4(self, v: Vec4) -> Vec4:
        vals = [v.x, v.y, v.z, v.w]
        out = [sum(self._m[r][c] * vals[c] for c in range(4)) for r in range(4)]
        return Vec4(*out)

    def transform_point(self, v: Vec3) -> Vec3:
        return self.transform_vec4(v.to_vec4(1.0)).to_vec3()

    def transform_direction(self, v: Vec3) -> Vec3:
        return self.transform_vec4(v.to_vec4(0.0)).to_vec3()

    def transpose(self) -> "Matrix4x4":
        return Matrix4x4([[self._m[j][i] for j in range(4)] for i in range(4)])

    # ── Factory methods ────────────────────────────────────────────────────
    @classmethod
    def translate(cls, tx: float, ty: float, tz: float) -> "Matrix4x4":
        m = cls()
        m[0, 3], m[1, 3], m[2, 3] = tx, ty, tz
        return m

    @classmethod
    def scale(cls, sx: float, sy: float, sz: float) -> "Matrix4x4":
        m = cls()
        m[0, 0], m[1, 1], m[2, 2] = sx, sy, sz
        return m

    @classmethod
    def rotate_x(cls, angle: float) -> "Matrix4x4":
        c, s = math.cos(angle), math.sin(angle)
        m = cls()
        m[1, 1], m[1, 2] = c, -s
        m[2, 1], m[2, 2] = s, c
        return m

    @classmethod
    def rotate_y(cls, angle: float) -> "Matrix4x4":
        c, s = math.cos(angle), math.sin(angle)
        m = cls()
        m[0, 0], m[0, 2] = c, s
        m[2, 0], m[2, 2] = -s, c
        return m

    @classmethod
    def rotate_z(cls, angle: float) -> "Matrix4x4":
        c, s = math.cos(angle), math.sin(angle)
        m = cls()
        m[0, 0], m[0, 1] = c, -s
        m[1, 0], m[1, 1] = s, c
        return m

    @classmethod
    def perspective(cls, fov_y: float, aspect: float, near: float, far: float) -> "Matrix4x4":
        f = 1.0 / math.tan(fov_y / 2)
        m = cls.zeros()
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    @classmethod
    def look_at(cls, eye: Vec3, center: Vec3, up: Vec3) -> "Matrix4x4":
        f = (center - eye).normalize()
        r = f.cross(up).normalize()
        u = r.cross(f)
        m = cls()
        m[0, 0], m[0, 1], m[0, 2], m[0, 3] = r.x, r.y, r.z, -r.dot(eye)
        m[1, 0], m[1, 1], m[1, 2], m[1, 3] = u.x, u.y, u.z, -u.dot(eye)
        m[2, 0], m[2, 1], m[2, 2], m[2, 3] = -f.x, -f.y, -f.z, f.dot(eye)
        m[3, 0], m[3, 1], m[3, 2], m[3, 3] = 0, 0, 0, 1
        return m

    def __repr__(self) -> str:
        rows = ["  [" + "  ".join(f"{v:7.3f}" for v in row) + "]" for row in self._m]
        return "Matrix4x4[\n" + "\n".join(rows) + "\n]"


# ─── Camera ──────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    pos: Vec3 = field(default_factory=lambda: Vec3(0, 0, -5))
    target: Vec3 = field(default_factory=Vec3)
    up: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    fov: float = math.radians(60)
    near: float = 0.1
    far: float = 100.0
    aspect: float = 2.0  # width/height (chars are ~2:1)

    def view_matrix(self) -> Matrix4x4:
        return Matrix4x4.look_at(self.pos, self.target, self.up)

    def projection_matrix(self) -> Matrix4x4:
        return Matrix4x4.perspective(self.fov, self.aspect, self.near, self.far)

    def vp_matrix(self) -> Matrix4x4:
        return self.projection_matrix() @ self.view_matrix()


# ─── Mesh ─────────────────────────────────────────────────────────────────────

@dataclass
class Mesh:
    name: str = "mesh"
    vertices: List[Vec3] = field(default_factory=list)
    faces: List[Tuple[int, ...]] = field(default_factory=list)
    transform: Matrix4x4 = field(default_factory=Matrix4x4)

    def get_edges(self) -> Iterator[Tuple[int, int]]:
        seen: set = set()
        for face in self.faces:
            for i in range(len(face)):
                a, b = face[i], face[(i + 1) % len(face)]
                edge = (min(a, b), max(a, b))
                if edge not in seen:
                    seen.add(edge)
                    yield edge

    @classmethod
    def cube(cls, name: str = "cube", size: float = 1.0) -> "Mesh":
        s = size / 2
        verts = [
            Vec3(-s, -s, -s), Vec3(s, -s, -s), Vec3(s, s, -s), Vec3(-s, s, -s),
            Vec3(-s, -s,  s), Vec3(s, -s,  s), Vec3(s, s,  s), Vec3(-s, s,  s),
        ]
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                 (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)]
        return cls(name=name, vertices=verts, faces=faces)

    @classmethod
    def pyramid(cls, name: str = "pyramid", base: float = 1.0, height: float = 1.5) -> "Mesh":
        b = base / 2
        verts = [Vec3(-b, 0, -b), Vec3(b, 0, -b), Vec3(b, 0, b), Vec3(-b, 0, b), Vec3(0, height, 0)]
        faces = [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
        return cls(name=name, vertices=verts, faces=faces)

    @classmethod
    def icosahedron(cls, name: str = "icosahedron", radius: float = 1.0) -> "Mesh":
        phi = (1 + math.sqrt(5)) / 2
        verts = []
        for a, b in [(-1, phi), (1, phi), (-1, -phi), (1, -phi),
                     (0, -1), (0, 1), (-phi, 0), (phi, 0), (0, -phi), (0, phi),
                     (-phi, 1), (phi, 1)]:
            v = Vec3(a, b, 0).normalize()
            verts.append(Vec3(v.x * radius, v.y * radius, v.z * radius))
        # simplified faces (subset for demo)
        faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11)]
        return cls(name=name, vertices=verts, faces=faces)


# ─── OBJ Parser ──────────────────────────────────────────────────────────────

def load_obj(path: str, name: Optional[str] = None) -> Mesh:
    """Parse a Wavefront OBJ file (vertices + faces only)."""
    vertices: List[Vec3] = []
    faces: List[Tuple[int, ...]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append(Vec3(float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                # support: "f 1 2 3" and "f 1/1/1 2/2/2 3/3/3"
                indices = [int(re.split(r"[/\\]", p)[0]) - 1 for p in line.split()[1:]]
                faces.append(tuple(indices))
    return Mesh(name=name or os.path.basename(path), vertices=vertices, faces=faces)


# ─── Projection helper ───────────────────────────────────────────────────────

def project_vertex(v: Vec3, mvp: Matrix4x4, screen_w: int, screen_h: int) -> Optional[Tuple[int, int, float]]:
    """
    Project a world-space vertex through MVP to screen coordinates.

    Returns (screen_x, screen_y, depth) or None if behind the camera.
    """
    clip = mvp.transform_vec4(v.to_vec4())
    if abs(clip.w) < 1e-9:
        return None
    # Perspective divide → NDC [-1, 1]
    ndc_x = clip.x / clip.w
    ndc_y = clip.y / clip.w
    depth = clip.z / clip.w
    if depth < -1 or depth > 1:
        return None
    sx = int((ndc_x + 1) * 0.5 * screen_w)
    sy = int((1 - (ndc_y + 1) * 0.5) * screen_h)
    return sx, sy, depth


# ─── Bresenham line ───────────────────────────────────────────────────────────

def bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterator[Tuple[int, int]]:
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


# ─── Scene ───────────────────────────────────────────────────────────────────

class Scene:
    """A 3D scene containing meshes and a camera."""

    def __init__(self, width: int = 80, height: int = 24) -> None:
        self.width = width
        self.height = height
        self._meshes: Dict[str, Mesh] = {}
        self.camera = Camera(aspect=width / height)
        self._depth_buf: List[List[float]] = []
        self._char_buf: List[List[str]] = []

    def add_mesh(self, mesh: Mesh) -> None:
        self._meshes[mesh.name] = mesh

    def remove_mesh(self, name: str) -> bool:
        return self._meshes.pop(name, None) is not None

    @property
    def meshes(self) -> List[Mesh]:
        return list(self._meshes.values())

    def _clear_buffers(self) -> None:
        self._depth_buf = [[float("inf")] * self.width for _ in range(self.height)]
        self._char_buf = [[" "] * self.width for _ in range(self.height)]

    def _put_pixel(self, x: int, y: int, depth: float, ch: str) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            if depth < self._depth_buf[y][x]:
                self._depth_buf[y][x] = depth
                self._char_buf[y][x] = ch
                return True
        return False

    def render_frame(self, char: str = "*") -> List[str]:
        """Render all meshes and return frame as list of row strings."""
        self._clear_buffers()
        vp = self.camera.vp_matrix()

        for mesh in self._meshes.values():
            mvp = vp @ mesh.transform
            # Project vertices
            projected: Dict[int, Optional[Tuple[int, int, float]]] = {}
            for i, vert in enumerate(mesh.vertices):
                projected[i] = project_vertex(vert, mvp, self.width, self.height)

            # Draw edges
            for a_idx, b_idx in mesh.get_edges():
                pa = projected.get(a_idx)
                pb = projected.get(b_idx)
                if pa is None or pb is None:
                    continue
                ax, ay, ad = pa
                bx, by, bd = pb
                steps = max(abs(bx - ax), abs(by - ay)) + 1
                for px, py in bresenham(ax, ay, bx, by):
                    # Linear interpolate depth along edge
                    t = 0 if steps <= 1 else ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5 / steps
                    d = ad + (bd - ad) * t
                    # Map depth to a shading char
                    shade_idx = max(1, int((1 - min(1, max(0, (d + 1) / 2))) * (len(_DEPTH_CHARS) - 1)))
                    self._put_pixel(px, py, d, _DEPTH_CHARS[shade_idx])

        return ["".join(row) for row in self._char_buf]

    def print_frame(self) -> None:
        border = "+" + "-" * self.width + "+"
        print(border)
        for row in self.render_frame():
            print("|" + row + "|")
        print(border)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def run_demo(frames: int = 36) -> None:
    scene = Scene(width=80, height=30)
    cube = Mesh.cube("cube", size=2.0)
    pyramid = Mesh.pyramid("pyramid", base=1.5, height=2.0)
    # offset pyramid
    pyramid.transform = Matrix4x4.translate(3.5, 0, 0)
    scene.add_mesh(cube)
    scene.add_mesh(pyramid)
    scene.camera = Camera(pos=Vec3(0, 2, -6), target=Vec3(0, 0, 0), fov=math.radians(55), aspect=80 / 30)

    os.system("clear" if os.name == "posix" else "cls")
    sys.stdout.write("\033[?25l")
    try:
        for i in range(frames):
            angle = i * math.pi / 18
            cube.transform = Matrix4x4.rotate_y(angle) @ Matrix4x4.rotate_x(angle * 0.5)
            pyramid.transform = Matrix4x4.translate(3.5, 0, 0) @ Matrix4x4.rotate_y(-angle)
            sys.stdout.write("\033[H")
            scene.print_frame()
            import time; time.sleep(0.05)
    finally:
        sys.stdout.write("\033[?25h")
    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
