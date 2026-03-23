# blackroad-3d-renderer

> 3D wireframe renderer with matrix math and ASCII terminal output

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Interactive](https://github.com/BlackRoad-Interactive)

---

# blackroad-3d-renderer

> Software 3D wireframe renderer with full matrix math, perspective projection, and ASCII terminal output.

Part of **BlackRoad-Interactive** — production game and graphics infrastructure.

## Architecture

```
Scene
├── Camera(pos, target, fov) — view + projection matrices
├── Mesh(vertices, faces)    — geometry + per-mesh transform
│   ├── Mesh.cube()
│   ├── Mesh.pyramid()
│   └── Mesh.icosahedron()
└── Matrix4x4
    ├── translate / scale / rotate_x/y/z
    ├── perspective(fov, aspect, near, far)
    └── look_at(eye, center, up)
```

## Pipeline

```
World vertices
  → Model matrix (per mesh)
  → View matrix  (camera look_at)
  → Projection   (perspective)
  → NDC [-1,1]
  → Screen space
  → Bresenham line raster
  → Depth-shaded ASCII buffer
```

## Quick Start

```python
import math
from src.renderer import Scene, Mesh, Camera, Vec3, Matrix4x4

scene = Scene(width=80, height=30)
scene.add_mesh(Mesh.cube("my_cube", size=2.0))
scene.camera = Camera(pos=Vec3(0, 2, -6), target=Vec3(0, 0, 0))

# Rotate the cube
scene.meshes[0].transform = Matrix4x4.rotate_y(math.radians(30))
scene.print_frame()
```

## OBJ Loading

```python
from src.renderer import load_obj
mesh = load_obj("model.obj", name="spaceship")
scene.add_mesh(mesh)
```

## Run Demo

```bash
python src/renderer.py
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## CI

GitHub Actions · Python 3.11 + 3.12 · pytest + flake8 + coverage
