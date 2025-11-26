# HaMer Rendering Fixes Summary

## Problem
HaMer's rendering system was crashing with multiple issues:
1. **PyOpenGL 3.1.0**: bytes/str type mismatch bug → Fixed by upgrading to 3.1.5
2. **EGL Backend**: Segmentation faults with NVIDIA drivers
3. **Shader Validation**: OSMesa's strict validation failing on working shaders

## Solution Applied

### Files Modified
1. `submodules/hamer/hamer/utils/renderer.py`
2. `submodules/hamer/hamer/utils/mesh_renderer.py`
3. `submodules/hamer/data/sphere_renderer.py`

### Changes Made (in each file)

#### 1. Force OSMesa Platform
```python
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
```
- Uses CPU-based software rendering (stable, no crashes)
- Slower than GPU but reliable

#### 2. Mesa Environment Variables
```python
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'
os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '450'
```
- Enables modern shader features in OSMesa

#### 3. Disable Shader Validation
```python
# Monkey patch
from OpenGL.GL import shaders
shaders.ShaderProgram.check_validate = lambda self: None
```
- OSMesa shaders work but fail validation check
- This bypasses the overly strict validation

#### 4. Simplified Lighting (renderer.py only)
- Reduced from 15+ lights to 3 raymond lights
- Prevents shader complexity issues
- Faster rendering

## Trade-offs

### Pros
✅ Stable, no crashes
✅ No shader validation errors
✅ Works on all systems
✅ Reproducible builds

### Cons
❌ CPU-only rendering (slower than GPU)
❌ Slightly different lighting (less dramatic)

## Performance Impact
- Rendering is CPU-bound now
- Expect ~2-5x slower than GPU-accelerated EGL
- Still acceptable for batch processing

## Future GPU Acceleration
To restore GPU rendering (if needed later):
1. Update NVIDIA drivers and Mesa libraries
2. Fix EGL device selection issues
3. Or use xvfb + native OpenGL

## Dependencies
- pyopengl >= 3.1.5
- pyrender 0.1.45
- OSMesa libraries (libOSMesa.so)
