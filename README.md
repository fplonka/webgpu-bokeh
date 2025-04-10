# good-bg-blur

web app for background blur using depth estimation and webgpu

## requirements

- go 1.21+
- tailwind cli
- browser with webgpu support

## setup

```sh
# start tailwind watcher
tailwindcss -i ./static/input.css -o ./static/output.css --watch

# run server
go run .
```

server runs on :8080 by default

## implementation

### backend (go)

- uses hugging face depth-anything-v2 api
- two-phase depth estimation:
  1. upload image, get event_id
  2. poll for results (30s timeout)
- processes depth map:
  - downloads grayscale png
  - normalizes to float32 (0-1)
  - sends to frontend as json
- handles multiple image formats (jpg/png/webp/tiff)
- 10mb file size limit

### frontend (js/webgpu)

- two-pass blur pipeline:
  1. compute circle of confusion (coc.wgsl)
     - uses depth map to determine blur radius
     - controlled by focus depth and dof params
  2. apply bokeh blur (blur.wgsl)
     - variable-radius blur based on coc
     - rgba8 texture format
     - uses compute shaders for performance

### data flow

1. user uploads image
2. backend gets depth map from hf
3. frontend creates webgpu buffers:
   - original image (rgba8)
   - depth map (float32)
   - coc values (float32)
   - output image (rgba8)
4. compute shaders process buffers
5. result copied to canvas
