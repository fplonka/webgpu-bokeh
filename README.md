# webgpu-bokeh

This is the code for [webgpu-bokeh](https://bokeh.fplonka.dev/), a web application that creates realistic bokeh blur effects on images using depth estimation and WebGPU.

![demo](https://github.com/user-attachments/assets/2d867bbd-9322-42c1-b5cd-29466ad968b3)

You upload an image, click to set the focus point, then adjust the sliders to control the blur effect. Everything runs locally in the browser, using WebGPU to compute the blur effect quickly.

Requires browser WebGPU support. In newer Chrome and Edge versions everything should just work, for Firefox you need Firefox Nightly, for Safari you need Safari Technology Preview. See [here](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility) for detailed implementation status.

## Implementation

When you upload an image, a [depth map](https://en.wikipedia.org/wiki/Depth_map) is created using the [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) model. This model runs in the browser thanks to [transformers.js](https://github.com/huggingface/transformers.js). Then when you click the image, we check the depth of the clicked pixel and compute for every pixel in the image the [circle of confusion](https://en.wikipedia.org/wiki/Circle_of_confusion), i.e. a measure of how out of focus the pixel is. This is then used to blur the image. 

For the blurring computation, I stole the techniques from [this paper](http://ivizlab.sfu.ca/papers/cgf2012.pdf) to do support different bokeh shapes (hexagon, octagon, square) efficiently. 

## Development

The app is just static html/js + .wgsl shaders, deployed on Cloudflare Pages. For generating tailwind output run:

```sh
tailwindcss -i ./static/input.css -o ./static/output.css --watch
```