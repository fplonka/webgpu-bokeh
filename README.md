# Bokeh Magic

This is the code for [bokeh-magic](https://bokeh.fplonka.dev/), a web application that creates realistic bokeh blur effects on images using depth estimation and WebGPU.

TODO: demo gif

You upload an image, click to set the focus point, then adjust the sliders to control the blur effect. Bokeh Magic runs entirely in the browser, using WebGPU to compute the blur effect quickly.

Requires browser WebGPU support # TODO: 

## Implementation

When you upload an image, a [depth map](https://en.wikipedia.org/wiki/Depth_map) is created using the Depth Anything v2 model. This model runs in the browser thanks to [transformers.js](https://github.com/huggingface/transformers.js). Then when you click the image, we check the depth of the clicked pixel and compute for every pixel in the image the [circle of confusion](https://en.wikipedia.org/wiki/Circle_of_confusion), i.e. a measure of how in focus the pixel is. This is then used to blur the image. 

For the blurring computation, I implemented the techniques in [this paper](http://ivizlab.sfu.ca/papers/cgf2012.pdf) to do support different bokeh shapes (hexagon, octagon, square) efficiently. 

## Development

The app is just static html/js + .wgsl shaders, deployed on Cloudflare Pages. For generating tailwind output run:

```sh
tailwindcss -i ./static/input.css -o ./static/output.css --watch
```