// Vertex shader for rendering a fullscreen quad
struct VertexOutput {
  @builtin(position) Position : vec4f,
  @location(0) fragUV : vec2f,
}

@vertex
fn vertex_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
  // Fullscreen quad vertices
  const pos = array(
    vec2( 1.0,  1.0),  // Bottom right
    vec2( 1.0, -1.0),  // Top right
    vec2(-1.0, -1.0),  // Top left
    vec2( 1.0,  1.0),  // Bottom right
    vec2(-1.0, -1.0),  // Top left
    vec2(-1.0,  1.0),  // Bottom left
  );

  const uv = array(
    vec2(1.0, 1.0),  // Bottom right
    vec2(1.0, 0.0),  // Top right
    vec2(0.0, 0.0),  // Top left
    vec2(1.0, 1.0),  // Bottom right
    vec2(0.0, 0.0),  // Top left
    vec2(0.0, 1.0),  // Bottom left
  );

  var output : VertexOutput;
  output.Position = vec4(pos[VertexIndex], 0.0, 1.0);
  output.fragUV = vec2(uv[VertexIndex].x, 1.0 - uv[VertexIndex].y);
  return output;
}

// Fragment shader for displaying the final image
@group(0) @binding(0) var linearTexture: texture_2d<f32>;
@group(0) @binding(1) var textureSampler: sampler;

@fragment
fn fragment_main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  return textureSample(linearTexture, textureSampler, fragUV);
}
