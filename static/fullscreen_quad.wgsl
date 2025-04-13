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

// Fragment shader for converting linear RGB to sRGB and displaying
@group(0) @binding(0) var linearTexture: texture_2d<f32>;

// Convert linear RGB to sRGB
fn linearToSrgb(x: f32) -> f32 {
  if x <= 0.0031308 {
    return 12.92 * x;
  } else {
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
  }
}

@fragment
fn fragment_main(@location(0) fragUV : vec2f) -> @location(0) vec4f {
  // Get texture dimensions directly from the texture
  let dims = textureDimensions(linearTexture);
  
  let x = i32(fragUV.x * f32(dims.x));
  let y = i32(fragUV.y * f32(dims.y));
  
  // Bounds check
  if (x >= i32(dims.x) || y >= i32(dims.y)) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  
  // Read linear color from texture
  let linearColor = textureLoad(linearTexture, vec2<i32>(x, y), 0);
  
  // Convert linear to sRGB
  return vec4f(
    linearColor.r,
    linearColor.g,
    linearColor.b,
    linearColor.a
  );
}
