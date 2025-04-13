struct Params {
    width: u32,
    height: u32,
}

// Use rgba8unorm-srgb format for input texture when creating the GPU texture
// This will automatically handle sRGB to linear conversion
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Get dimensions directly from the texture
    let dims = textureDimensions(inputTexture);
    
    // Bounds check
    if x >= dims.x || y >= dims.y {
        return;
    }
    
    // Read color directly from texture - if using rgba8unorm-srgb format,
    // the GPU will automatically convert from sRGB to linear
    let color = textureLoad(inputTexture, vec2<u32>(x, y), 0);
    
    // Store the color (already in linear space) to output texture
    textureStore(outputTexture, vec2<u32>(x, y), color);
}
