struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var packedTexture: texture_2d<u32>;
@group(0) @binding(1) var linearTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;

// Convert sRGB to linear RGB
fn srgbToLinear(x: f32) -> f32 {
    if x <= 0.04045 {
        return x / 12.92;
    } else {
        return pow((x + 0.055) / 1.055, 2.4);
    }
}

// Unpack u32 RGBA color to linear vec4<f32>
fn unpackToLinear(color: u32) -> vec4<f32> {
    let r = f32((color >> 0u) & 0xFFu) / 255.0;
    let g = f32((color >> 8u) & 0xFFu) / 255.0;
    let b = f32((color >> 16u) & 0xFFu) / 255.0;
    let a = f32((color >> 24u) & 0xFFu) / 255.0;
    
    return vec4<f32>(
        srgbToLinear(r),
        srgbToLinear(g),
        srgbToLinear(b),
        a
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Bounds check
    if x >= params.width || y >= params.height {
        return;
    }
    
    // Read packed u32 from texture
    let packedColor = textureLoad(packedTexture, vec2<u32>(x, y), 0).r;
    
    // Convert packed u32 to linear vec4<f32> and write to output texture
    textureStore(linearTexture, vec2<u32>(x, y), unpackToLinear(packedColor));
}
