struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<storage, read> inputLinear: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputPacked: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

// Convert linear RGB to sRGB
fn linearToSrgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    } else {
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
}

// Pack linear vec4<f32> to u32 RGBA color in sRGB space
fn packFromLinear(color: vec4<f32>) -> u32 {
    let srgb = vec4<f32>(
        linearToSrgb(color.r),
        linearToSrgb(color.g),
        linearToSrgb(color.b),
        color.a
    );
    
    let r = u32(clamp(srgb.r, 0.0, 1.0) * 255.0);
    let g = u32(clamp(srgb.g, 0.0, 1.0) * 255.0);
    let b = u32(clamp(srgb.b, 0.0, 1.0) * 255.0);
    let a = u32(clamp(srgb.a, 0.0, 1.0) * 255.0);
    
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Bounds check
    if x >= params.width || y >= params.height {
        return;
    }
    
    let idx = y * params.width + x;
    
    // Convert linear vec4<f32> to packed u32
    outputPacked[idx] = packFromLinear(inputLinear[idx]);
}
