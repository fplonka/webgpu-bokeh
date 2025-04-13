// Depth texture using r32float format for high precision depth values
@group(0) @binding(0) var depthTexture: texture_2d<f32>;
// R32float is one of the few formats that supports read_write operations
@group(0) @binding(1) var cocTexture: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    focus_depth: f32,
    max_coc: f32,
    depth_of_field: f32,
    background_only: u32,  // 0 = blur all, 1 = blur background only
}

fn calculate_coc_radius(pixel_depth: f32, focal_depth: f32) -> f32 {
    if (params.background_only != 0u && pixel_depth >= focal_depth) {
        return 0;
    }
    // trader math. trust.
    var dist = focal_depth - pixel_depth;
    let a = 2 / (2 - params.depth_of_field);
    let b = 1 - a;
    return params.max_coc * max(abs(a*(dist + params.depth_of_field / 2) + b) + b, 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    // Get dimensions directly from the texture
    let dims = textureDimensions(depthTexture);
    
    if x >= dims.x || y >= dims.y {
        return;
    }
    
    // Read depth from texture
    let depth = textureLoad(depthTexture, vec2<u32>(x, y), 0).r;
    
    // Write coc radius to texture - only need the r component since we're using r32float
    textureStore(cocTexture, vec2<u32>(x, y), vec4<f32>(calculate_coc_radius(depth, params.focus_depth), 0.0, 0.0, 0.0));
} 