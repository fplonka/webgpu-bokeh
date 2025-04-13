@group(0) @binding(0) var depthTexture: texture_2d<f32>;
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
    let idx = global_id.xy;
    
    textureStore(cocTexture, idx, vec4<f32>(calculate_coc_radius(textureLoad(depthTexture, idx, 0).r, params.focus_depth)));
} 