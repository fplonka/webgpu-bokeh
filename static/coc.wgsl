@group(0) @binding(0) var<storage, read> depthMap: array<f32>;
@group(0) @binding(1) var<storage, read_write> cocBuffer: array<f32>;
@group(0) @binding(2) var<uniform> dimensions: vec2<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    focus_depth: f32,
    max_coc: f32,
    depth_of_field: f32,
}

fn calculate_coc_radius(pixel_depth: f32, focal_depth: f32) -> f32 {
    return params.max_coc * max(abs(focal_depth - pixel_depth) - params.depth_of_field / 2.0, 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= dimensions.x || y >= dimensions.y {
        return;
    }

    let idx = y * dimensions.x + x;
    let depth = depthMap[idx];
    cocBuffer[idx] = calculate_coc_radius(depth, params.focus_depth);
} 