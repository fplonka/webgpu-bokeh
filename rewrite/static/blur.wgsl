@group(0) @binding(0) var<storage, read> inputImage: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<u32>;
@group(0) @binding(2) var<uniform> dimensions: vec2<u32>;
@group(0) @binding(3) var<storage, read> cocBuffer: array<f32>;

fn unpackRGBA(color: u32) -> vec4<f32> {
    let r = f32((color >> 0u) & 0xFFu) / 255.0;
    let g = f32((color >> 8u) & 0xFFu) / 255.0;
    let b = f32((color >> 16u) & 0xFFu) / 255.0;
    let a = f32((color >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, a);
}

fn packRGBA(color: vec4<f32>) -> u32 {
    let r = u32(clamp(color.r, 0.0, 1.0) * 255.0);
    let g = u32(clamp(color.g, 0.0, 1.0) * 255.0);
    let b = u32(clamp(color.b, 0.0, 1.0) * 255.0);
    let a = u32(clamp(color.a, 0.0, 1.0) * 255.0);
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Skip if outside the image
    if x >= dimensions.x || y >= dimensions.y {
        return;
    }

    let centerIdx = y * dimensions.x + x;
    let centerCoC = cocBuffer[centerIdx];

    var sumColor = vec4<f32>(0.0);
    var sumWeight = 0.0;
    let radius = i32(ceil(centerCoC));

    // Bokeh blur: weighted average based on CoC
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let sampleX = i32(x) + dx;
            let sampleY = i32(y) + dy;

            // Skip samples outside the image
            if sampleX < 0 || sampleX >= i32(dimensions.x) || sampleY < 0 || sampleY >= i32(dimensions.y) {
                continue;
            }

            let sampleIdx = u32(sampleY) * dimensions.x + u32(sampleX);
            let sampleCoC = cocBuffer[sampleIdx];
            
            // Distance from center pixel
            let dist = sqrt(f32(dx * dx + dy * dy));

            if dist > sampleCoC {
                continue;
            }
            
            // Weight based on distance and CoC
            var weight = 1.0;

            sumColor += unpackRGBA(inputImage[sampleIdx]) * weight;
            sumWeight += weight;
        }
    }

    // Normalize and write output
    let outputIdx = y * dimensions.x + x;
    if sumWeight > 0.0 {
        outputImage[outputIdx] = packRGBA(sumColor / sumWeight);
    } else {
        outputImage[outputIdx] = inputImage[outputIdx];
    }
} 