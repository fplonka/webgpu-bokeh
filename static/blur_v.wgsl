struct Params {
    width: u32,
    height: u32,
    shape: u32,  // 0 = circle, 1 = square, 2 = hexagon
}

@group(0) @binding(0) var<storage, read> inputImage: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cocBuffer: array<f32>;

fn extractChannel(color: u32, shift: u32) -> f32 {
    return f32((color >> shift) & 0xFFu);
}

fn combineChannels(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let ru = u32(clamp(r, 0.0, 255.0));
    let gu = u32(clamp(g, 0.0, 255.0));
    let bu = u32(clamp(b, 0.0, 255.0));
    let au = u32(clamp(a, 0.0, 255.0));
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if x >= params.width || y >= params.height {
        return;
    }

    let centerIdx = y * params.width + x;
    let centerCoC = cocBuffer[centerIdx];

    var sumColor = vec4<f32>(0.0);
    var sumWeight = 0.0;
    let radius = i32(ceil(centerCoC));

    // Vertical pass only samples along y
    for (var dy = -radius; dy <= radius; dy++) {
        let sampleY = i32(y) + dy;
        
        // Skip samples outside the image
        if sampleY < 0 || sampleY >= i32(params.height) {
            continue;
        }

        let sampleIdx = u32(sampleY) * params.width + x;
        let sampleCoC = cocBuffer[sampleIdx];
        
        // For vertical pass, we only check y distance
        let fdy = f32(dy);
        var isInShape = false;
        
        if params.shape == 0u { // circle
            isInShape = abs(fdy) <= sampleCoC;
        } else if params.shape == 1u { // square
            isInShape = abs(fdy) <= sampleCoC;
        } else { // hexagon
            isInShape = abs(fdy) <= sampleCoC;
        }

        if !isInShape {
            continue;
        }

        sumColor += unpackRGBA(inputImage[sampleIdx]);
        sumWeight += 1.0;
    }

    if sumWeight > 0.0 {
        outputImage[centerIdx] = packRGBA(sumColor / sumWeight);
    } else {
        outputImage[centerIdx] = inputImage[centerIdx];
    }
}
