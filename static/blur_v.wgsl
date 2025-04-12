struct Params {
    width: u32,
    height: u32,
    shape: u32,  // 0 = circle, 1 = square, 2 = hexagon
}

@group(0) @binding(0) var<storage, read> inputImage: array<u32>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cocBuffer: array<f32>;

fn srgbToLinear(x: f32) -> f32 {
    if x <= 0.04045 {
        return x / 12.92;
    } else {
        return pow((x + 0.055) / 1.055, 2.4);
    }
}

fn linearToSrgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    } else {
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
}

fn unpackRGBA(color: u32) -> vec4<f32> {
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

fn packRGBA(color: vec4<f32>) -> u32 {
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
