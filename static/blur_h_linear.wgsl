struct Params {
    width: u32,
    height: u32,
    shape: u32,  // 0 = circle, 1 = square, 2 = hexagon
}

@group(0) @binding(0) var<storage, read> inputImage: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cocBuffer: array<f32>;

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

    // Horizontal pass only samples along x
    for (var dx = -radius; dx <= radius; dx++) {
        let sampleX = i32(x) + dx;
        
        // Skip samples outside the image
        if sampleX < 0 || sampleX >= i32(params.width) {
            continue;
        }

        let sampleIdx = y * params.width + u32(sampleX);
        let sampleCoC = cocBuffer[sampleIdx];
        
        // For horizontal pass, we only check x distance
        let fdx = f32(dx);
        var isInShape = false;
        
        if params.shape == 0u { // circle
            isInShape = abs(fdx) <= sampleCoC;
        } else if params.shape == 1u { // square
            isInShape = abs(fdx) <= sampleCoC;
        } else { // hexagon
            isInShape = abs(fdx) <= sampleCoC;
        }

        if !isInShape {
            continue;
        }

        // Directly add the vec4 values - no need for unpacking
        sumColor += inputImage[sampleIdx];
        sumWeight += 1.0;
    }

    if sumWeight > 0.0 {
        outputImage[centerIdx] = sumColor / sumWeight;
    } else {
        outputImage[centerIdx] = inputImage[centerIdx];
    }
}
