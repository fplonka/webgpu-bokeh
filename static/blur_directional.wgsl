struct Params {
    width: u32,
    height: u32,
    num_samples: u32,
}

@group(0) @binding(0) var<storage, read> inputImage: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> cocBuffer: array<f32>;
@group(0) @binding(4) var<storage, read> offsets: array<vec2<f32>>;

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

    // Sample along the provided offsets
    for (var i = 0u; i < params.num_samples; i++) {
        let offset = offsets[i];
        let floatSampleX = round(f32(x) + offset.x * centerCoC);
        let floatSampleY = round(f32(y) + offset.y * centerCoC);
        if floatSampleX < 0.0 || floatSampleX >= f32(params.width) ||
           floatSampleY < 0.0 || floatSampleY >= f32(params.height) {
            continue;
        }

        let sampleX = u32(floatSampleX);
        let sampleY = u32(floatSampleY);
        
        // Check bounds
        if (sampleX >= 0 && sampleX < params.width && 
            sampleY >= 0 && sampleY < params.height) {
            
            let sampleIdx = sampleY * params.width + sampleX;
            
            let sampleCoC = cocBuffer[sampleIdx];
            let dist = length(vec2<f32>(offset.x * centerCoC, offset.y * centerCoC));
            if dist > sampleCoC {
                continue;
            }

            let sampleColor = inputImage[sampleIdx];
            
            // Use a simple box filter weight for now
            // Could be extended to use different weight functions
            let weight = 1.0;
            
            sumColor += sampleColor * weight;
            sumWeight += weight;
        }
    }

    // Normalize the result
    if (sumWeight > 0.0) {
        outputImage[centerIdx] = sumColor / sumWeight;
    } else {
        outputImage[centerIdx] = inputImage[centerIdx];
    }
}
