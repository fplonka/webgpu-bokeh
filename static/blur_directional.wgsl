struct Params {
    width: u32,
    height: u32,
    num_samples: u32,
}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var cocTexture: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> offsets: array<vec2<f32>>;
@group(0) @binding(5) var textureSampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let centerCoC = textureLoad(cocTexture, vec2<u32>(x, y), 0).r;

    var sumColor = vec4<f32>(0.0);
    var sumWeight = 0.0;

    // Sample along the provided offsets with texture sampling
    for (var i = 0u; i < params.num_samples; i++) {
        let offset = offsets[i];
        
        // Calculate sample position in normalized texture coordinates
        let samplePosX = f32(x) + offset.x * centerCoC;
        let samplePosY = f32(y) + offset.y * centerCoC;
        let sampleUV = vec2<f32>(samplePosX / f32(params.width), samplePosY / f32(params.height));
        
        // Skip if outside texture bounds
        if (sampleUV.x < 0.0 || sampleUV.x >= 1.0 || sampleUV.y < 0.0 || sampleUV.y >= 1.0) {
            continue;
        }
        
        // Get the nearest integer coordinates for CoC lookup
        let nearestX = u32(round(samplePosX));
        let nearestY = u32(round(samplePosY));
        
        // Clamp to texture bounds for CoC lookup
        let cocX = min(max(nearestX, 0u), params.width - 1u);
        let cocY = min(max(nearestY, 0u), params.height - 1u);
        
        // Read sample CoC from texture at nearest integer position
        let sampleCoC = textureLoad(cocTexture, vec2<u32>(cocX, cocY), 0).r;
        // let sampleCoC = textureSampleLevel(cocTexture, textureSampler, sampleUV, 0.0).r; // TODO:
        
        // Check if this sample is within the CoC of its center
        let dist = length(vec2<f32>(offset.x * centerCoC, offset.y * centerCoC));
        if (dist > sampleCoC) {
            continue;
        }
        
        // Use smooth texture sampling for the color
        sumColor += textureSampleLevel(inputTexture, textureSampler, sampleUV, 0.0);
        sumWeight += 1.0;
    }

    if (sumWeight > 0.0) {
        textureStore(outputTexture, vec2<u32>(x, y), sumColor / sumWeight);
    } else {
        textureStore(outputTexture, vec2<u32>(x, y), textureLoad(inputTexture, vec2<u32>(x, y), 0));
    }
}
