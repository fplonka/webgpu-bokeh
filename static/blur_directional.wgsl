struct Params {
    num_samples: u32,
}

// Input texture containing the image data to be blurred
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
// Output texture for the blurred result
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: Params;
// Circle of Confusion texture - using r32float allows for precise CoC values
@group(0) @binding(3) var cocTexture: texture_2d<f32>;
// Sample offsets for the blur pattern
@group(0) @binding(4) var<storage, read> offsets: array<vec2<f32>>;
// Sampler for smooth texture interpolation
@group(0) @binding(5) var textureSampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Get dimensions directly from the textures
    let dims = textureDimensions(inputTexture);
    
    // Bounds check
    if x >= dims.x || y >= dims.y {
        return;
    }

    let centerCoC = textureLoad(cocTexture, vec2<u32>(x, y), 0).r;

    var sumColor = vec4<f32>(0.0);
    var sumWeight = 0.0;

    // Sample along the provided offsets with texture sampling
    for (var i = 0u; i < params.num_samples; i++) {
        let offset = offsets[i];
        
        // Calculate sample position in normalized texture coordinates
        let samplePosX = f32(x) + offset.x * centerCoC;
        let samplePosY = f32(y) + offset.y * centerCoC;
        let sampleUV = vec2<f32>(samplePosX / f32(dims.x), samplePosY / f32(dims.y));
        
        // Skip if outside texture bounds
        if (sampleUV.x < 0.0 || sampleUV.x >= 1.0 || sampleUV.y < 0.0 || sampleUV.y >= 1.0) {
            continue;
        }
        
        // Get the nearest integer coordinates for CoC lookup
        let nearestX = u32(round(samplePosX));
        let nearestY = u32(round(samplePosY));
        
        // Clamp to texture bounds for CoC lookup
        let cocX = min(max(nearestX, 0u), dims.x - 1u);
        let cocY = min(max(nearestY, 0u), dims.y - 1u);
        
        // Read sample CoC from texture at nearest integer position
        let sampleCoC = textureLoad(cocTexture, vec2<u32>(cocX, cocY), 0).r;
        
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
