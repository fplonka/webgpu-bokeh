struct Params {
    num_samples: u32,
}

// Input texture containing the image data to be blurred
@group(0) @binding(0) var inputTexture: texture_2d<f32>;
// Output texture for the blurred result
@group(0) @binding(1) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;
// Circle of Confusion texture - using r32float allows for precise CoC values
@group(0) @binding(3) var cocTexture: texture_2d<f32>;
// Sample offsets for the blur pattern
@group(0) @binding(4) var<storage, read> offsets: array<vec2<f32>>;
// Sampler for smooth texture interpolation
@group(0) @binding(5) var textureSampler: sampler;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.xy;
    
    // Get dimensions directly from the textures
    let dims = textureDimensions(inputTexture);

    let centerCoC = textureLoad(cocTexture, idx, 0).r;

    var sumColor = vec4<f32>(0.0);
    var sumWeight = 0.0;

    // Sample along the provided offsets with texture sampling
    if (centerCoC <= 0.5) { // basically sharp - don't do anything
        textureStore(outputTexture, idx, textureLoad(inputTexture, idx, 0));
        return;
    }

    for (var i = 0u; i < params.num_samples; i++) {
        let offset = offsets[i];
        
        // Calculate sample position in normalized texture coordinates
        let sampleUv = (vec2<f32>(idx) + offset * centerCoC) / vec2<f32>(dims);
        
        // Get the nearest integer coordinates for CoC lookup
        let cocIdx = vec2<u32>(round(vec2<f32>(idx) + offset * centerCoC));
        
        // Read sample CoC from texture at nearest integer position
        let sampleCoC = textureLoad(cocTexture, cocIdx, 0).r;
        
        // Check if this sample is within the CoC of its center
        let dist = length(offset * centerCoC);
        if (dist > sampleCoC) {
            continue;
        }
        
        sumColor += textureSampleLevel(inputTexture, textureSampler, sampleUv, 0.0);
        sumWeight += 1.0;
    }

    if (sumWeight > 0.0) {
        textureStore(outputTexture, idx, sumColor / sumWeight);
    } else {
        textureStore(outputTexture, idx, textureLoad(inputTexture, idx, 0));
    }
}
