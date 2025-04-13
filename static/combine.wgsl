// Combine shader - takes the less bright pixel between two textures
// This is equivalent to the combine_less_bright function in the Python code

@group(0) @binding(0) var textureA: texture_2d<f32>;
@group(0) @binding(1) var textureB: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// Calculates perceived brightness (luminance) of a color
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.xy;
    
    let colorA = textureLoad(textureA, idx, 0);
    let colorB = textureLoad(textureB, idx, 0);
    
    let brightnessA = luminance(colorA.rgb);
    let brightnessB = luminance(colorB.rgb);
    
    let resultColor = select(colorA, colorB, brightnessA > brightnessB);
    textureStore(outputTexture, idx, resultColor);
}
