#version 330

// River Carving Shader
// Applies river channels to heightmap based on flow accumulation

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;      // Original terrain height
uniform sampler2D u_riverFlow;      // River flow data (R=accum, G=intensity, B=dir, A=slope)
uniform vec2 u_texelSize;
uniform float u_riverDepth;         // Maximum river depth carving (default: 0.05)
uniform float u_riverWidth;         // River width multiplier (default: 1.0)
uniform float u_bankSlope;          // Steepness of river banks (default: 2.0)
uniform float u_meander;            // Meandering strength (default: 0.3)
uniform float u_seed;               // Seed for meander noise

// Simple hash for noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// Value noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// FBM noise for meander
float fbm(vec2 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

void main() {
    vec4 riverData = texture(u_riverFlow, uv);
    float flowAccum = riverData.r;
    float riverIntensity = riverData.g;
    float flowDir = riverData.b;
    float slope = riverData.a;
    
    float height = texture(u_heightmap, uv).r;
    
    // Calculate river properties based on flow accumulation
    float riverStrength = riverIntensity;
    
    // River width increases with flow (logarithmic)
    float width = u_riverWidth * (0.5 + 0.5 * log(1.0 + flowAccum * 100.0));
    width = clamp(width, 0.5, 5.0);
    
    // Add meandering using noise
    vec2 meanderOffset = vec2(
        fbm(uv * 50.0 + u_seed, 3) - 0.5,
        fbm(uv * 50.0 + u_seed + 100.0, 3) - 0.5
    ) * u_meander * u_texelSize * width;
    
    // Sample river intensity with meander offset
    vec2 offsetUV = uv + meanderOffset;
    offsetUV = clamp(offsetUV, vec2(0.0), vec2(1.0));
    float meanderRiver = texture(u_riverFlow, offsetUV).g;
    riverStrength = max(riverStrength, meanderRiver * 0.8);
    
    // Calculate river depth (deeper for larger rivers)
    float depth = u_riverDepth * riverStrength * (0.5 + 0.5 * sqrt(flowAccum));
    depth = clamp(depth, 0.0, u_riverDepth);
    
    // Create smooth river banks using sigmoid
    float bankFalloff = smoothstep(0.0, 1.0, riverStrength);
    bankFalloff = pow(bankFalloff, u_bankSlope);
    
    // Apply carving
    float carvedHeight = height - depth * bankFalloff;
    
    // Ensure rivers stay above minimum height
    carvedHeight = max(carvedHeight, 0.0);
    
    // Calculate river mask (where water would be)
    float waterLevel = carvedHeight + depth * 0.3; // Water fills 30% of carved depth
    float riverMask = riverStrength > 0.1 ? 1.0 : 0.0;
    
    // Moisture map (rivers increase nearby moisture)
    float moisture = riverStrength;
    
    // Spread moisture to nearby areas
    for (int dx = -2; dx <= 2; dx++) {
        for (int dy = -2; dy <= 2; dy++) {
            vec2 neighborUV = uv + vec2(float(dx), float(dy)) * u_texelSize;
            neighborUV = clamp(neighborUV, vec2(0.0), vec2(1.0));
            float neighborRiver = texture(u_riverFlow, neighborUV).g;
            float dist = length(vec2(float(dx), float(dy)));
            moisture = max(moisture, neighborRiver * exp(-dist * 0.5));
        }
    }
    
    // Output:
    // R: Carved height
    // G: River mask (binary-ish)
    // B: Moisture map
    // A: Water depth
    fragColor = vec4(carvedHeight, riverMask, moisture, depth * bankFalloff);
}
