#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform float u_waterHeight;

// Scatter Rules
// R: Trees (Moderate slope, moderate height, high moisture/erosion)
// G: Rocks (Steep slope, any height)
// B: Grass (Flat slope, low-mid height)
// A: Unused

void main() {
    vec4 data = texture(u_heightmap, uv);
    float height = data.x;
    vec3 normal;
    normal.xy = data.yz;
    normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
    float erosion = data.w;
    
    // Calculate slope (0 = flat, 1 = vertical)
    // In our normal map, Z is up. So slope is 1.0 - normal.z
    float slope = 1.0 - normal.z;
    
    // --- Trees ---
    // Prefer:
    // - Height: Above water, below snow line (approx 0.8)
    // - Slope: Flat to moderate (< 0.4)
    // - Erosion: High (proxy for moisture/drainage channels)
    float treeHeight = smoothstep(u_waterHeight + 0.01, u_waterHeight + 0.1, height) * 
                       smoothstep(0.8, 0.6, height);
    float treeSlope = smoothstep(0.4, 0.2, slope);
    float treeMoisture = smoothstep(0.3, 0.6, erosion); // More erosion = more water flow
    float treeDensity = treeHeight * treeSlope * (0.2 + 0.8 * treeMoisture);
    
    // --- Rocks ---
    // Prefer:
    // - Slope: Steep (> 0.3)
    // - Height: Any, but more common high up
    float rockSlope = smoothstep(0.2, 0.5, slope);
    float rockHeight = smoothstep(u_waterHeight, u_waterHeight + 0.05, height);
    float rockDensity = rockSlope * rockHeight;
    
    // --- Grass ---
    // Prefer:
    // - Height: Above water, below high peaks
    // - Slope: Flat (< 0.3)
    // - Inverse of trees/rocks (fill the gaps)
    float grassHeight = smoothstep(u_waterHeight + 0.005, u_waterHeight + 0.05, height) * 
                        smoothstep(0.9, 0.7, height);
    float grassSlope = smoothstep(0.3, 0.1, slope);
    float grassDensity = grassHeight * grassSlope;
    
    // Suppress grass where trees or rocks are dense
    grassDensity *= (1.0 - treeDensity * 0.8);
    grassDensity *= (1.0 - rockDensity);
    
    fragColor = vec4(treeDensity, rockDensity, grassDensity, 1.0);
}
