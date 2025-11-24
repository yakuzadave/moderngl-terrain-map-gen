#version 330

// River Flow Accumulation - Proper D8 Algorithm
// Pass 1: Initialize with flow direction from heightmap
// Pass N: Accumulate flow from upstream cells

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform sampler2D u_flowAccum;
uniform vec2 u_texelSize;
uniform float u_riverThreshold;

// 8 directions
const vec2 dirs[8] = vec2[8](
    vec2(-1.0,  0.0), vec2( 1.0,  0.0),
    vec2( 0.0, -1.0), vec2( 0.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0, -1.0),
    vec2(-1.0,  1.0), vec2( 1.0,  1.0)
);
const float dists[8] = float[8](1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414);
const int opposites[8] = int[8](1, 0, 3, 2, 7, 6, 5, 4);

void main() {
    float h = texture(u_heightmap, uv).r;
    vec4 prev = texture(u_flowAccum, uv);
    
    // Always compute flow direction from heightmap (stable)
    float maxSlope = 0.0;
    int myFlowDir = -1;
    for (int i = 0; i < 8; i++) {
        vec2 nUV = uv + dirs[i] * u_texelSize;
        if (nUV.x < 0.0 || nUV.x > 1.0 || nUV.y < 0.0 || nUV.y > 1.0) continue;
        float nh = texture(u_heightmap, nUV).r;
        float slope = (h - nh) / dists[i];
        if (slope > maxSlope) {
            maxSlope = slope;
            myFlowDir = i;
        }
    }
    
    // Start with base flow (rainfall = 1)
    float newFlow = 1.0;
    
    // Add flow from neighbors that flow INTO this cell
    for (int i = 0; i < 8; i++) {
        vec2 nUV = uv + dirs[i] * u_texelSize;
        if (nUV.x < 0.0 || nUV.x > 1.0 || nUV.y < 0.0 || nUV.y > 1.0) continue;
        
        // Compute neighbor's flow direction from HEIGHTMAP (not from flow texture!)
        float nh = texture(u_heightmap, nUV).r;
        float nMaxSlope = 0.0;
        int nFlowDir = -1;
        for (int j = 0; j < 8; j++) {
            vec2 nnUV = nUV + dirs[j] * u_texelSize;
            if (nnUV.x < 0.0 || nnUV.x > 1.0 || nnUV.y < 0.0 || nnUV.y > 1.0) continue;
            float nnh = texture(u_heightmap, nnUV).r;
            float slope = (nh - nnh) / dists[j];
            if (slope > nMaxSlope) {
                nMaxSlope = slope;
                nFlowDir = j;
            }
        }
        
        // If neighbor flows toward us (opposite direction)
        if (nFlowDir == opposites[i]) {
            float nFlow = texture(u_flowAccum, nUV).r;
            newFlow += nFlow;
        }
    }
    
    // Cap to prevent explosion
    newFlow = min(newFlow, 10000.0);
    
    // River intensity
    float riverVis = smoothstep(u_riverThreshold * 0.3, u_riverThreshold, newFlow);
    
    // Store flow direction
    float dirEnc = myFlowDir >= 0 ? float(myFlowDir) / 7.0 : 0.5;
    
    fragColor = vec4(newFlow, riverVis, dirEnc, maxSlope);
}
