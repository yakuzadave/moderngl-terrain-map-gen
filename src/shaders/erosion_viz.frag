#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform sampler2D u_detail;
uniform float u_waterHeight;
uniform vec3 u_sunDir;
uniform int u_mode;
uniform int u_useTriplanar; // 1 to enable
uniform float u_time;

#define CLIFF_COLOR    vec3(0.22, 0.2, 0.2)
#define DIRT_COLOR     vec3(0.6, 0.5, 0.4)
#define GRASS_COLOR1   vec3(0.15, 0.3, 0.1)
#define GRASS_COLOR2   vec3(0.4, 0.5, 0.2)
#define SAND_COLOR     vec3(0.8, 0.7, 0.6)
#define WATER_COLOR    vec3(0.0, 0.05, 0.1)
#define WATER_SHORE_COLOR vec3(0.0, 0.25, 0.25)
#define AMBIENT_COLOR  (vec3(0.3, 0.5, 0.7) * 0.1)
#define SUN_COLOR      (vec3(1.0, 0.98, 0.95) * 2.0)

#define PI 3.14159265358979
#define saturate(x) clamp(x, 0.0, 1.0)
#define sq(x) (x*x)

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), f.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += noise(p) * a;
        p *= 2.0;
        a *= 0.5;
    }
    return v;
}

vec3 Triplanar(sampler2D tex, vec3 p, vec3 n, float scale) {
    vec3 x = texture(tex, p.yz * scale).rgb;
    vec3 y = texture(tex, p.xz * scale).rgb;
    vec3 z = texture(tex, p.xy * scale).rgb;
    
    vec3 w = abs(n);
    w = w / (w.x + w.y + w.z);
    
    return x * w.x + y * w.y + z * w.z;
}

// Raymarch shadows in 2.5D heightmap space
float CalculateShadow(vec3 pos, vec3 lightDir) {
    vec3 dir = normalize(lightDir);
    vec2 uvDir = dir.xz;
    float slope = dir.y;
    
    // Step size in UV space
    float stepSize = 1.0 / 256.0; 
    float t = stepSize;
    float shadow = 1.0;
    
    // Optimization: Scale step size by slope to avoid over-stepping on flat angles
    // But fixed step is safer for heightmaps
    
    for (int i = 0; i < 128; i++) {
        vec2 sampleUV = pos.xz + uvDir * t;
        
        // Check bounds
        if (sampleUV.x < 0.0 || sampleUV.x > 1.0 || sampleUV.y < 0.0 || sampleUV.y > 1.0) break;
        
        float h = texture(u_heightmap, sampleUV).x;
        float rayH = pos.y + slope * t;
        
        if (h > rayH) {
            return 0.0; // Hard shadow
        }
        
        // Soft shadow approximation (optional, can produce artifacts with low step count)
        // float hDiff = rayH - h;
        // shadow = min(shadow, 16.0 * hDiff / t);
        
        t += stepSize;
    }
    return shadow;
}

vec3 SkyColor(vec3 normal, vec3 sun) {
    float costh = dot(normal, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

float D_GGX(float linearRoughness, float NoH, const vec3 h) {
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

float V_SmithGGXCorrelated(float linearRoughness, float NoV, float NoL) {
    float a2 = linearRoughness * linearRoughness;
    float GGXV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    float GGXL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    return 0.5 / (GGXV + GGXL);
}

vec3 F_Schlick(const vec3 f0, float VoH) {
    return f0 + (vec3(1.0) - f0) * pow5(1.0 - VoH);
}

float Fd_Burley(float linearRoughness, float NoV, float NoL, float LoH) {
    float f90 = 0.5 + 2.0 * linearRoughness * LoH * LoH;
    float lightScatter = 1.0 + (f90 - 1.0) * pow5(1.0 - NoL);
    float viewScatter  = 1.0 + (f90 - 1.0) * pow5(1.0 - NoV);
    return lightScatter * viewScatter * (1.0 / PI);
}

float Fd_Lambert() {
    return 1.0 / PI;
}

vec3 Shade(vec3 diffuse, vec3 f0, float smoothness, vec3 n, vec3 v, vec3 l, vec3 lc) {
    vec3 h = normalize(v + l);

    float NoV = abs(dot(n, v)) + 1e-5;
    float NoL = saturate(dot(n, l));
    float NoH = saturate(dot(n, h));
    float LoH = saturate(dot(l, h));

    float roughness = 1.0 - smoothness;
    float linearRoughness = roughness * roughness;
    float D = D_GGX(linearRoughness, NoH, h);
    float V = V_SmithGGXCorrelated(linearRoughness, NoV, NoL);
    vec3 F = F_Schlick(f0, LoH);
    vec3 Fr = (D * V) * F;

    vec3 Fd = diffuse * Fd_Burley(linearRoughness, NoV, NoL, LoH);

    return (Fd + Fr) * lc * NoL;
}

void main() {
    vec4 data = texture(u_heightmap, uv);
    float height = data.x;
    vec3 normal;
    normal.xy = data.yz;
    normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
    float erosion = data.w;

    if (u_mode == 1) {
        fragColor = vec4(vec3(height), 1.0);
        return;
    }
    if (u_mode == 2) {
        fragColor = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }
    if (u_mode == 3) {
        fragColor = vec4(vec3(erosion), 1.0);
        return;
    }

    // Calculate derivatives for advanced viz
    vec2 texSize = vec2(textureSize(u_heightmap, 0));
    vec2 texel = 1.0 / texSize;
    
    float h_L = texture(u_heightmap, uv + vec2(-texel.x, 0)).x;
    float h_R = texture(u_heightmap, uv + vec2(texel.x, 0)).x;
    float h_D = texture(u_heightmap, uv + vec2(0, -texel.y)).x;
    float h_U = texture(u_heightmap, uv + vec2(0, texel.y)).x;

    if (u_mode == 4) {
        // Slope (Magnitude of Gradient)
        vec2 gradient = vec2(h_R - h_L, h_U - h_D) * 0.5;
        float slope = length(gradient) * 100.0; // Scale for visibility
        fragColor = vec4(vec3(slope), 1.0);
        return;
    }
    if (u_mode == 5) {
        // Curvature (Laplacian)
        // Convex (Peak) > 0, Concave (Valley) < 0
        float curvature = (h_L + h_R + h_D + h_U - 4.0 * height);
        float curveVis = curvature * 200.0 + 0.5;
        fragColor = vec4(vec3(curveVis), 1.0);
        return;
    }

    vec3 sun = normalize(u_sunDir);
    vec3 viewDir = vec3(0.0, 0.0, 1.0);

    // Cloud Shadows
    float cloudShadow = 1.0;
    if (u_time > 0.0) {
        vec2 cloudUV = uv * 2.0 + vec2(u_time * 0.05, u_time * 0.02);
        float cloudNoise = fbm(cloudUV);
        cloudShadow = mix(1.0, 0.6, smoothstep(0.4, 0.7, cloudNoise));
    }

    vec3 breakupVec;
    float timeOffset = u_time * 0.1;
    breakupVec.x = fbm(uv * 32.0 + vec2(0.0, 17.2) + vec2(timeOffset, 0.0));
    breakupVec.y = fbm(uv * 58.0 + vec2(4.1, -9.7) + vec2(0.0, timeOffset));
    breakupVec.z = fbm(uv * 96.0 + vec2(-21.3, 3.8) - vec2(timeOffset, timeOffset));
    breakupVec = breakupVec * 2.0 - 1.0;

    float breakup = breakupVec.x * 0.45;
    float macroBreak = breakupVec.y * 0.25;
    float microBreak = breakupVec.z * 0.15;

    float occlusion = sq(saturate(erosion + 0.5));

    vec3 diffuseColor = CLIFF_COLOR * smoothstep(0.4, 0.52, height);
    diffuseColor = mix(diffuseColor, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion + breakup));

    vec3 grassMix = mix(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, height - erosion * 0.05 + breakup * 0.3));
    diffuseColor = mix(diffuseColor, grassMix, smoothstep(u_waterHeight + 0.05, u_waterHeight + 0.02, height - breakup * 0.02) * smoothstep(0.8, 1.0, normal.y + breakup * 0.1));

    diffuseColor = mix(diffuseColor, vec3(1.0), smoothstep(0.53, 0.6, height + breakup * 0.1));

    // Apply Tri-Planar Mapping if enabled
    if (u_useTriplanar > 0) {
        // Construct 3D position from UV and Height
        // Scale height to match UV scale roughly (assuming 0-1 UV space)
        vec3 p = vec3(uv.x, height * 0.5, uv.y);
        
        // Normal in viz shader: z is up.
        // Triplanar expects y to be up usually, but let's adapt.
        // If z is up, then:
        // Top projection: xy plane (using normal.z)
        // Side projections: yz and xz planes (using normal.x and normal.y)
        
        // Let's use a modified Triplanar for Z-up
        vec3 detail = vec3(0.0);
        float scale = 4.0;
        
        vec3 x = texture(u_detail, p.yz * scale).rgb; // Side
        vec3 y = texture(u_detail, p.xz * scale).rgb; // Side
        vec3 z = texture(u_detail, p.xy * scale).rgb; // Top
        
        vec3 w = abs(normal);
        w = w / (w.x + w.y + w.z);
        
        detail = x * w.x + y * w.y + z * w.z;
        
        // Modulate diffuse color
        diffuseColor *= (detail * 0.5 + 0.5);
    }

    // Calculate Shadow
    // Construct position in "Viz Space" (Y-up): X=uv.x, Y=height, Z=uv.y
    vec3 worldPos = vec3(uv.x, height, uv.y);
    float shadow = CalculateShadow(worldPos, sun);
    shadow *= cloudShadow; // Apply cloud shadow

    vec3 f0 = vec3(0.04);
    float smoothness = 0.0;

    if (height <= u_waterHeight + 0.0005) {
        float shore = normal.z < 0.99 ? exp(-(u_waterHeight - height) * 60.0) : 0.0;
        float foam = normal.z < 0.99 ? smoothstep(0.005, 0.0, (u_waterHeight - height) + breakup * 0.005) : 0.0;

        diffuseColor = mix(WATER_COLOR, WATER_SHORE_COLOR, shore);
        diffuseColor = mix(diffuseColor, vec3(1.0), foam);

        f0 = vec3(0.02);
        smoothness = 0.9;

        vec3 waterNormal = normalize(vec3(breakupVec.x, breakupVec.y, 15.0));
        normal = normalize(vec3(0.0, 0.0, 1.0) + waterNormal * 0.1);

        occlusion = 1.0;
    } else {
        vec3 sandColor = mix(SAND_COLOR * 0.8, SAND_COLOR * 1.1, saturate(0.5 + breakupVec.y * 0.5));
        float sandBlend = smoothstep(u_waterHeight + 0.005, u_waterHeight, height + breakup * 0.01);
        diffuseColor = mix(diffuseColor, sandColor, sandBlend);

        // Refined PBR Material Properties
        smoothness = 0.1; // Default (Grass/Dirt)
        
        // Sand is smoother
        smoothness = mix(smoothness, 0.3, sandBlend);
        
        // Wet sand (very close to water) is glossy
        float wetBlend = smoothstep(u_waterHeight + 0.002, u_waterHeight, height);
        smoothness = mix(smoothness, 0.7, wetBlend);
        
        // Cliffs (steep slopes) are slightly smoother than grass (rock face)
        float slope = 1.0 - normal.z; // normal.z is up in this shader's convention
        float rockBlend = smoothstep(0.1, 0.3, slope);
        smoothness = mix(smoothness, 0.2, rockBlend);
    }

    vec3 color = diffuseColor * SkyColor(normal, sun) * occlusion * Fd_Lambert();
    color += Shade(diffuseColor, f0, smoothness, normal, viewDir, sun, SUN_COLOR) * shadow;
    color += diffuseColor * SUN_COLOR * (dot(normal, sun * vec3(1.0,-1.0, 1.0)) * 0.5 + 0.5) * Fd_Lambert() / PI * occlusion;

    // Atmospheric Height Fog (Visual Depth Cue)
    float fogDensity = 0.0;
    float fogBase = u_waterHeight;
    float fogFalloff = 8.0;
    if (height > fogBase) {
        float h = height - fogBase;
        fogDensity = exp(-h * fogFalloff) * 0.4; // Max 40% fog
    }
    vec3 fogColor = vec3(0.7, 0.8, 0.9);
    color = mix(color, fogColor, fogDensity);

    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    color = (color * (a * color + b)) / (color * (c * color + d) + e);
    color = pow(color, vec3(1.0 / 2.2));

    fragColor = vec4(color, 1.0);
}
