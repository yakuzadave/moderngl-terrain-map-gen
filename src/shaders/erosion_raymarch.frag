#version 330

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_heightmap;
uniform sampler2D u_detail;
uniform vec2 u_res;
uniform float u_time;
uniform vec3 u_camPos;
uniform vec3 u_lookAt;
uniform float u_waterHeight;
uniform vec3 u_sunDir;
uniform float u_exposure;
uniform vec3 u_fogColor;
uniform float u_fogDensity;
uniform float u_fogHeight;
uniform float u_sunIntensity;
uniform int u_rayMaxSteps;
uniform float u_rayMinStep;
uniform float u_shadowSoftness;
uniform float u_aoStrength;

#define PI 3.14159265
#define saturate(x) clamp(x, 0.0, 1.0)
#define sq(x) (x*x)

#define CLIFF_COLOR    vec3(0.22, 0.2, 0.2)
#define DIRT_COLOR     vec3(0.6, 0.5, 0.4)
#define GRASS_COLOR1   vec3(0.15, 0.3, 0.1)
#define GRASS_COLOR2   vec3(0.4, 0.5, 0.2)
#define SAND_COLOR     vec3(0.8, 0.7, 0.6)
#define WATER_COLOR    vec3(0.0, 0.05, 0.1)
#define WATER_SHORE    vec3(0.0, 0.25, 0.25)
#define AMBIENT_COLOR  (vec3(0.3, 0.5, 0.7) * 0.1)
#define SUN_COLOR      (vec3(1.0, 0.98, 0.95) * 2.0)

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

float GetCloudShadow(vec2 pos) {
    if (u_time <= 0.0) return 1.0;
    vec2 cloudUV = pos * 0.5 + vec2(u_time * 0.02, u_time * 0.01);
    float cloud = fbm(cloudUV);
    return mix(1.0, 0.6, smoothstep(0.4, 0.8, cloud));
}

vec3 SkyColor(vec3 rd, vec3 sun) {
    float costh = dot(rd, sun);
    return AMBIENT_COLOR * PI * (1.0 - abs(costh) * 0.8);
}

vec3 Tonemap_ACES(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return (x * (a * x + b)) / (x * (c * x + d) + e);
}

float pow5(float x) { float x2 = x*x; return x2*x2*x; }
float D_GGX(float linearRoughness, float NoH, const vec3 h) {
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
    float a = NoH * linearRoughness;
    float k = linearRoughness / (oneMinusNoHSquared + a * a);
    return k * k * (1.0 / PI);
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
    return (1.0 + (f90 - 1.0) * pow5(1.0 - NoL)) * (1.0 + (f90 - 1.0) * pow5(1.0 - NoV)) * (1.0 / PI);
}
float Fd_Lambert() { return 1.0 / PI; }

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

vec4 map(vec3 p, out float erosion) {
    vec2 texUV = p.xz + 0.5;
    if (texUV.x < 0.0 || texUV.x > 1.0 || texUV.y < 0.0 || texUV.y > 1.0) {
        return vec4(-100.0, 0.0, 1.0, 0.0);
    }
    vec4 tex = texture(u_heightmap, texUV);
    float height = tex.x;
    vec3 normal;
    normal.xy = tex.yz;
    normal.z = sqrt(max(0.0, 1.0 - dot(normal.xy, normal.xy)));
    vec3 worldNormal = vec3(normal.x, normal.z, normal.y);
    erosion = tex.w;
    return vec4(height, worldNormal);
}

// Calculates soft shadows by raymarching towards the light source.
// k controls the softness (higher = sharper).
float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for(int i = 0; i < 64; i++) {
        if(t > maxt) break;
        vec3 p = ro + rd * t;
        float erosion;
        vec4 mapData = map(p, erosion);
        float h = mapData.x;
        float h_diff = p.y - h;
        if(h_diff < 0.001) return 0.0;
        res = min(res, k * h_diff / t);
        t += max(0.02, h_diff * 0.5);
    }
    return clamp(res, 0.0, 1.0);
}

// Calculates Horizon-Based Ambient Occlusion (HBAO) approximation.
// Samples height differences in the normal direction to estimate occlusion.
float calculateAO(vec3 p, vec3 n) {
    float occ = 0.0;
    float sca = 1.0;
    for(int i = 0; i < 5; i++) {
        float h = 0.05 + 0.15 * float(i);
        vec3 pos = p + n * h;
        float erosion;
        vec4 mapData = map(pos, erosion);
        float d = pos.y - mapData.x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ * u_aoStrength, 0.0, 1.0);
}

float march(vec3 ro, vec3 rd, out vec3 normal, out int material, out float t) {
    t = 0.0;
    vec3 boxMin = vec3(-0.5, 0.0, -0.5);
    vec3 boxMax = vec3(0.5, 1.0, 0.5);
    vec3 invRd = 1.0 / rd;
    vec3 t0 = (boxMin - ro) * invRd;
    vec3 t1 = (boxMax - ro) * invRd;
    vec3 tMin = min(t0, t1);
    vec3 tMaxBox = max(t0, t1);
    float tBoxIn = max(max(tMin.x, tMin.y), tMin.z);
    float tBoxOut = min(min(tMaxBox.x, tMaxBox.y), tMaxBox.z);

    if (tBoxIn > tBoxOut || tBoxOut < 0.0) return -1.0;

    t = max(0.0, tBoxIn);
    int maxSteps = max(1, u_rayMaxSteps);
    for(int i=0; i<256; i++) {
        if (i >= maxSteps) break;
        vec3 p = ro + rd * t;
        if (p.x < -0.5 || p.x > 0.5 || p.z < -0.5 || p.z > 0.5) {
            t += 0.05; continue;
        }

        float erosion;
        vec4 mapData = map(p, erosion);
        float h = mapData.x;
        normal = mapData.yzw;

        float d_terrain = p.y - h;
        float d_water = p.y - u_waterHeight;

        if (d_water < 0.001 && d_terrain > 0.0) {
            material = 1;
            normal = vec3(0.0, 1.0, 0.0);
            t += d_water;
            return t;
        }

        if (d_terrain < 0.001) {
            material = 0;
            t += d_terrain;
            return t;
        }

        float dt = min(abs(d_terrain), abs(d_water));
        t += max(u_rayMinStep, dt * 0.5);
        if (t > tBoxOut) break;
    }

    return -1.0;
}

vec3 ApplyFog(vec3 rgb, float dist, vec3 rayDir, vec3 sunDir) {
    // Distance fog (Exponential Squared for more realistic falloff)
    float fogDist = dist * u_fogDensity;
    float fogAmount = 1.0 - exp(-fogDist * fogDist);
    
    // Height fog (keeps valleys misty)
    float heightFog = exp(-max(0.0, (rayDir.y * dist)) * (1.0 / max(1e-3, u_fogHeight)));
    fogAmount = mix(fogAmount, 1.0, heightFog * 0.5);
    
    // Sun glare (Mie scattering)
    float sunAmount = max(dot(rayDir, sunDir), 0.0);
    vec3 fogColor = mix(u_fogColor, SUN_COLOR * 0.5, pow(sunAmount, 8.0));
    
    return mix(rgb, fogColor, saturate(fogAmount));
}

vec3 Triplanar(sampler2D tex, vec3 p, vec3 n, float scale) {
    vec3 x = texture(tex, p.yz * scale).rgb;
    vec3 y = texture(tex, p.xz * scale).rgb;
    vec3 z = texture(tex, p.xy * scale).rgb;
    
    vec3 w = abs(n);
    w = w / (w.x + w.y + w.z);
    
    return x * w.x + y * w.y + z * w.z;
}

void main() {
    vec3 ro = u_camPos;
    vec3 forward = normalize(u_lookAt - ro);
    vec3 right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, forward);

    vec2 screenUV = uv * 2.0 - 1.0;
    screenUV.x *= u_res.x / u_res.y;

    vec3 rd = normalize(forward * 2.0 + right * screenUV.x + up * screenUV.y);

    vec3 sun = normalize(u_sunDir);
    vec3 color = SkyColor(rd, sun);

    vec3 normal;
    int material;
    float t;
    float hit = march(ro, rd, normal, material, t);

    if (hit > 0.0) {
        vec3 p = ro + rd * hit;
        float erosion;
        vec4 mapData = map(p, erosion);

        // Triplanar mapping for detail texture
        vec3 detail = Triplanar(u_detail, p, normal, 4.0);
        
        vec3 diffuse = vec3(0.5);
        vec3 f0 = vec3(0.04);
        float smoothness = 0.0;
        float occlusion = saturate(erosion + 0.5);

        if (material == 0) {
            diffuse = CLIFF_COLOR * smoothstep(0.4, 0.52, p.y);
            diffuse = mix(diffuse, DIRT_COLOR, smoothstep(0.3, 0.0, occlusion));
            vec3 grassMix = mix(GRASS_COLOR1, GRASS_COLOR2, smoothstep(0.4, 0.6, p.y - erosion * 0.05));
            diffuse = mix(diffuse, grassMix, smoothstep(u_waterHeight + 0.05, u_waterHeight + 0.02, p.y) * smoothstep(0.7, 1.0, normal.y));
            diffuse = mix(diffuse, vec3(1.0), smoothstep(0.53, 0.6, p.y));
            vec3 sandColor = SAND_COLOR;
            float sandBlend = smoothstep(u_waterHeight + 0.005, u_waterHeight, p.y);
            diffuse = mix(diffuse, sandColor, sandBlend);
            
            // Apply detail texture modulation
            diffuse *= (detail * 0.5 + 0.5);
        } else {
            // Water rendering
            float waterDepth = max(0.0, p.y - mapData.x);
            
            // Beer's Law absorption
            vec3 absorption = vec3(0.8, 0.2, 0.1); // Absorbs red fast, blue slow
            vec3 transmittance = exp(-waterDepth * absorption * 20.0);
            
            // Base water color mixed with shore
            float shore = exp(-waterDepth * 60.0);
            vec3 waterBase = mix(WATER_COLOR, WATER_SHORE, shore);
            
            // Apply absorption to the terrain below (fake refraction)
            // Ideally we'd trace the refracted ray, but for now we tint based on depth
            diffuse = waterBase * transmittance;
            
            f0 = vec3(0.02);
            smoothness = 0.9;
            
            // Animate water detail
            vec3 p_water = p;
            if (u_time > 0.0) {
                p_water.x += u_time * 0.1;
                p_water.z += u_time * 0.05;
            }
            vec3 waterDetail = Triplanar(u_detail, p_water, vec3(0,1,0), 4.0);
            normal = normalize(vec3(0.0, 1.0, 0.0) + waterDetail * 0.05);
            
            // Add sky reflection
            vec3 refDir = reflect(rd, normal);
            vec3 skyRef = SkyColor(refDir, sun);
            float fresnel = pow(1.0 - max(dot(-rd, normal), 0.0), 5.0);
            diffuse = mix(diffuse, skyRef, fresnel * 0.5);
        }

        vec3 viewDir = -rd;

        // Calculate Soft Shadow and AO
        float shadow = softShadow(p + normal * 0.01, sun, 0.02, 5.0, u_shadowSoftness);
        shadow *= GetCloudShadow(p.xz); // Add cloud shadow
        float ao = calculateAO(p, normal);
        occlusion *= ao;

        vec3 color = diffuse * SkyColor(normal, sun) * occlusion * Fd_Lambert();
        color += Shade(diffuse, f0, smoothness, normal, viewDir, sun, SUN_COLOR * u_sunIntensity * shadow);

        // Apply improved fog
        color = ApplyFog(color, hit, rd, sun);
    }

    color *= u_exposure;
    color = Tonemap_ACES(color);
    color = pow(color, vec3(1.0 / 2.2));
    fragColor = vec4(color, 1.0);
}
