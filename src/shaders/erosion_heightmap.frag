#version 330

in vec2 uv;
out vec4 fragColor;

uniform int useErosion;
uniform float u_seed;
uniform int u_seamless;  // NEW: 1 for seamless, 0 for regular

uniform float u_heightTiles;
uniform int   u_heightOctaves;
uniform float u_heightAmp;
uniform float u_heightGain;
uniform float u_heightLacunarity;
uniform float u_waterHeight;

uniform float u_erosionTiles;
uniform int   u_erosionOctaves;
uniform float u_erosionGain;
uniform float u_erosionLacunarity;
uniform float u_erosionSlopeStrength;
uniform float u_erosionBranchStrength;
uniform float u_erosionStrength;
uniform float u_warpStrength; // Strength of domain warping (0.0 to 1.0+)
uniform int u_ridgeNoise;     // 1 to enable inverted absolute noise (sharp peaks)

uniform vec2 u_texelSize;

#define PI 3.14159265358979

// Standard hash function
vec2 hash(in vec2 x) {
    const vec2 k = vec2(0.3183099, 0.3678794);
    x = x * k + k.yx;
    return -1.0 + 2.0 * fract(16.0 * k * fract(x.x * x.y * (x.x + x.y)));
}

// Seamless hash - wraps coordinates modulo tile size
vec2 hashSeamless(in vec2 x, float tileSize) {
    vec2 wrapped = mod(x, tileSize);
    return hash(wrapped);
}

// Seamless noise using domain repetition
vec3 noisedSeamless(in vec2 p, float tileSize) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    // Sample 4 corners with wrapping
    vec2 ga = hashSeamless(i + vec2(0.0, 0.0), tileSize);
    vec2 gb = hashSeamless(i + vec2(1.0, 0.0), tileSize);
    vec2 gc = hashSeamless(i + vec2(0.0, 1.0), tileSize);
    vec2 gd = hashSeamless(i + vec2(1.0, 1.0), tileSize);

    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));

    return vec3(
        va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
        ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
        du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va)
    );
}

vec3 noised(in vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    vec2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    vec2 ga = hash(i + vec2(0.0, 0.0));
    vec2 gb = hash(i + vec2(1.0, 0.0));
    vec2 gc = hash(i + vec2(0.0, 1.0));
    vec2 gd = hash(i + vec2(1.0, 1.0));

    float va = dot(ga, f - vec2(0.0, 0.0));
    float vb = dot(gb, f - vec2(1.0, 0.0));
    float vc = dot(gc, f - vec2(0.0, 1.0));
    float vd = dot(gd, f - vec2(1.0, 1.0));

    return vec3(
        va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd),
        ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) +
        du * (u.yx * (va - vb - vc + vd) + vec2(vb, vc) - va)
    );
}

vec3 erosion(in vec2 p, vec2 dir) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float f = 2.0 * PI;
    vec3 va = vec3(0.0);
    float wt = 0.0;

    for (int i = -2; i <= 1; i++) {
        for (int j = -2; j <= 1; j++) {
            vec2 o = vec2(float(i), float(j));
            vec2 h = hash(ip - o) * 0.5;
            vec2 pp = fp + o - h;
            float d = dot(pp, pp);
            float w = exp(-d * 2.0);
            wt += w;
            float mag = dot(pp, dir);
            va += vec3(cos(mag * f), -sin(mag * f) * dir) * w;
        }
    }
    return va / wt;
}

// Seamless erosion using domain repetition
vec3 erosionSeamless(in vec2 p, vec2 dir, float tileSize) {
    vec2 ip = floor(p);
    vec2 fp = fract(p);
    float f = 2.0 * PI;
    vec3 va = vec3(0.0);
    float wt = 0.0;

    for (int i = -2; i <= 1; i++) {
        for (int j = -2; j <= 1; j++) {
            vec2 o = vec2(float(i), float(j));
            vec2 h = hashSeamless(ip - o, tileSize) * 0.5;
            vec2 pp = fp + o - h;
            float d = dot(pp, pp);
            float w = exp(-d * 2.0);
            wt += w;
            float mag = dot(pp, dir);
            va += vec3(cos(mag * f), -sin(mag * f) * dir) * w;
        }
    }
    return va / wt;
}

vec2 generateHeightmap(vec2 uv) {
    vec2 p = (uv * u_heightTiles) + vec2(u_seed * 17.123, u_seed * 23.456);

    // Domain Warping: Distort the coordinate space using FBM noise
    // This creates fluid, organic shapes instead of rigid grid-like patterns.
    if (u_warpStrength > 0.0) {
        vec2 q = vec2(0.0);
        q.x = noised(p + vec2(0.0, 0.0)).x;
        q.y = noised(p + vec2(5.2, 1.3)).x;
        p += q * u_warpStrength;
    }

    vec3 n = vec3(0.0);
    float nf = 1.0;
    float na = u_heightAmp;

    // Use seamless noise if enabled
    if (u_seamless > 0) {
        for (int i = 0; i < 16; i++) {
            if (i >= u_heightOctaves) break;
            vec3 noiseVal = noisedSeamless(p * nf, u_heightTiles);
            
            // Ridge Noise: Invert and square the noise to create sharp "creases"
            // (1.0 - abs(n))^2
            if (u_ridgeNoise > 0) {
                noiseVal.x = 1.0 - abs(noiseVal.x);
                noiseVal.x = noiseVal.x * noiseVal.x; // Sharpen ridges
            }
            n += noiseVal * na * vec3(1.0, nf, nf);
            na *= u_heightGain;
            nf *= u_heightLacunarity;
        }
    } else {
        for (int i = 0; i < 16; i++) {
            if (i >= u_heightOctaves) break;
            vec3 noiseVal = noised(p * nf);
            if (u_ridgeNoise > 0) {
                noiseVal.x = 1.0 - abs(noiseVal.x);
                noiseVal.x = noiseVal.x * noiseVal.x; // Sharpen ridges
            }
            n += noiseVal * na * vec3(1.0, nf, nf);
            na *= u_heightGain;
            nf *= u_heightLacunarity;
        }
    }
    
    if (u_ridgeNoise > 0) {
        n.x = n.x * 0.5; // Scale down ridge noise slightly
    } else {
        n.x = n.x * 0.5 + 0.5;
    }

    vec2 dir = n.zy * vec2(1.0, -1.0) * u_erosionSlopeStrength;
    vec3 h = vec3(0.0);
    float a = 0.5;
    float f = 1.0;
    a *= smoothstep(u_waterHeight - 0.1, u_waterHeight + 0.2, n.x);

    int octaves = useErosion > 0 ? u_erosionOctaves : 0;

    // Use seamless erosion if enabled
    if (u_seamless > 0) {
        for (int i = 0; i < 16; i++) {
            if (i >= octaves) break;
            vec2 currentDir = dir + h.zy * vec2(1.0, -1.0) * u_erosionBranchStrength;
            h += erosionSeamless(p * u_erosionTiles * f, currentDir, u_erosionTiles) * a * vec3(1.0, f, f);
            a *= u_erosionGain;
            f *= u_erosionLacunarity;
        }
    } else {
        for (int i = 0; i < 16; i++) {
            if (i >= octaves) break;
            vec2 currentDir = dir + h.zy * vec2(1.0, -1.0) * u_erosionBranchStrength;
            h += erosion(p * u_erosionTiles * f, currentDir) * a * vec3(1.0, f, f);
            a *= u_erosionGain;
            f *= u_erosionLacunarity;
        }
    }

    float finalHeight = n.x + (h.x - 0.5) * u_erosionStrength;
    return vec2(finalHeight, h.x);
}

void main() {
    vec2 heightData = generateHeightmap(uv);
    float height = heightData.x;

    float h_dx = generateHeightmap(uv + vec2(u_texelSize.x, 0.0)).x;
    float h_dy = generateHeightmap(uv + vec2(0.0, u_texelSize.y)).x;

    vec3 normal = normalize(cross(
        vec3(u_texelSize.x, 0.0, h_dx - height),
        vec3(0.0, u_texelSize.y, h_dy - height)
    ));

    // Store X/Z components of the normal; Y can be reconstructed in sampling shader
    fragColor = vec4(height, normal.x, normal.z, heightData.y);
}
