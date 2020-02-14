/*
    Copyright (c) 2015-2017 Telecom ParisTech (France).
    Authors: Stephane Calderon and Tamy Boubekeur.
    All rights reserved.

    This file is part of Broxy, the reference implementation for
    the paper:
        Bounding Proxies for Shape Approximation.
        Stephane Calderon and Tamy Boubekeur.
        ACM Transactions on Graphics (Proc. SIGGRAPH 2017),
        vol. 36, no. 5, art. 57, 2017.

    You can redistribute it and/or modify it under the terms of the GNU
    General Public License as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.

    Licensees holding a valid commercial license may use this file in
    accordance with the commercial license agreement provided with the software.

    This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
    WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*/


#version 140

const float PI = 3.14159265358979323846;
const float DEFAULT_NON_METALLIC_FRESNEL_REFLECTANCE = 0.03;

uniform mat4 cameraViewMatrix;

struct Light { vec3 pos; vec3 color; };
Light L[4] = Light[](Light (vec3 (42.0, 374.0, 161.0), vec3 (0.6, 0.6, 0.6)),
                    Light (vec3 (42.0, -74.0, 161.0), vec3 (0.4, 0.3, 0.4)),
                    Light (vec3 (473.0, -351.0, -259.0), vec3 (0.18, 0.29, 0.6)),
                    Light (vec3 (-438.0, 1.0, 0.0), vec3 (0.59, 0.39, 0.43)));

uniform float scaleAlpha;
uniform int cullMode;

const vec4 upCol = vec4 (1.0, 0.98, 0.66, 1.0);
const vec4 downCol  = vec4 (0.01, 0.2, 0.55, 1.0);

const vec3 albedo = vec3 (0.8, 0.8, 0.8);
float roughness = 0.72;
bool metallic = true;

float sqr(float x) { return x*x; }

// Lambert diffuse term
vec3 LambertBRDF (vec3 albedo) {
        return albedo/PI;
}

// Trowbridge-Reitz normal distribution term for the GGX BRDF
float GGXTR (float alpha, float NdotH) {
        float alphaSqr = sqr (alpha);
        return (alphaSqr / (PI * sqr (1.0 + (alphaSqr-1.0)*sqr (NdotH))));
}

// Schlick Fresnel term approximation
vec3 SchlickFresnel (vec3 FresnelReflectance, float VdotH) {
        return FresnelReflectance + (1.0 - FresnelReflectance) * pow (1.0 - VdotH, 5.0);
}

// Schlick approximation of the Beckmann geometry term match to GGX
float SchlickGeometry (float NdotV, float alpha) {
        float k = alpha/2.0;
        return NdotV/(NdotV*(1.0-k)+k);
}

// General Smith geometry term family
float SmithG (float NdotV, float NdotL, float alpha) {
        return SchlickGeometry (NdotV, alpha) * SchlickGeometry (NdotL, alpha);
}

// Ground Glass Reflection BRDF model
vec3 GGXBRDF(vec3 L, vec3 V, vec3 N, vec3 albedo, float alpha, bool metallic) {
        vec3 H = normalize( L + V );
        float NdotL = dot(N, L);
        float NdotV = dot(N, V);
        float NdotH = dot(N, H);
        float VdotH = dot (V, H);
        float D = GGXTR (alpha, NdotH);
        vec3 F = SchlickFresnel (metallic ? albedo : vec3 (DEFAULT_NON_METALLIC_FRESNEL_REFLECTANCE), VdotH);
        float G = SmithG (NdotL, NdotV, alpha);
        return D*F*G / (4.0 * NdotL * NdotV);
}

// Physically based reflectance model summing Lambert diffuse and GGX specular models.
vec3 BRDF (vec3 L, vec3 V, vec3 N, vec3 albedo, float roughness, bool metallic) {
        float alpha = sqr (roughness); // Unreal/Disney reparametrization
        return (LambertBRDF (albedo) + GGXBRDF (L, V, N, albedo, alpha, metallic));
}

const float diff = 0.6;
const float spec = 0.4;
const float shin = 10.0;

in vec4 P;
in vec3 N;
in vec4 C;

out vec4 radiance;

void main (void) {
     vec3 p = vec3 (cameraViewMatrix * P);
     vec3 n = normalize (N);
     vec3 v = normalize (-p);
//     radiance = vec4 (0.0, 0.0, 0.0, 1.0);//0.25 * mix (upCol, downCol, (1.0+n.y)/2.0);
//     for (int i = 0; i < 3; i++) {
//        vec3 l = normalize (L[i].pos- p.xyz);
//        radiance.rgb += L[i].color * PI * BRDF (l, v, n, albedo, roughness, metallic) * max (dot (n, l), 0.0) ;
//     }
     radiance = 0.25 * mix (upCol, downCol, (1.0+n.y)/2.0);
     for (int i = 0; i < 3; i++) {
        vec3 l = normalize (L[i].pos- p.xyz);
        float d   = max (dot (l, n), 0.0);
        vec3 r = reflect (-l, n);
        float s = pow (max(dot(r, v), 0.0), shin);
        radiance += diff * d * vec4 (L[i].color, 1.0) + spec * s * vec4 (0.8, 0.8, 0.8, 1.0);
     }

		 radiance *= C;
		 if (cullMode == 0 && dot (n, v) < 0)
			 radiance.a = 0.f;
		 else if (cullMode == 1 && dot (n, v) > 0)
			 radiance.a = 0.f;
		 else
			 radiance.a = scaleAlpha;
}

