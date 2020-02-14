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

uniform mat4 cameraViewMatrix;

struct Light { vec3 pos; vec3 color; };
Light L[3] = Light[](Light (vec3 (42.0, 374.0, 161.0), vec3 (1.0, 1.0, 1.0)),
                    Light (vec3 (473.0, -351.0, -259.0), vec3 (0.28, 0.39, 1.0)),
                    Light (vec3 (-438.0, 167.0, -48.0), vec3 (1.0, 0.69, 0.23)));

const float cageAlpha = 0.1f;
const vec4 upCol = vec4 (1.0, 0.98, 0.66, 1.0);
const vec4 downCol  = vec4 (0.01, 0.2, 0.55, 1.0);

const float diff = 0.6;
const float spec = 0.4;
const float shin = 7.0;

in vec4 P;
in vec3 N;
in vec4 C;


out vec4 radiance;

void main (void) {
     vec3 p = vec3 (cameraViewMatrix * P);
     vec3 n = normalize (N);
     vec3 v = normalize (-p);
     radiance = 0.25 * mix (upCol, downCol, (1.0+n.y)/2.0);
     for (int i = 0; i < 3; i++) {
        vec3 l = normalize (L[i].pos- p.xyz);
        float d   = max (dot (l, n), 0.0);
        vec3 r = reflect (-l, n);
        float s = pow (max(dot(r, v), 0.0), shin);
        radiance += diff * d * vec4 (L[i].color, 1.0) + spec * s * vec4 (0.8, 0.8, 0.8, 1.0);
     }
     radiance *= C;
	 radiance.a = cageAlpha;
}
 
