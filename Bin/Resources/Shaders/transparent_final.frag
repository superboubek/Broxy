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
#version 420

layout (binding = 0) uniform sampler2D accumTex;
layout (binding = 1) uniform sampler2D revealTex;
layout (binding = 2) uniform sampler2D opaqueTex;
uniform vec2 screenSize;

uniform float step;

out vec4 composite;

void main (void) {
  vec2 uv = gl_FragCoord.xy/screenSize;
  vec4 opaque = texture2D (opaqueTex, uv);
  vec4 accum = texture2D (accumTex, uv);
  vec4 reveal = texture2D (revealTex, uv);
	vec4 transp = vec4 (accum.rgb/clamp (accum.a, 1e-4, 5e4), reveal.r);
	composite = step*vec4 ((1.f - reveal.r)*transp.rgb + reveal.r*opaque.rgb, 1.f);
}
