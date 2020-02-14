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

#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

out PerVertex
{
  vec4 vGridSpacePosition;
} gs_data_out;

out gl_PerVertex
{
  vec4 gl_Position;
};

void main() {
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;

	// Emit geometry
	gl_Position = vec4(p0, 1.0f);
	gs_data_out.vGridSpacePosition = vec4 (p0, 1.f);
	EmitVertex();
	gl_Position = vec4(p1, 1.0f);
	gs_data_out.vGridSpacePosition = vec4 (p1, 1.f);
	EmitVertex();
	gl_Position = vec4(p2, 1.0f);
	gs_data_out.vGridSpacePosition = vec4 (p2, 1.f);
	EmitVertex();
	EndPrimitive();
}
