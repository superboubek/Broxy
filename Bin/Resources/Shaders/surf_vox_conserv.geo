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

uniform float sq_ar_thresh;

out PerVertex
{
  vec4 vGridSpacePosition;
} gs_data_out;

out gl_PerVertex
{
  vec4 gl_Position;
};

float sq_aspect_ratio (vec3 p0, vec3 p1, vec3 p2) {

	vec3 d0 = p0 - p1;
	vec3 d1 = p1 - p2;
	// finds the max squared edge length
	float l2, maxl2 = dot (d0, d0);
	if ((l2=dot (d1, d1)) > maxl2)
		maxl2 = l2;
	// keep searching for the max squared edge length
	d1 = p2 - p0;
	if ((l2=dot (d1, d1)) > maxl2)
		maxl2 = l2;

	// squared area of the parallelogram spanned by d0 and d1
	vec3 cross_d0_d1 = cross (d0, d1);
	float a2 = dot (cross_d0_d1, cross_d0_d1);
	// the area of the triangle would be
	// sqrt(a2)/2 or length * height / 2
	// aspect ratio = length / height
	//              = length * length / (2*area)
	//              = length * length / sqrt(a2)

	// returns the length of the longest edge
	//         divided by its corresponding height
	//         ... squared

	float sq_ar = (maxl2 * maxl2) / a2;

	return sq_ar;
}

void main() {
	// Compute the dominant axis
	vec3 clipSpacePosition0;
	vec3 clipSpacePosition1;
	vec3 clipSpacePosition2;
	float dominantAxis;

	// Project on dominant plane
	vec3 p0 = gl_in[0].gl_Position.xyz;
	vec3 p1 = gl_in[1].gl_Position.xyz;
	vec3 p2 = gl_in[2].gl_Position.xyz;
	vec3 gridSpaceNormal = cross(p1 - p0,
															 p2 - p0);
	float sq_ar = sq_aspect_ratio (p0, p1, p2);

	float max_normal = max (abs(gridSpaceNormal.x), 
													max (abs(gridSpaceNormal.y), abs(gridSpaceNormal.z))
												 );

	if (abs(gridSpaceNormal.x) == max_normal) {
		clipSpacePosition0 = gl_in[0].gl_Position.yzx;
		clipSpacePosition1 = gl_in[1].gl_Position.yzx;
		clipSpacePosition2 = gl_in[2].gl_Position.yzx;
		dominantAxis = 0.5f;
	} else if (abs(gridSpaceNormal.y) == max_normal) {
		clipSpacePosition0 = gl_in[0].gl_Position.zxy;
		clipSpacePosition1 = gl_in[1].gl_Position.zxy;
		clipSpacePosition2 = gl_in[2].gl_Position.zxy;
		dominantAxis = 1.5f;
	} else {
		clipSpacePosition0 = gl_in[0].gl_Position.xyz;
		clipSpacePosition1 = gl_in[1].gl_Position.xyz;
		clipSpacePosition2 = gl_in[2].gl_Position.xyz;
		dominantAxis = 2.5f;
	}

	bool emit_geo = sq_ar < sq_ar_thresh;
	if (emit_geo) {
		// Emit geometry
		gl_Position = vec4(clipSpacePosition0.xyz, 1.0f);
		gs_data_out.vGridSpacePosition = vec4 (p0, dominantAxis);
		EmitVertex();
		gl_Position = vec4(clipSpacePosition1.xyz, 1.0f);
		gs_data_out.vGridSpacePosition = vec4 (p1, dominantAxis);
		EmitVertex();
		gl_Position = vec4(clipSpacePosition2.xyz, 1.0f);
		gs_data_out.vGridSpacePosition = vec4 (p2, dominantAxis);
		EmitVertex();
		EndPrimitive();
	}
}
