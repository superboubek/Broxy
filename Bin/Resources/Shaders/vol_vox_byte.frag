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

layout (binding = 0, r32ui) coherent uniform uimage3D vox_grid;

uniform uint res;

in PerVertex
{
  vec4 vGridSpacePosition;
} fs_data_in;

void main() {
	ivec3 final_grid_position;
 	ivec3 uGridSize = ivec3 (res); 
	vec3 clipSpacePosition = (fs_data_in.vGridSpacePosition.xyz + 1.0f) * 0.5f;
	final_grid_position = ivec3(clipSpacePosition * uGridSize);

	unsigned int val_vox = 0xffffffff;
	int num_shifts = 3 - (final_grid_position.z % 4);
	val_vox = (val_vox >> (8*num_shifts));
	final_grid_position.z = final_grid_position.z/4;	
	imageAtomicXor (vox_grid, final_grid_position, val_vox);

	for (int i = final_grid_position.z - 1; i >= 0; i--) {
		final_grid_position.z = i;	
		val_vox = 0xffffffff;
		imageAtomicXor (vox_grid, final_grid_position, val_vox);
	}
}
