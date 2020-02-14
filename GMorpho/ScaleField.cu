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

#include "ScaleField.h"

#include <stdio.h>
#include "timing.h"
#include "cuda_math.h"

using namespace MorphoGraphics;

template<typename T>
void ScaleField::FreeGPUResource (T ** res) {
	if (*res != NULL) {
		cudaFree (*res);
		*res = NULL;
	} 
}

void ScaleField::print (const std::string & msg) {
	std::cout << "[ScaleField] : " << msg << std::endl;
}

void ScaleField::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if(err != cudaSuccess) {
		ScaleField::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw ScaleField::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

void ScaleField::ShowGPUMemoryUsage () {
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);
	used = total - avail;
	std::cout << "[ScaleField] : Device memory used: " << (float)used/1e6 
		<< " of " << (float)total/1e6 << std::endl;
}


void ScaleField::Init (const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
										 const MorphoGraphics::Vec3<unsigned int> & res, float cell_size) {
	bbox_min_ = bbox_min;
	bbox_max_ = bbox_max;
	res_ = res;
	cell_size_ = cell_size;

	if (scale_grid_ == NULL)
		cudaMalloc (&scale_grid_, res_[0]*res_[1]*res_[2]*sizeof (char));

	cudaMemset (scale_grid_, 0, res_[0]*res_[1]*res_[2]*sizeof (char));
}

void ScaleField::GetScaleGridCPU (char ** scale_grid_cpu_ptr) { 
	*scale_grid_cpu_ptr = new char [res_[0]*res_[1]*res_[2]];
	cudaMemcpy (*scale_grid_cpu_ptr, scale_grid_, 
							res_[0]*res_[1]*res_[2]*sizeof (char), 
							cudaMemcpyDeviceToHost);
}

__global__ void UpdateScaleGrid (char * scale_grid, 
																 unsigned int width, 
																 unsigned int height, 
																 unsigned int depth, 
																 float3 bbox_min, 
																 float3 bbox_max, 
																 float cell_size, 
																 float3 scale_point_position,
																 float scale_point_scale, 
																 float scale_point_support, 
																 float global_scale, 
																 EditMode scale_point_edit_mode) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x; 
	int id_y = blockIdx.y * blockDim.y + threadIdx.y; 
	int id_z = blockIdx.z * blockDim.z + threadIdx.z; 

	if (id_x >= width || id_y >= height || id_z >= depth)
		return;

	unsigned int i = height*height*id_z + width*id_y + id_x;

	// Fill successives scales
	float3 u;
	u.x = (float)(id_x);
	u.y = (float)(id_y);
	u.z = (float)(id_z);
	u = cell_size*u + bbox_min;

	float3 pl = scale_point_position;
	float scale = scale_point_scale;
	float v = scale;
	float h = (scale_point_support - v)/(sqrt (1 - sqrt (0.5f)));
	float dist_u_pl = sqrt (distanceS (u, pl));
	float distrib = tukey (dist_u_pl, h, v);	
	float scale_prev = scale_grid[i];
	//	if (id_x == 0 && id_y == 0 && id_z == 0) {
	//		printf ("h : %f\n", h);
	//	}
	if (dist_u_pl < (v + h))
		if (scale_point_edit_mode == MAX_BRUSH) {
			distrib = scale_prev*(1.f - distrib) + distrib*(scale/cell_size);
			//			distrib = max (distrib, global_scale);
			//			distrib = floor (distrib/cell_size);
			distrib = floor (distrib);
			distrib = max (min (distrib, 127.f), 0.f);
			scale_grid[i] = floor (max (distrib, scale_prev));
		} else if (scale_point_edit_mode == MIN_BRUSH) {
			//			distrib = (scale - global_scale)*distrib + global_scale;
			//			distrib = min (distrib, global_scale);
			//			distrib = floor (distrib/cell_size);
			distrib = scale_prev*(1.f - distrib) + distrib*(scale/cell_size);
			distrib = floor (distrib);
			distrib = max (min (distrib, 127.f), 0.f);
			scale_grid[i] = floor (min (distrib, scale_prev));
		} else if (scale_point_edit_mode == ZERO_BRUSH) {
			distrib = scale_prev*(1.f - distrib) + distrib*(global_scale/cell_size);
			distrib = floor (distrib);
			distrib = max (min (distrib, 127.f), 0.f);
			scale_grid[i] = distrib;
		}
}

void ScaleField::UpdateGridGlobalScale () {
	cudaMemset (scale_grid_, floor (global_scale_/cell_size_), 
							res_[0]*res_[1]*res_[2]*sizeof (char));
}

void ScaleField::UpdateGrid () {
	double time1, time2;
	size_t width = res_[0];
	size_t height = res_[1];
	size_t depth = res_[2];

	// Run fill kernel
	dim3 block_dim, grid_dim;
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((width/block_dim.x)+1, 
									 (height/block_dim.y)+1, 
									 (depth/block_dim.z)+1);
	float3 cu_bbox_min = make_float3 (bbox_min_[0], bbox_min_[1], bbox_min_[2]);
	float3 cu_bbox_max = make_float3 (bbox_max_[0], bbox_max_[1], bbox_max_[2]);

	if (points_.size () != 0) {
		MorphoGraphics::Vec3f last_position = points_[points_.size () - 1].position ();
		float last_scale = points_[points_.size () - 1].scale ();
		float last_support = points_[points_.size () - 1].support ();
		EditMode last_edit_mode = points_[points_.size () - 1].edit_mode ();

		float3 cu_scale_point_position = make_float3 (last_position[0], 
																									last_position[1], 
																									last_position[2]);
		float scale_point_scale = last_scale;
		float scale_point_support = last_support;
		EditMode scale_point_edit_mode = last_edit_mode;

		//	float * cu_scale_set = NULL;
		//	cudaMalloc (&cu_scale_set, 4*sizeof (float)*points_.size ());
		//	cudaMemcpy (cu_scale_set, &(points_[0]), 
		//							4*sizeof (float)*points_.size (), 
		//							cudaMemcpyHostToDevice);

		time1 = GET_TIME ();
		UpdateScaleGrid<<<grid_dim, block_dim>>> 
			(scale_grid_, width, height, depth, 
			 cu_bbox_min, 
			 cu_bbox_max, 
			 cell_size_, 
			 cu_scale_point_position, 
			 scale_point_scale, 
			 scale_point_support, 
			 global_scale_, 
			 scale_point_edit_mode);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[ScaleField] : " << "Scale Grid updated in " 
			<< time2 - time1 << " ms. with se_size : " << floor (2.f*global_scale_/cell_size_) 
			<< " points size : " << points_.size ()
			<< std::endl;
		//	FreeGPUResource (&cu_scale_set);
	}
}
