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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <curand.h>

#include "MarchingCubesMesher.h"
#include "Grid.h"

using namespace MorphoGraphics;

template<typename T>
void Grid::FreeGPUResource (T ** res) {
	if (*res != NULL) {
		cudaFree (*res);
		*res = NULL;
	}
}

void Grid::print (const std::string & msg) {
	std::cout << "[Grid] : " << msg << std::endl;
}

void Grid::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if (err != cudaSuccess) {
		Grid::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw Grid::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

void Grid::ShowGPUMemoryUsage () {
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);
	used = total - avail;
	std::cout << "[Grid] : Device memory used: " << (float)used / 1e6
	          << " of " << (float)total / 1e6 << std::endl;
}

__global__ void ConvertVoxelGridByBit (unsigned char * voxelGrid, uint3 res,
                                       float * voxelGrid2x2x2) {
	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (idX >= (2 * res.x - 1) || idY >= (2 * res.y - 1) || idZ >= (2 * res.z - 1))
		return;

	unsigned char val;

	val = voxelGrid[(idX / 2) + res.x * (idY / 2) + res.x * res.y * (idZ / 2)] & (1 << (idX % 2 + (idY % 2) * 2 + (idZ % 2) * 4));

	voxelGrid2x2x2[idX + 2 * res.x * idY + 4 * res.x * res.y * idZ] = val ? 0.f : 1.f;
}

__global__ void ConvertVoxelGridByBitUCHAR (unsigned char * voxelGrid, uint3 res,
        unsigned char * voxelGrid2x2x2) {
	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (idX >= (2 * res.x - 1) || idY >= (2 * res.y - 1) || idZ >= (2 * res.z - 1))
		return;

	unsigned char val;

	val = voxelGrid[(idX / 2) + res.x * (idY / 2) + res.x * res.y * (idZ / 2)] & (1 << (idX % 2 + (idY % 2) * 2 + (idZ % 2) * 4));

	voxelGrid2x2x2[idX + 2 * res.x * idY + 4 * res.x * res.y * idZ] = val ? 0 : 255;
}

Grid::Grid () {
	grid_gpu_.voxels = NULL;
	for (int k = 0; k < MAX_MIPMAP_DEPTH; k++)
		grid_gpu_.mipmap[k] = NULL;

	grid_gpu_.mipmapped_array = NULL;
}

Grid::~Grid () {

}

void Grid::Init (const BoundingBox & bbox,
                 const Vec3ui & res, const Vec3ui & data_res,
                 float cell_size) {
	// CPU member initialization
	bbox_ = bbox;
	res_ = res;
	data_res_ = data_res;
	cell_size_ = cell_size;

	// GPU member initialization
	grid_gpu_.bbox_min = make_float3 (bbox_.min()[0],
	                                  bbox_.min()[1],
	                                  bbox_.min()[2]);
	grid_gpu_.bbox_max = make_float3 (bbox_.max()[0],
	                                  bbox_.max()[1],
	                                  bbox_.max()[2]);
	grid_gpu_.res = make_uint3 (res_[0], res_[1], res_[2]);
	grid_gpu_.data_res = make_uint3 (data_res_[0], data_res_[1], data_res_[2]);
	grid_gpu_.cell_size = cell_size;

	// GPU allocation
	int num_cells = res_[0] * res_[1] * res_[2];
	cudaMalloc (&grid_gpu_.voxels, num_cells * sizeof (unsigned char));
	CheckCUDAError ();
}

void Grid::SetGridValue (unsigned char value) {
	int num_cells = res_[0] * res_[1] * res_[2];
	cudaMemset (grid_gpu_.voxels, value, num_cells * sizeof (unsigned char));
	CheckCUDAError ();
}

void Grid::CopyVoxelsFrom (const Grid & grid) {
	int num_cells_dest = res_[0] * res_[1] * res_[2];
	int num_cells_src = grid.res ()[0] * grid.res ()[1] * grid.res ()[2];
	if (num_cells_dest == num_cells_src) {
		cudaMemcpy (grid_gpu_.voxels, grid.grid_gpu ().voxels,
		            num_cells_dest * sizeof (unsigned char),
		            cudaMemcpyDeviceToDevice);
		CheckCUDAError ();
	} else {
		std::cout << "[Grid] : "
		          << "impossible copy : size don't match" << std::endl;
	}
}

__global__ void BuildMipmapUpperLevel (unsigned char * upper_level,
                                       unsigned char * lower_level,
                                       unsigned int upper_res) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((id_x >= upper_res) || (id_y >= upper_res) || (id_z >= upper_res))
		return;

	unsigned int i = upper_res * upper_res * id_z + upper_res * id_y + id_x;

	unsigned int lower_res = 2 * upper_res;
	unsigned int sq_lower_res = lower_res * lower_res;
	id_x *= 2; id_y *= 2; id_z *= 2;

	unsigned char val[8];
	val[0] = lower_level[sq_lower_res * id_z + lower_res * id_y + id_x];
	val[1] = lower_level[sq_lower_res * id_z + lower_res * id_y + id_x + 1];
	val[2] = lower_level[sq_lower_res * id_z + lower_res * (id_y + 1) + id_x];
	val[3] = lower_level[sq_lower_res * id_z + lower_res * (id_y + 1) + id_x + 1];
	val[4] = lower_level[sq_lower_res * (id_z + 1) + lower_res * id_y + id_x];
	val[5] = lower_level[sq_lower_res * (id_z + 1) + lower_res * id_y + id_x + 1];
	val[6] = lower_level[sq_lower_res * (id_z + 1) + lower_res * (id_y + 1) + id_x];
	val[7] = lower_level[sq_lower_res * (id_z + 1) + lower_res * (id_y + 1) + id_x + 1];

	unsigned char upper_val = 0x00;
	upper_val |= val[0] != 0x00 ? C0 : 0x00;
	upper_val |= val[1] != 0x00 ? C1 : 0x00;
	upper_val |= val[2] != 0x00 ? C2 : 0x00;
	upper_val |= val[3] != 0x00 ? C3 : 0x00;
	upper_val |= val[4] != 0x00 ? C4 : 0x00;
	upper_val |= val[5] != 0x00 ? C5 : 0x00;
	upper_val |= val[6] != 0x00 ? C6 : 0x00;
	upper_val |= val[7] != 0x00 ? C7 : 0x00;

	upper_level[i] = upper_val;
}

__global__ void BuildMortonMipmap (unsigned char * morton_mipmap,
                                   unsigned char * linear_mipmap,
                                   unsigned int res) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	if ((id_x >= res) || (id_y >= res) || (id_z >= res))
		return;

	unsigned int morton = EncodeMorton3 (id_x, id_y, id_z);
	morton_mipmap[morton] = linear_mipmap[res * res * id_z + res * id_y + id_x];
}

__global__ void BuildExclusiveMipmapUpperLevel (unsigned char * upper_level,
        unsigned char * lower_level,
        unsigned int upper_res) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if ((id_x >= upper_res) || (id_y >= upper_res) || (id_z >= upper_res))
		return;

	unsigned int i = upper_res * upper_res * id_z + upper_res * id_y + id_x;

	unsigned int lower_res = 2 * upper_res;
	unsigned int sq_lower_res = lower_res * lower_res;
	id_x *= 2; id_y *= 2; id_z *= 2;

	unsigned char val[8];
	val[0] = lower_level[sq_lower_res * id_z + lower_res * id_y + id_x];
	val[1] = lower_level[sq_lower_res * id_z + lower_res * id_y + id_x + 1];
	val[2] = lower_level[sq_lower_res * id_z + lower_res * (id_y + 1) + id_x];
	val[3] = lower_level[sq_lower_res * id_z + lower_res * (id_y + 1) + id_x + 1];
	val[4] = lower_level[sq_lower_res * (id_z + 1) + lower_res * id_y + id_x];
	val[5] = lower_level[sq_lower_res * (id_z + 1) + lower_res * id_y + id_x + 1];
	val[6] = lower_level[sq_lower_res * (id_z + 1) + lower_res * (id_y + 1) + id_x];
	val[7] = lower_level[sq_lower_res * (id_z + 1) + lower_res * (id_y + 1) + id_x + 1];

	unsigned char upper_val = 0x00;
	upper_val |= val[0] == 0xff ? C0 : 0x00;
	upper_val |= val[1] == 0xff ? C1 : 0x00;
	upper_val |= val[2] == 0xff ? C2 : 0x00;
	upper_val |= val[3] == 0xff ? C3 : 0x00;
	upper_val |= val[4] == 0xff ? C4 : 0x00;
	upper_val |= val[5] == 0xff ? C5 : 0x00;
	upper_val |= val[6] == 0xff ? C6 : 0x00;
	upper_val |= val[7] == 0xff ? C7 : 0x00;

	upper_level[i] = upper_val;
}

void Grid::BuildMipmaps () {
	double time1, time2;
	int max_mipmap_depth = log2 ((float) res_[0]);

	// Mipmap GPU Allocation
	// The mipmap has always one more level than the base res since
	// 8 cells of 1 bit are encoded as chars ...
	max_mipmap_depth++;
	grid_gpu_.max_mipmap_depth = max_mipmap_depth;

	// Determine the overall size allocation
	// Note: the level 0 is encoded by a unsigned char ... instead of 1 bit
	int overall_num_cells = 1 + ((1 << 3 * max_mipmap_depth) - 1) / 7;
	cudaMalloc (&grid_gpu_.mipmap[0],
	            overall_num_cells * sizeof (unsigned char));
	CheckCUDAError ();
	cudaMalloc (&grid_gpu_.morton_mipmap[0],
	            overall_num_cells * sizeof (unsigned char));
	CheckCUDAError ();
	cudaMalloc (&grid_gpu_.ex_mipmap[0],
	            overall_num_cells * sizeof (unsigned char));
	CheckCUDAError ();
	cudaMalloc (&grid_gpu_.ex_morton_mipmap[0],
	            overall_num_cells * sizeof (unsigned char));
	CheckCUDAError ();
	std::cout << "[Grid] : " << overall_num_cells
	          << " overall cell number" << std::endl;

	int mipmap_res = 1;
	overall_num_cells = 1;
	grid_gpu_.mipmap[1] = grid_gpu_.mipmap[0] + 1;
	grid_gpu_.morton_mipmap[1] = grid_gpu_.morton_mipmap[0] + 1;
	grid_gpu_.ex_mipmap[1] = grid_gpu_.ex_mipmap[0] + 1;
	grid_gpu_.ex_morton_mipmap[1] = grid_gpu_.ex_morton_mipmap[0] + 1;
	for (int k = 1; k < max_mipmap_depth; k++) {
		int num_cells = mipmap_res * mipmap_res * mipmap_res;
		unsigned int num_cells_offset = ((1 << (3 * k)) - 1) / 7;
		overall_num_cells += num_cells;
		std::cout << "[Grid] : " << "mipmap level " << k << " is coded with "
		          << mipmap_res << "x" << mipmap_res << "x" << mipmap_res
		          << " unsigned chars " << overall_num_cells << " " << num_cells_offset << std::endl;
		grid_gpu_.mipmap[k + 1] = grid_gpu_.mipmap[1]
		                          + num_cells_offset;
		grid_gpu_.morton_mipmap[k + 1] = grid_gpu_.morton_mipmap[1]
		                                 + num_cells_offset;
		grid_gpu_.ex_mipmap[k + 1] = grid_gpu_.ex_mipmap[1]
		                             + num_cells_offset;
		grid_gpu_.ex_morton_mipmap[k + 1] = grid_gpu_.ex_morton_mipmap[1]
		                                    + num_cells_offset;
		mipmap_res *= 2;
	}
	std::cout << "[Grid] : " << overall_num_cells
	          << " overall cell number" << std::endl;

	dim3 block_dim, grid_dim;
	int upper_mipmap_res = res_[0] / 2;
	int lower_mipmap_res = res_[0];
	block_dim = dim3 (32, 4, 4);

	// The last level of the mipmap is the voxels_ itself
	std::cout << "max_mipmap_depth : " << max_mipmap_depth << std::endl;
	cudaMemcpy (grid_gpu_.mipmap[max_mipmap_depth], grid_gpu_.voxels,
	            pow3i (lower_mipmap_res)*sizeof (unsigned char),
	            cudaMemcpyDeviceToDevice);
	// The last level of the exclusive mipmap is the voxels_ itself
	cudaMemcpy (grid_gpu_.ex_mipmap[max_mipmap_depth], grid_gpu_.voxels,
	            pow3i (lower_mipmap_res)*sizeof (unsigned char),
	            cudaMemcpyDeviceToDevice);

	// Build from deepest level
	for (int k = 0; k < max_mipmap_depth; k++) {
		if (k < (max_mipmap_depth - 1)) {
			grid_dim = dim3 ((upper_mipmap_res / block_dim.x) + 1,
			                 (upper_mipmap_res / block_dim.y) + 1,
			                 (upper_mipmap_res / block_dim.z) + 1);
			time1 = GET_TIME ();
			BuildMipmapUpperLevel <<< grid_dim, block_dim>>>
			(grid_gpu_.mipmap[max_mipmap_depth - k - 1],
			 grid_gpu_.mipmap[max_mipmap_depth - k], upper_mipmap_res);
			cudaDeviceSynchronize ();
			CheckCUDAError ();
			time2 = GET_TIME ();
			std::cout << "[Grid] : " << max_mipmap_depth - k - 1
			          << "th level built in " << time2 - time1 << " ms." << std::endl;

			time1 = GET_TIME ();
			BuildExclusiveMipmapUpperLevel <<< grid_dim, block_dim>>>
			(grid_gpu_.ex_mipmap[max_mipmap_depth - k - 1],
			 grid_gpu_.ex_mipmap[max_mipmap_depth - k], upper_mipmap_res);
			cudaDeviceSynchronize ();
			CheckCUDAError ();
			time2 = GET_TIME ();
			std::cout << "[Grid] : " << max_mipmap_depth - k - 1
			          << "th exclusive level built in " << time2 - time1
			          << " ms." << std::endl;
		}
		grid_dim = dim3 ((lower_mipmap_res / block_dim.x) + 1,
		                 (lower_mipmap_res / block_dim.y) + 1,
		                 (lower_mipmap_res / block_dim.z) + 1);
		time1 = GET_TIME ();
		BuildMortonMipmap <<< grid_dim, block_dim>>>
		(grid_gpu_.morton_mipmap[max_mipmap_depth - k],
		 grid_gpu_.mipmap[max_mipmap_depth - k], lower_mipmap_res);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[Grid] : " << max_mipmap_depth - k
		          << "th morton mipmap built in " << time2 - time1 << " ms." << std::endl;

		time1 = GET_TIME ();
		BuildMortonMipmap <<< grid_dim, block_dim>>>
		(grid_gpu_.ex_morton_mipmap[max_mipmap_depth - k],
		 grid_gpu_.ex_mipmap[max_mipmap_depth - k], lower_mipmap_res);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[Grid] : " << max_mipmap_depth - k
		          << "th exclusive morton mipmap built in " << time2 - time1
		          << " ms." << std::endl;

		upper_mipmap_res /= 2; lower_mipmap_res /= 2;
	}
	std::cout << "[Grid] : " << "there are "
	          << max_mipmap_depth << " mipmap levels" << std::endl;

	ShowGPUMemoryUsage ();
	//	MipmapMeshing2x2x2 (max_mipmap_depth);
}

__global__ void FillDualLevel0 (const unsigned char * voxels,
                                unsigned int width,
                                unsigned int height,
                                unsigned int depth,
                                unsigned char se_size,
                                cudaSurfaceObject_t level0) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (id_x >= width || id_y >= height || id_z >= depth)
		return;

	unsigned int i = height * height * id_z + width * id_y + id_x;

	ulonglong2 vox_pack = make_ulonglong2 (0, 0);
	unsigned long long occup = voxels[i];

	// Fill primal occup first
	vox_pack.y = occup;

	unsigned char uchar_radius;

	// Fill primal successives scales
	for (int k = 0; k < 8; k++) {
		vox_pack.y = vox_pack.y << 7;
		uchar_radius = 20;
		vox_pack.y |= (uchar_radius & (0x7f));
	}

	// Fill dual occup first
	vox_pack.x |= occup;

	// Fill dual successives scales
	for (int k = 0; k < 8; k++) {
		vox_pack.x = vox_pack.x << 7;
		uchar_radius = 20;
		vox_pack.x |= (uchar_radius & (0x7f));
	}

	surf3Dwrite (vox_pack, level0, id_x * sizeof (ulonglong2), id_y, id_z);
}

__global__ void BuildDualMipmapFromTo (cudaTextureObject_t level_from,
                                       unsigned int to_width,
                                       unsigned int to_height,
                                       unsigned int to_depth,
                                       cudaSurfaceObject_t level_to) {
	int to_id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int to_id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int to_id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (to_id_x >= to_width || to_id_y >= to_height || to_id_z >= to_depth)
		return;

	unsigned int from_width = 2 * to_width;
	unsigned int from_height = 2 * to_height;
	unsigned int from_depth = 2 * to_depth;
	int from_id_x = 2 * to_id_x;
	int from_id_y = 2 * to_id_y;
	int from_id_z = 2 * to_id_z;

	float px = 1.f / float (from_width);
	float py = 1.f / float (from_height);
	float pz = 1.f / float (from_depth);

	int4 vals[8];
	vals[0] = tex3D<int4> (level_from,
	                       from_id_x * px, from_id_y * py, from_id_z * pz);
	vals[1] = tex3D<int4> (level_from,
	                       (from_id_x + 1) * px, from_id_y * py, from_id_z * pz);
	vals[2] = tex3D<int4> (level_from,
	                       from_id_x * px, (from_id_y + 1) * py, from_id_z * pz);
	vals[3] = tex3D<int4> (level_from,
	                       (from_id_x + 1) * px, (from_id_y + 1) * py, from_id_z * pz);
	vals[4] = tex3D<int4> (level_from,
	                       from_id_x * px, from_id_y * py, (from_id_z + 1) * pz);
	vals[5] = tex3D<int4> (level_from,
	                       (from_id_x + 1) * px, from_id_y * py, (from_id_z + 1) * pz);
	vals[6] = tex3D<int4> (level_from,
	                       from_id_x * px, (from_id_y + 1) * py, (from_id_z + 1) * pz);
	vals[7] = tex3D<int4> (level_from,
	                       (from_id_x + 1) * px, (from_id_y + 1) * py, (from_id_z + 1) * pz);


	unsigned char mask_scale = 0x7f;
	ulonglong2 vox_pack_in;
	ulonglong2 vox_pack_out = make_ulonglong2 (0, 0);
	ulonglong2 scale_out = make_ulonglong2 (0, 0);
	ulonglong2 occup_out = make_ulonglong2 (0, 0);
	// Fill successives scales and occupation
	for (int l = 7; l >= 0; l--) {
		unsigned char max_scale = 0;
		int4 val = vals[l];
		vox_pack_in = *reinterpret_cast<ulonglong2*>(&val);

		if (from_id_x == 128 && from_id_y == 128 && from_id_z == 0) {
			printf ("x%.16llx x%.16llx l : %i\n",
			        vox_pack_in.y, vox_pack_in.x, l);
		}

		for (int k = 0; k < 8; k++) {
			unsigned char curr_scale = mask_scale & vox_pack_in.y;
			max_scale = curr_scale > max_scale ? curr_scale : max_scale;
			vox_pack_in.y = vox_pack_in.y >> 7;
			vox_pack_in.x = vox_pack_in.x >> 7;
		}
		scale_out.y = scale_out.y << 7;
		scale_out.y |= (max_scale & (0x7f));

		occup_out.y = occup_out.y << 1;
		occup_out.y |= (vox_pack_in.y != 0) ? 0x01 : 0;

		occup_out.x = occup_out.x << 1;
		occup_out.x |= (vox_pack_in.x == 0xff) ? 0x01 : 0;
	}
	occup_out.y = occup_out.y << 56;
	occup_out.x = occup_out.x << 56;

	vox_pack_out.y = scale_out.y;
	vox_pack_out.y |= occup_out.y;

	vox_pack_out.x = scale_out.x;
	vox_pack_out.x |= occup_out.x;

	ulonglong2 to_val = make_ulonglong2 (0, 0);
	to_val.y = vox_pack_out.y;

	if (from_id_x == 128 && from_id_y == 128 && from_id_z == 0) {
		printf ("x%.16llx x%.16llx\n",
		        to_val.y, to_val.x);
	}

	surf3Dwrite (to_val, level_to,
	             to_id_x * sizeof (ulonglong2), to_id_y, to_id_z);
}

void Grid::BuildTexDualMipmaps (float se_size) {
	double time1, time2;
	ShowGPUMemoryUsage ();

	// Allocate the mipmap : occupation is coded on 1bit and scale on 7bits,
	// giving (1+7)*8 = 64bits for one pack of voxels/cells
	size_t width = res_[0];
	size_t height = res_[1];
	size_t depth = res_[2];
	cudaChannelFormatDesc mipmap_channel_desc;
	mipmap_channel_desc = cudaCreateChannelDesc (32, 32, 32, 32,
	                      cudaChannelFormatKindUnsigned);
	cudaExtent cu_mipmap_extent = make_cudaExtent (width, height, depth);

	cudaMipmappedArray_t cu_mipmap_array;
	cudaMallocMipmappedArray (&cu_mipmap_array,
	                          &mipmap_channel_desc, cu_mipmap_extent, 9);

	// Fill Level-0
	cudaArray_t level_first;
	cudaGetMipmappedArrayLevel(&level_first, cu_mipmap_array, 0);

	// Generate surface object for writing
	cudaSurfaceObject_t surf_output0;
	cudaResourceDesc surf_res0;
	memset (&surf_res0, 0, sizeof(cudaResourceDesc));
	surf_res0.resType = cudaResourceTypeArray;
	surf_res0.res.array.array = level_first;
	cudaCreateSurfaceObject (&surf_output0, &surf_res0);
	CheckCUDAError ();

	// Run fill kernel
	dim3 block_dim, grid_dim;
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((width / block_dim.x) + 1,
	                 (height / block_dim.y) + 1,
	                 (depth / block_dim.z) + 1);
	time1 = GET_TIME ();
	FillDualLevel0 <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, width, height, depth,
	 floor (2.f * se_size / cell_size_),
	 surf_output0);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[BuildTexDualMipmaps] : " << "level-0 filled in "
	          << time2 - time1 << " ms." << std::endl;

	cudaDestroySurfaceObject(surf_output0);

	uint level = 0;

	while (width != 1) {
		width /= 2; height /= 2; depth /= 2;
		cudaArray_t level_from;
		cudaGetMipmappedArrayLevel(&level_from, cu_mipmap_array, level);
		cudaArray_t level_to;
		cudaGetMipmappedArrayLevel(&level_to, cu_mipmap_array, level + 1);

		cudaExtent level_to_size;
		cudaArrayGetInfo (NULL, &level_to_size, NULL, level_to);

		// Generate texture object for reading
		cudaTextureObject_t tex_input;

		cudaResourceDesc tex_res;
		memset (&tex_res, 0, sizeof(cudaResourceDesc));
		tex_res.resType = cudaResourceTypeArray;
		tex_res.res.array.array = level_from;

		cudaTextureDesc tex_descr;
		memset (&tex_descr, 0, sizeof(cudaTextureDesc));
		tex_descr.normalizedCoords = 1;
		tex_descr.filterMode = cudaFilterModePoint;
		tex_descr.addressMode[0] = cudaAddressModeClamp;
		tex_descr.addressMode[1] = cudaAddressModeClamp;
		tex_descr.addressMode[2] = cudaAddressModeClamp;
		tex_descr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject (&tex_input, &tex_res, &tex_descr, NULL);

		// Generate surface object for writing
		cudaSurfaceObject_t surf_output;
		cudaResourceDesc surf_res;
		memset (&surf_res, 0, sizeof(cudaResourceDesc));
		surf_res.resType = cudaResourceTypeArray;
		surf_res.res.array.array = level_to;
		cudaCreateSurfaceObject(&surf_output, &surf_res);

		block_dim = dim3 (8, 4, 4);
		grid_dim = dim3 ((width / block_dim.x) + 1,
		                 (height / block_dim.y) + 1,
		                 (depth / block_dim.z) + 1);
		time1 = GET_TIME ();
		BuildDualMipmapFromTo <<< grid_dim, block_dim>>>
		(tex_input, width, height, depth, surf_output);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[BuildTexDualMipmaps] : "
		          << "level-" << level + 1 << " computed in "
		          << time2 - time1 << " ms." << std::endl;


		cudaDestroySurfaceObject(surf_output);
		cudaDestroyTextureObject(tex_input);
		level++;
	}

	// Create texture object descriptors
	cudaResourceDesc res_desc;
	memset (&res_desc, 0, sizeof (res_desc));
	res_desc.resType = cudaResourceTypeMipmappedArray;
	res_desc.res.mipmap.mipmap = cu_mipmap_array;

	cudaTextureDesc tex_desc;
	memset (&tex_desc, 0, sizeof (tex_desc));
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;
	tex_desc.normalizedCoords = false;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.mipmapFilterMode = cudaFilterModePoint;
	tex_desc.minMipmapLevelClamp = 0;
	tex_desc.maxMipmapLevelClamp = grid_gpu_.max_mipmap_depth - 1;

	// Create texture object: we only have to do this once!
	cudaCreateTextureObject (&grid_gpu_.tex_dual_mipmap,
	                         &res_desc, &tex_desc, NULL);
	CheckCUDAError ();
	ShowGPUMemoryUsage ();
}

__global__ void FillLevel0 (const unsigned char * voxels,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            unsigned char se_size,
                            cudaSurfaceObject_t level0) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (id_x >= width || id_y >= height || id_z >= depth)
		return;

	unsigned int i = height * height * id_z + width * id_y + id_x;

	unsigned long long vox_pack = 0;

	// Fill occup first
	vox_pack = voxels[i];

	unsigned char uchar_radius;
	int res = width;
	int fine_id_x, fine_id_y, fine_id_z;
	float radius, radius0, radius1, radius2;

	// Fill successives scales
	float cx0 = -0.22f;
	float cy0 = 0.24f;
	float cz0 = 0.26f;
	float sigmax0 = 0.03;
	float sigmay0 = 0.08;
	float sigmaz0 = 0.08;
	float theta0 = 22.f * M_PI / 180.f;

	float cx1 = -0.12f;
	float cy1 = 0.27f;
	float cz1 = 0.265f;
	float sigmax1 = 0.02;
	float sigmay1 = 0.09;
	float sigmaz1 = 0.1;
	float theta1 = 5.f * M_PI / 180.f;

	float cx2 = -0.31f;
	float cy2 = 0.16f;
	float sigmax2 = 0.02;
	float sigmay2 = 0.04;
	float theta2 = 25.f * M_PI / 180.f;

	float ux, uy, uz;
	for (int k = 7; k >= 0; k--) {
		unsigned int mask_k = (1 << k);
		fine_id_x = 2 * id_x + ((0xaa & mask_k) ? 1 : 0);
		fine_id_y = 2 * id_y + ((0xcc & mask_k) ? 1 : 0);
		fine_id_z = 2 * id_z + ((0xf0 & mask_k) ? 1 : 0);
		vox_pack = vox_pack << 7;
		radius = se_size; radius /= (2.f * res); radius *= fine_id_y;
		radius = se_size; radius /= (2.f * res); radius *= fine_id_x;
		ux = (float)(fine_id_x) / (2.f * res);
		uy = (float)(fine_id_y) / (2.f * res);
		uz = (float)(fine_id_z) / (2.f * res);
		ux = ux - 0.5f;
		uy = uy - 0.5f;
		radius0 =
		    square ((cos (theta0) * (ux - cx0) + sin (theta0) * (uy - cy0)) / (sigmax0)) +
		    square ((-sin (theta0) * (uy - cy0) + cos (theta0) * (uy - cy0)) / (sigmay0)) +
		    square ((uz - cz0) / (sigmaz0));
		radius1 =
		    square ((cos (theta1) * (ux - cx1) + sin (theta1) * (uy - cy1)) / (sigmax1)) +
		    square ((-sin (theta1) * (uy - cy1) + cos (theta1) * (uy - cy1)) / (sigmay1)) +
		    square ((uz - cz1) / (sigmaz1));
		radius2 =
		    square ((cos (theta2) * (ux - cx2) + sin (theta2) * (uy - cy2)) / (sigmax2)) +
		    square ((-sin (theta2) * (uy - cy2) + cos (theta2) * (uy - cy2)) / (sigmay2));
		radius = radius0;
		if (radius1 < radius)
			radius = radius1;
		if (radius2 < radius)
			radius = radius2;
		radius = (1.f - exp (-radius));
		radius *= se_size;
		uchar_radius = radius;
		uchar_radius += 2;
		uchar_radius = se_size;

		vox_pack |= (uchar_radius & (0x7f));
	}

	surf3Dwrite (vox_pack, level0, id_x * sizeof (unsigned long long), id_y, id_z);
}

__global__ void BuildMipmapFromTo (cudaTextureObject_t level_from,
                                   unsigned int to_width,
                                   unsigned int to_height,
                                   unsigned int to_depth,
                                   cudaSurfaceObject_t level_to) {
	int to_id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int to_id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int to_id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (to_id_x >= to_width || to_id_y >= to_height || to_id_z >= to_depth)
		return;

	unsigned int from_width = 2 * to_width;
	unsigned int from_height = 2 * to_height;
	unsigned int from_depth = 2 * to_depth;
	int from_id_x = 2 * to_id_x;
	int from_id_y = 2 * to_id_y;
	int from_id_z = 2 * to_id_z;

	float px = 1.f / float (from_width);
	float py = 1.f / float (from_height);
	float pz = 1.f / float (from_depth);

	uint2 val[8];
	val[0] = tex3D<uint2> (level_from,
	                       from_id_x * px, from_id_y * py, from_id_z * pz);
	val[1] = tex3D<uint2> (level_from,
	                       (from_id_x + 1) * px, from_id_y * py, from_id_z * pz);
	val[2] = tex3D<uint2> (level_from,
	                       from_id_x * px, (from_id_y + 1) * py, from_id_z * pz);
	val[3] = tex3D<uint2> (level_from,
	                       (from_id_x + 1) * px, (from_id_y + 1) * py, from_id_z * pz);
	val[4] = tex3D<uint2> (level_from,
	                       from_id_x * px, from_id_y * py, (from_id_z + 1) * pz);
	val[5] = tex3D<uint2> (level_from,
	                       (from_id_x + 1) * px, from_id_y * py, (from_id_z + 1) * pz);
	val[6] = tex3D<uint2> (level_from,
	                       from_id_x * px, (from_id_y + 1) * py, (from_id_z + 1) * pz);
	val[7] = tex3D<uint2> (level_from,
	                       (from_id_x + 1) * px, (from_id_y + 1) * py, (from_id_z + 1) * pz);


	unsigned long long mask_scale = 0x7f;
	unsigned long long vox_pack_out = 0;
	unsigned long long vox_pack_in;
	// Fill successives scales
	for (int l = 7; l >= 0; l--) {
		unsigned char max_scale = 0;
		vox_pack_in = *reinterpret_cast<unsigned long long*> (&val[l]);
		for (int k = 0; k < 8; k++) {
			unsigned char curr_scale = mask_scale & vox_pack_in;
			max_scale = curr_scale > max_scale ? curr_scale : max_scale;
			vox_pack_in = vox_pack_in >> 7;
		}
		vox_pack_out = vox_pack_out << 7;
		vox_pack_out |= (max_scale & (0x7f));
	}

	uint2 to_val = make_uint2 (0, 0);
	to_val = *reinterpret_cast<uint2*> (&vox_pack_out);

	to_val.y |= (val[0].y & 0xff000000) != 0 ? LC0 : 0;
	to_val.y |= (val[1].y & 0xff000000) != 0 ? LC1 : 0;
	to_val.y |= (val[2].y & 0xff000000) != 0 ? LC2 : 0;
	to_val.y |= (val[3].y & 0xff000000) != 0 ? LC3 : 0;
	to_val.y |= (val[4].y & 0xff000000) != 0 ? LC4 : 0;
	to_val.y |= (val[5].y & 0xff000000) != 0 ? LC5 : 0;
	to_val.y |= (val[6].y & 0xff000000) != 0 ? LC6 : 0;
	to_val.y |= (val[7].y & 0xff000000) != 0 ? LC7 : 0;

	surf3Dwrite (to_val, level_to,
	             to_id_x * sizeof (unsigned long long), to_id_y, to_id_z);
}

void Grid::BuildTexMipmaps (float se_size) {
	double time1, time2;
	ShowGPUMemoryUsage ();

	bool is_allocated = false;
	if (grid_gpu_.mipmapped_array == NULL)
		is_allocated = false;
	else
		is_allocated = true;

	int max_mipmap_level = log2 ((float) res_[0]) + 1;
	size_t width = res_[0];
	size_t height = res_[1];
	size_t depth = res_[2];

	cudaMipmappedArray_t cu_mipmap_array = NULL;
	if (!is_allocated) {
		// Allocate the mipmap : occupation is coded on 1bit and scale on 7bits,
		// giving (1+7)*8 = 64bits for one pack of voxels/cells
		cudaChannelFormatDesc mipmap_channel_desc;
		mipmap_channel_desc = cudaCreateChannelDesc (32, 32, 0, 0,
		                      cudaChannelFormatKindUnsigned);
		cudaExtent cu_mipmap_extent = make_cudaExtent (width, height, depth);

		cudaMallocMipmappedArray (&cu_mipmap_array,
		                          &mipmap_channel_desc, cu_mipmap_extent,
		                          max_mipmap_level);
		grid_gpu_.mipmapped_array = cu_mipmap_array;
	} else {
		cu_mipmap_array = grid_gpu_.mipmapped_array;
	}

	// Fill Level-0
	cudaArray_t level_first;
	cudaGetMipmappedArrayLevel(&level_first, cu_mipmap_array, 0);

	// Generate surface object for writing
	cudaSurfaceObject_t surf_output0;
	cudaResourceDesc surf_res0;
	memset (&surf_res0, 0, sizeof(cudaResourceDesc));
	surf_res0.resType = cudaResourceTypeArray;
	surf_res0.res.array.array = level_first;
	cudaCreateSurfaceObject (&surf_output0, &surf_res0);
	CheckCUDAError ();

	// Run fill kernel
	dim3 block_dim, grid_dim;
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((width / block_dim.x) + 1,
	                 (height / block_dim.y) + 1,
	                 (depth / block_dim.z) + 1);
	time1 = GET_TIME ();
	FillLevel0 <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, width, height, depth,
	 floor (2.f * se_size / cell_size_),
	 surf_output0);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[BuildTexMipmaps] : " << "level-0 filled in "
	          << time2 - time1 << " ms. with se_size : " << floor (2.f * se_size / cell_size_)
	          << std::endl;

	cudaDestroySurfaceObject(surf_output0);

	uint level = 0;

	while (width != 1) {
		width /= 2; height /= 2; depth /= 2;
		cudaArray_t level_from;
		cudaGetMipmappedArrayLevel(&level_from, cu_mipmap_array, level);
		cudaArray_t level_to;
		cudaGetMipmappedArrayLevel(&level_to, cu_mipmap_array, level + 1);

		cudaExtent level_to_size;
		cudaArrayGetInfo (NULL, &level_to_size, NULL, level_to);

		// Generate texture object for reading
		cudaTextureObject_t tex_input;

		cudaResourceDesc tex_res;
		memset (&tex_res, 0, sizeof(cudaResourceDesc));
		tex_res.resType = cudaResourceTypeArray;
		tex_res.res.array.array = level_from;

		cudaTextureDesc tex_descr;
		memset (&tex_descr, 0, sizeof(cudaTextureDesc));
		tex_descr.normalizedCoords = 1;
		tex_descr.filterMode = cudaFilterModePoint;
		tex_descr.addressMode[0] = cudaAddressModeClamp;
		tex_descr.addressMode[1] = cudaAddressModeClamp;
		tex_descr.addressMode[2] = cudaAddressModeClamp;
		tex_descr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject (&tex_input, &tex_res, &tex_descr, NULL);

		// Generate surface object for writing
		cudaSurfaceObject_t surf_output;
		cudaResourceDesc surf_res;
		memset (&surf_res, 0, sizeof(cudaResourceDesc));
		surf_res.resType = cudaResourceTypeArray;
		surf_res.res.array.array = level_to;
		cudaCreateSurfaceObject(&surf_output, &surf_res);

		block_dim = dim3 (8, 4, 4);
		grid_dim = dim3 ((width / block_dim.x) + 1,
		                 (height / block_dim.y) + 1,
		                 (depth / block_dim.z) + 1);
		time1 = GET_TIME ();
		BuildMipmapFromTo <<< grid_dim, block_dim>>>
		(tex_input, width, height, depth, surf_output);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[BuildTexMipmaps] : "
		          << "level-" << level + 1 << " computed in "
		          << time2 - time1 << " ms." << std::endl;

		cudaDestroySurfaceObject(surf_output);
		cudaDestroyTextureObject(tex_input);
		level++;
	}

	if (!is_allocated) {
		// Create texture object descriptors
		cudaResourceDesc res_desc;
		memset (&res_desc, 0, sizeof (res_desc));
		res_desc.resType = cudaResourceTypeMipmappedArray;
		res_desc.res.mipmap.mipmap = cu_mipmap_array;

		cudaTextureDesc tex_desc;
		memset (&tex_desc, 0, sizeof (tex_desc));
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.addressMode[0] = cudaAddressModeClamp;
		tex_desc.addressMode[1] = cudaAddressModeClamp;
		tex_desc.addressMode[2] = cudaAddressModeClamp;
		tex_desc.normalizedCoords = false;
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.maxMipmapLevelClamp = max_mipmap_level - 1;

		// Create texture object: we only have to do this once!
		cudaCreateTextureObject (&grid_gpu_.tex_mipmap,
		                         &res_desc, &tex_desc, NULL);
		grid_gpu_.max_mipmap_level = max_mipmap_level;
		CheckCUDAError ();
	}
	ShowGPUMemoryUsage ();
}

__global__ void FillLevel0 (GridGPU grid_gpu,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            char * scale_grid,
                            float global_scale,
                            cudaSurfaceObject_t level0) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (id_x >= width || id_y >= height || id_z >= depth)
		return;

	unsigned int i = height * height * id_z + width * id_y + id_x;

	unsigned long long vox_pack = 0;

	// Fill occup first
	vox_pack = grid_gpu.voxels[i];

	unsigned char uchar_radius;
	int res = width;
	int fine_id_x, fine_id_y, fine_id_z;

	// Fill successives scales
	float radius;
	for (int k = 7; k >= 0; k--) {
		unsigned int mask_k = (1 << k);
		fine_id_x = 2 * id_x + ((0xaa & mask_k) ? 1 : 0);
		fine_id_y = 2 * id_y + ((0xcc & mask_k) ? 1 : 0);
		fine_id_z = 2 * id_z + ((0xf0 & mask_k) ? 1 : 0);
		vox_pack = vox_pack << 7;
		radius = (float)scale_grid[fine_id_x + 2 * res * fine_id_y + 4 * res * res * fine_id_z];
		radius = max (min (radius, 127.f), 2.f);
		uchar_radius = radius;

		vox_pack |= (uchar_radius & (0x7f));
	}

	surf3Dwrite (vox_pack, level0, id_x * sizeof (unsigned long long), id_y, id_z);
}

void Grid::BuildTexMipmaps (const ScaleField & scale_field) {
	double time1, time2;
	ShowGPUMemoryUsage ();

	bool is_allocated = false;
	if (grid_gpu_.mipmapped_array == NULL)
		is_allocated = false;
	else
		is_allocated = true;

	int max_mipmap_level = log2 ((float) res_[0]) + 1;
	size_t width = res_[0];
	size_t height = res_[1];
	size_t depth = res_[2];

	cudaMipmappedArray_t cu_mipmap_array = NULL;
	if (!is_allocated) {
		// Allocate the mipmap : occupation is coded on 1bit and scale on 7bits,
		// giving (1+7)*8 = 64bits for one pack of voxels/cells
		cudaChannelFormatDesc mipmap_channel_desc;
		mipmap_channel_desc = cudaCreateChannelDesc (32, 32, 0, 0,
		                      cudaChannelFormatKindUnsigned);
		cudaExtent cu_mipmap_extent = make_cudaExtent (width, height, depth);

		cudaMallocMipmappedArray (&cu_mipmap_array,
		                          &mipmap_channel_desc, cu_mipmap_extent,
		                          max_mipmap_level);
		grid_gpu_.mipmapped_array = cu_mipmap_array;
	} else {
		cu_mipmap_array = grid_gpu_.mipmapped_array;
	}

	// Fill Level-0
	cudaArray_t level_first;
	cudaGetMipmappedArrayLevel(&level_first, cu_mipmap_array, 0);

	// Generate surface object for writing
	cudaSurfaceObject_t surf_output0;
	cudaResourceDesc surf_res0;
	memset (&surf_res0, 0, sizeof(cudaResourceDesc));
	surf_res0.resType = cudaResourceTypeArray;
	surf_res0.res.array.array = level_first;
	cudaCreateSurfaceObject (&surf_output0, &surf_res0);
	CheckCUDAError ();

	// Run fill kernel
	dim3 block_dim, grid_dim;
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((width / block_dim.x) + 1,
	                 (height / block_dim.y) + 1,
	                 (depth / block_dim.z) + 1);
	float * cu_scale_field = NULL;
	cudaMalloc (&cu_scale_field, 4 * sizeof (float)*scale_field.GetNumberOfPoints ());
	cudaMemcpy (cu_scale_field, &(scale_field.points ()[0]),
	            4 * sizeof (float)*scale_field.GetNumberOfPoints (),
	            cudaMemcpyHostToDevice);
	time1 = GET_TIME ();
	FillLevel0 <<< grid_dim, block_dim>>>
	(grid_gpu_, width, height, depth,
	 scale_field.scale_grid (),
	 scale_field.global_scale (),
	 surf_output0);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[BuildTexMipmaps] : " << "level-0 filled in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&cu_scale_field);
	cudaDestroySurfaceObject(surf_output0);

	uint level = 0;

	while (width != 1) {
		width /= 2; height /= 2; depth /= 2;
		cudaArray_t level_from;
		cudaGetMipmappedArrayLevel(&level_from, cu_mipmap_array, level);
		cudaArray_t level_to;
		cudaGetMipmappedArrayLevel(&level_to, cu_mipmap_array, level + 1);

		cudaExtent level_to_size;
		cudaArrayGetInfo (NULL, &level_to_size, NULL, level_to);

		// Generate texture object for reading
		cudaTextureObject_t tex_input;

		cudaResourceDesc tex_res;
		memset (&tex_res, 0, sizeof(cudaResourceDesc));
		tex_res.resType = cudaResourceTypeArray;
		tex_res.res.array.array = level_from;

		cudaTextureDesc tex_descr;
		memset (&tex_descr, 0, sizeof(cudaTextureDesc));
		tex_descr.normalizedCoords = 1;
		tex_descr.filterMode = cudaFilterModePoint;
		tex_descr.addressMode[0] = cudaAddressModeClamp;
		tex_descr.addressMode[1] = cudaAddressModeClamp;
		tex_descr.addressMode[2] = cudaAddressModeClamp;
		tex_descr.readMode = cudaReadModeElementType;
		cudaCreateTextureObject (&tex_input, &tex_res, &tex_descr, NULL);

		// Generate surface object for writing
		cudaSurfaceObject_t surf_output;
		cudaResourceDesc surf_res;
		memset (&surf_res, 0, sizeof(cudaResourceDesc));
		surf_res.resType = cudaResourceTypeArray;
		surf_res.res.array.array = level_to;
		cudaCreateSurfaceObject(&surf_output, &surf_res);

		block_dim = dim3 (8, 4, 4);
		grid_dim = dim3 ((width / block_dim.x) + 1,
		                 (height / block_dim.y) + 1,
		                 (depth / block_dim.z) + 1);
		time1 = GET_TIME ();
		BuildMipmapFromTo <<< grid_dim, block_dim>>>
		(tex_input, width, height, depth, surf_output);
		cudaDeviceSynchronize ();
		CheckCUDAError ();
		time2 = GET_TIME ();
		std::cout << "[BuildTexMipmaps] : "
		          << "level-" << level + 1 << " computed in "
		          << time2 - time1 << " ms." << std::endl;

		cudaDestroySurfaceObject(surf_output);
		cudaDestroyTextureObject(tex_input);
		level++;
	}

	if (!is_allocated) {
		// Create texture object descriptors
		cudaResourceDesc res_desc;
		memset (&res_desc, 0, sizeof (res_desc));
		res_desc.resType = cudaResourceTypeMipmappedArray;
		res_desc.res.mipmap.mipmap = cu_mipmap_array;

		cudaTextureDesc tex_desc;
		memset (&tex_desc, 0, sizeof (tex_desc));
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.addressMode[0] = cudaAddressModeClamp;
		tex_desc.addressMode[1] = cudaAddressModeClamp;
		tex_desc.addressMode[2] = cudaAddressModeClamp;
		tex_desc.normalizedCoords = false;
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.maxMipmapLevelClamp = max_mipmap_level - 1;

		// Create texture object: we only have to do this once!
		cudaCreateTextureObject (&grid_gpu_.tex_mipmap,
		                         &res_desc, &tex_desc, NULL);
		grid_gpu_.max_mipmap_level = max_mipmap_level;
		CheckCUDAError ();
	}
	ShowGPUMemoryUsage ();
}

void Grid::ClearTexMipmaps () {
	//	cudaResourceDesc res_desc;
	//	cudaGetTextureObjectResourceDesc (&res_desc, grid_gpu_.tex_mipmap);
	//	cudaMipmappedArray_t cu_mipmap_array = res_desc.res.mipmap.mipmap;
	std::cout << "free : " << (void*) grid_gpu_.mipmapped_array << " "
	          << (void*)this << std::endl;
	cudaDestroyTextureObject (grid_gpu_.tex_mipmap);
	cudaFreeMipmappedArray (grid_gpu_.mipmapped_array);
}

__global__ void ComputeMortonCoords (float3 * morton_coords,
                                     unsigned int morton_coords_size) {
	unsigned int morton_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (morton_id >= morton_coords_size)
		return;

	morton_coords[morton_id] = make_float3 (DecodeMorton3X (morton_id),
	                                        DecodeMorton3Y (morton_id),
	                                        DecodeMorton3Z (morton_id));
}

void Grid::BuildMortonCoords () {
	double time1, time2;
	int max_mipmap_depth = log2 ((float) res_[0]);
	max_mipmap_depth++;
	unsigned int morton_coords_size = pow3i (1 << max_mipmap_depth);
	unsigned int morton_coords_blocksize = 512;
	cudaMalloc (&grid_gpu_.morton_coords,
	            pow3i (1 << max_mipmap_depth)*sizeof (float3));
	CheckCUDAError ();

	time1 = GET_TIME ();
	ComputeMortonCoords <<< ((morton_coords_size / morton_coords_blocksize) + 1),
	                    morton_coords_blocksize >>>
	                    (grid_gpu_.morton_coords, morton_coords_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Grid] : " << "the morton coords built in "
	          << time2 - time1 << " ms." << std::endl;
}

__global__ void TagVoxelsContourByErosion26C (unsigned char * voxels, uint3 res,
        unsigned char * tagged_voxels,
        unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char center = voxels[res.x * res.y * (id_z) + res.x * (id_y) + (id_x)];
	unsigned char tag = 0x00;
	unsigned char val = 0x00;
	int3 id3;

	if (center == 0xff)
		return;

	// Check the 26-connectivity for the 8 voxels contained
	// in one unsigned char
	id3 = make_int3 (id_x - 1, id_y - 1, id_z - 1);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C0) != 0 ? C1 : 0;
	tag |= (val & C2) != 0 ? C3 : 0;
	tag |= (val & C4) != 0 ? C5 : 0;
	tag |= (val & C6) != 0 ? C7 : 0;

	tagged_voxels[key] = tag & (~center);
}

__global__ void TagVoxelsContourByErosion (unsigned char * voxels, uint3 res,
        unsigned char * tagged_voxels,
        unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char center = voxels[res.x * res.y * (id_z) + res.x * (id_y) + (id_x)];
	unsigned char tag = 0x00;
	unsigned char val = 0x00;
	int3 id3;

	if (center == 0xff)
		return;

	// Check the 6-connectivity for the 8 voxels contained
	// in one unsigned char
	id3 = make_int3 (id_x + 1, id_y + 0, id_z + 0);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C0) != 0 ? C1 : 0;
	tag |= (val & C2) != 0 ? C3 : 0;
	tag |= (val & C4) != 0 ? C5 : 0;
	tag |= (val & C6) != 0 ? C7 : 0;

	id3 = make_int3 (id_x - 1, id_y + 0, id_z + 0);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C1) != 0 ? C0 : 0;
	tag |= (val & C3) != 0 ? C2 : 0;
	tag |= (val & C5) != 0 ? C4 : 0;
	tag |= (val & C7) != 0 ? C6 : 0;

	id3 = make_int3 (id_x + 0, id_y + 1, id_z + 0);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C0) != 0 ? C2 : 0;
	tag |= (val & C1) != 0 ? C3 : 0;
	tag |= (val & C4) != 0 ? C6 : 0;
	tag |= (val & C5) != 0 ? C7 : 0;

	id3 = make_int3 (id_x, id_y - 1, id_z + 0);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C2) != 0 ? C0 : 0;
	tag |= (val & C3) != 0 ? C1 : 0;
	tag |= (val & C6) != 0 ? C4 : 0;
	tag |= (val & C7) != 0 ? C5 : 0;

	id3 = make_int3 (id_x + 0, id_y + 0, id_z + 1);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C0) != 0 ? C4 : 0;
	tag |= (val & C1) != 0 ? C5 : 0;
	tag |= (val & C2) != 0 ? C6 : 0;
	tag |= (val & C3) != 0 ? C7 : 0;

	id3 = make_int3 (id_x + 0, id_y + 0, id_z - 1);
	val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
	tag |= (val & C4) != 0 ? C0 : 0;
	tag |= (val & C5) != 0 ? C1 : 0;
	tag |= (val & C6) != 0 ? C2 : 0;
	tag |= (val & C7) != 0 ? C3 : 0;

	tag |= (center & C1) != 0 ? C0 : 0;
	tag |= (center & C2) != 0 ? C0 : 0;
	tag |= (center & C4) != 0 ? C0 : 0;

	tag |= (center & C0) != 0 ? C1 : 0;
	tag |= (center & C3) != 0 ? C1 : 0;
	tag |= (center & C5) != 0 ? C1 : 0;

	tag |= (center & C0) != 0 ? C2 : 0;
	tag |= (center & C3) != 0 ? C2 : 0;
	tag |= (center & C6) != 0 ? C2 : 0;

	tag |= (center & C1) != 0 ? C3 : 0;
	tag |= (center & C2) != 0 ? C3 : 0;
	tag |= (center & C7) != 0 ? C3 : 0;

	tag |= (center & C5) != 0 ? C4 : 0;
	tag |= (center & C6) != 0 ? C4 : 0;
	tag |= (center & C0) != 0 ? C4 : 0;

	tag |= (center & C4) != 0 ? C5 : 0;
	tag |= (center & C7) != 0 ? C5 : 0;
	tag |= (center & C1) != 0 ? C5 : 0;

	tag |= (center & C4) != 0 ? C6 : 0;
	tag |= (center & C7) != 0 ? C6 : 0;
	tag |= (center & C2) != 0 ? C6 : 0;

	tag |= (center & C5) != 0 ? C7 : 0;
	tag |= (center & C6) != 0 ? C7 : 0;
	tag |= (center & C3) != 0 ? C7 : 0;


	tagged_voxels[key] = tag & (~center);
}

__global__ void TagVoxelsContourMCM (unsigned char * voxels,
                                     uint3 res,
                                     unsigned char * tagged_voxels,
                                     unsigned char inside) {

	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	bool tag = false;
	unsigned char val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7;
	unsigned char v0, v1, v2, v3, v4, v5, v6, v7;
	unsigned int resxy = res.x * res.y;
	unsigned int resx = res.x;

	val_0 = 0; val_1 = 0; val_2 = 0; val_3 = 0;
	val_4 = 0; val_5 = 0; val_6 = 0; val_7 = 0;


	v0 = voxels[resxy * (id_z) + resx * (id_y) + (id_x)];
	v1 = voxels[resxy * (id_z) + resx * (id_y) + (id_x + 1)];
	v2 = voxels[resxy * (id_z) + resx * (id_y + 1) + (id_x)];
	v3 = voxels[resxy * (id_z) + resx * (id_y + 1) + (id_x + 1)];
	v4 = voxels[resxy * (id_z + 1) + resx * (id_y) + (id_x)];
	v5 = voxels[resxy * (id_z + 1) + resx * (id_y) + (id_x + 1)];
	v6 = voxels[resxy * (id_z + 1) + resx * (id_y + 1) + (id_x)];
	v7 = voxels[resxy * (id_z + 1) + resx * (id_y + 1) + (id_x + 1)];

	// Compute MCM Indices
	// v0 C0 | v0 C1 | v0 C3 | v0 C2
	// v0 C4 | v0 C5 | v0 C7 | v0 C6
	val_0 |= (v0 & C0) != 0 ? MC0 : 0;
	val_0 |= (v0 & C1) != 0 ? MC1 : 0;
	val_0 |= (v0 & C3) != 0 ? MC2 : 0;
	val_0 |= (v0 & C2) != 0 ? MC3 : 0;
	val_0 |= (v0 & C4) != 0 ? MC4 : 0;
	val_0 |= (v0 & C5) != 0 ? MC5 : 0;
	val_0 |= (v0 & C7) != 0 ? MC6 : 0;
	val_0 |= (v0 & C6) != 0 ? MC7 : 0;

	// v0 C1 | v1 C0 | v1 C2 | v0 C3
	// v0 C5 | v1 C4 | v1 C6 | v0 C7
	val_1 |= (v0 & C1) != 0 ? MC0 : 0;
	val_1 |= (v1 & C0) != 0 ? MC1 : 0;
	val_1 |= (v1 & C2) != 0 ? MC2 : 0;
	val_1 |= (v0 & C3) != 0 ? MC3 : 0;
	val_1 |= (v0 & C5) != 0 ? MC4 : 0;
	val_1 |= (v1 & C4) != 0 ? MC5 : 0;
	val_1 |= (v1 & C6) != 0 ? MC6 : 0;
	val_1 |= (v0 & C7) != 0 ? MC7 : 0;

	// v0 C2 | v0 C3 | v2 C1 | v2 C0
	// v0 C6 | v0 C7 | v2 C5 | v2 C4
	val_2 |= (v0 & C2) != 0 ? MC0 : 0;
	val_2 |= (v0 & C3) != 0 ? MC1 : 0;
	val_2 |= (v2 & C1) != 0 ? MC2 : 0;
	val_2 |= (v2 & C0) != 0 ? MC3 : 0;
	val_2 |= (v0 & C6) != 0 ? MC4 : 0;
	val_2 |= (v0 & C7) != 0 ? MC5 : 0;
	val_2 |= (v2 & C5) != 0 ? MC6 : 0;
	val_2 |= (v2 & C4) != 0 ? MC7 : 0;

	// v0 C3 | v1 C2 | v3 C0 | v2 C1
	// v0 C7 | v1 C6 | v3 C4 | v2 C5
	val_3 |= (v0 & C3) != 0 ? MC0 : 0;
	val_3 |= (v1 & C2) != 0 ? MC1 : 0;
	val_3 |= (v3 & C0) != 0 ? MC2 : 0;
	val_3 |= (v2 & C1) != 0 ? MC3 : 0;
	val_3 |= (v0 & C7) != 0 ? MC4 : 0;
	val_3 |= (v1 & C6) != 0 ? MC5 : 0;
	val_3 |= (v3 & C4) != 0 ? MC6 : 0;
	val_3 |= (v2 & C5) != 0 ? MC7 : 0;

	// ---------------------------- //

	// v0 C4 | v0 C5 | v0 C7 | v0 C6
	// v4 C0 | v4 C1 | v4 C3 | V4 C2
	val_4 |= (v0 & C4) != 0 ? MC0 : 0;
	val_4 |= (v0 & C5) != 0 ? MC1 : 0;
	val_4 |= (v0 & C7) != 0 ? MC2 : 0;
	val_4 |= (v0 & C6) != 0 ? MC3 : 0;
	val_4 |= (v4 & C0) != 0 ? MC4 : 0;
	val_4 |= (v4 & C1) != 0 ? MC5 : 0;
	val_4 |= (v4 & C3) != 0 ? MC6 : 0;
	val_4 |= (v4 & C2) != 0 ? MC7 : 0;

	// v0 C5 | v1 C4 | v1 C6 | v0 C7
	// v4 C1 | v5 C0 | v5 C2 | v4 C3
	val_5 |= (v0 & C5) != 0 ? MC0 : 0;
	val_5 |= (v1 & C4) != 0 ? MC1 : 0;
	val_5 |= (v1 & C6) != 0 ? MC2 : 0;
	val_5 |= (v0 & C7) != 0 ? MC3 : 0;
	val_5 |= (v4 & C1) != 0 ? MC4 : 0;
	val_5 |= (v5 & C0) != 0 ? MC5 : 0;
	val_5 |= (v5 & C2) != 0 ? MC6 : 0;
	val_5 |= (v4 & C3) != 0 ? MC7 : 0;

	// v0 C6 | v0 C7 | v2 C5 | v2 C4
	// v4 C2 | v4 C3 | v6 C1 | v6 C0
	val_6 |= (v0 & C6) != 0 ? MC0 : 0;
	val_6 |= (v0 & C7) != 0 ? MC1 : 0;
	val_6 |= (v2 & C5) != 0 ? MC2 : 0;
	val_6 |= (v2 & C4) != 0 ? MC3 : 0;
	val_6 |= (v4 & C2) != 0 ? MC4 : 0;
	val_6 |= (v4 & C3) != 0 ? MC5 : 0;
	val_6 |= (v6 & C1) != 0 ? MC6 : 0;
	val_6 |= (v6 & C0) != 0 ? MC7 : 0;

	// v0 C7 | v1 C6 | v3 C4 | v2 C5
	// v4 C3 | v5 C2 | v7 C0 | v6 C1
	val_7 |= (v0 & C7) != 0 ? MC0 : 0;
	val_7 |= (v1 & C6) != 0 ? MC1 : 0;
	val_7 |= (v3 & C4) != 0 ? MC2 : 0;
	val_7 |= (v2 & C5) != 0 ? MC3 : 0;
	val_7 |= (v4 & C3) != 0 ? MC4 : 0;
	val_7 |= (v5 & C2) != 0 ? MC5 : 0;
	val_7 |= (v7 & C0) != 0 ? MC6 : 0;
	val_7 |= (v6 & C1) != 0 ? MC7 : 0;

	tag |= (val_0 != 0) && (val_0 != 255);
	tag |= (val_1 != 0) && (val_1 != 255);
	tag |= (val_2 != 0) && (val_2 != 255);
	tag |= (val_3 != 0) && (val_3 != 255);
	tag |= (val_4 != 0) && (val_4 != 255);
	tag |= (val_5 != 0) && (val_5 != 255);
	tag |= (val_6 != 0) && (val_6 != 255);
	tag |= (val_7 != 0) && (val_7 != 255);

	tagged_voxels[key] = tag ? 0xff : 0x00;
}

__global__ void TagVoxelsContourBaseConservative (unsigned char * voxels,
        uint3 res,
        unsigned char * tagged_voxels,
        unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char center = voxels[key];
	unsigned char val;
	bool tag = false;
	int radius = 1;

	// Check all surrounding voxels in 26-connectivity
	for (int k = -radius; k <= radius; k++)
		for (int j = -radius; j <= radius; j++)
			for (int i = -radius; i <= radius; i++) {
				int3 id3 = make_int3 (id_x + i, id_y + j, id_z + k);

				if (0 <= id3.x && id3.x < res.x
				        && 0 <= id3.y && id3.y < res.y
				        && 0 <= id3.z && id3.z < res.z
				   )
					if (i != 0 && j != 0 && k != 0) {
						val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
						tag = tag || (val != center);
					}
			}
	tagged_voxels[key] = tag ? 0xff : 0x00;
}

__global__ void TagVoxelsContourBase (unsigned char * voxels,
                                      uint3 res,
                                      unsigned char * tagged_voxels,
                                      unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char center = voxels[key];
	unsigned char val;
	bool tag = false;
	int radius = 1;

	//	// Check all surrounding voxels in 26-connectivity
	//	for (int k = -radius; k <= radius; k++)
	//		for (int j = -radius; j <= radius; j++)
	//			for (int i = -radius; i <= radius; i++) {
	//				int3 id3 = make_int3 (id_x + i, id_y + j, id_z + k);
	//
	//				if (0 <= id3.x && id3.x < res.x
	//						&& 0 <= id3.y && id3.y < res.y
	//						&& 0 <= id3.z && id3.z < res.z
	//					 )
	//					if (i != 0 && j != 0 && k != 0) {
	//						val = voxels[res.x*res.y*id3.z + res.x*id3.y + id3.x];
	//						tag = tag || (val != center);
	//					}
	//			}
	//	tagged_voxels[key] = tag ? 0xff : 0x00;

	// Check all surrounding voxels in 6-connectivity
	for (int k = -radius; k <= radius; k++) {
		int3 id3 = make_int3 (id_x, id_y, id_z + k);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   )
			if (k != 0) {
				val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
				tag = tag || (val != center);
			}
	}
	for (int j = -radius; j <= radius; j++)	{
		int3 id3 = make_int3 (id_x, id_y + j, id_z);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   )
			if (j != 0) {
				val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
				tag = tag || (val != center);
			}
	}
	for (int i = -radius; i <= radius; i++) {
		int3 id3 = make_int3 (id_x + i, id_y, id_z);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   )
			if (i != 0) {
				val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
				tag = tag || (val != center);
			}
	}

	tagged_voxels[key] = tag ? 0xff : 0x00;

}

__global__ void TagVoxelsContourBaseTight (unsigned char * voxels,
        uint3 res,
        unsigned char * tagged_voxels,
        unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char v0, v1, v2, v3, v4, v5, v6, v7;
	unsigned char ind = 0;
	unsigned char isovalue = 0xff;

	v0 = voxels[res.x * res.y * (id_z) + res.x * (id_y) + (id_x)];
	v1 = voxels[res.x * res.y * (id_z) + res.x * (id_y) + (id_x + 1)];
	v2 = voxels[res.x * res.y * (id_z) + res.x * (id_y + 1) + (id_x)];
	v3 = voxels[res.x * res.y * (id_z) + res.x * (id_y + 1) + (id_x + 1)];
	v4 = voxels[res.x * res.y * (id_z + 1) + res.x * (id_y) + (id_x)];
	v5 = voxels[res.x * res.y * (id_z + 1) + res.x * (id_y) + (id_x + 1)];
	v6 = voxels[res.x * res.y * (id_z + 1) + res.x * (id_y + 1) + (id_x)];
	v7 = voxels[res.x * res.y * (id_z + 1) + res.x * (id_y + 1) + (id_x + 1)];

	if (v0 >= isovalue) ind |= 1;
	if (v1 >= isovalue) ind |= 2;
	if (v2 >= isovalue) ind |= 4;
	if (v3 >= isovalue) ind |= 8;
	if (v4 >= isovalue) ind |= 16;
	if (v5 >= isovalue) ind |= 32;
	if (v6 >= isovalue) ind |= 64;
	if (v7 >= isovalue) ind |= 128;

	tagged_voxels[key] = (ind != 0xff && ind != 0x00) ? 0xff : 0x00;
}


__global__ void TagVoxelsContourSymmetric (unsigned char * voxels, uint3 res,
        unsigned char * tagged_voxels,
        unsigned char inside) {
	int id_x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int id_y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int id_z = blockIdx.z * blockDim.z + threadIdx.z + 1;
	unsigned int key = res.x * res.y * id_z + res.x * id_y + id_x;

	if (id_x >= (res.x - 1) || id_y >= (res.y - 1) || id_z >= (res.z - 1))
		return;

	unsigned char ind = 0, center;
	center = voxels[res.x * res.y * id_z + res.x * id_y + id_x];

	if (center != 0 && center != 255) {
		ind = 128;
	} else if (center == inside) {
		int3 id3;
		unsigned char val;

		// Check all the surrounding faces
		id3 = make_int3 (id_x + 1, id_y + 0, id_z + 0);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C0 || C2 || C4 || C6)) != 0)
				ind = 128;
		}

		id3 = make_int3 (id_x - 1, id_y + 0, id_z + 0);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C1 || C3 || C5 || C7)) != 0)
				ind = 128;
		}

		id3 = make_int3 (id_x + 0, id_y + 1, id_z + 0);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C0 || C1 || C4 || C5)) != 0)
				ind = 128;
		}

		id3 = make_int3 (id_x, id_y - 1, id_z + 0);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C2 || C3 || C6 || C7)) != 0)
				ind = 128;
		}

		id3 = make_int3 (id_x + 0, id_y + 0, id_z + 1);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C0 || C1 || C2 || C3)) != 0)
				ind = 128;
		}

		id3 = make_int3 (id_x + 0, id_y + 0, id_z - 1);
		if (0 <= id3.x && id3.x < res.x
		        && 0 <= id3.y && id3.y < res.y
		        && 0 <= id3.z && id3.z < res.z
		   ) {
			val = voxels[res.x * res.y * id3.z + res.x * id3.y + id3.x];
			val = inside == 0xff ? ~val : val;
			if ((val & (C4 || C5 || C6 || C7)) != 0)
				ind = 128;
		}
		//TODO: Check edges and corners
	}

	tagged_voxels[key] = ind;
}

__global__ void FillCanonicalIndices (unsigned int * indices,
                                      unsigned int size) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		indices[idx] = idx;
}

//struct IsAContour {
//	__host__ __device__ bool operator() (const unsigned char u) {
//		return (u != 0 && u != 255);
//	}
//};

__global__ void ExpandIndicesAndValuesBy8 (unsigned int * exp_indices,
        unsigned char * exp_values,
        unsigned int exp_size,
        unsigned int * indices,
        unsigned char * values,
        uint3 res) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= exp_size)
		return;
	unsigned int idxDiv8 = idx / 8;
	unsigned char idxRem8 = idx % 8;
	unsigned int index = indices[idxDiv8];
	unsigned char val = values[index];

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res.x * res.y;
	id_z = index / resxy;
	remainder = index % resxy;
	id_y = remainder / res.x;
	id_x = remainder % res.x;

	//	if (idx == 30000)
	//		printf ("contour center : %i %i %i\n", id_x, id_y, id_z);

	id_x = 2 * id_x + ((0xaa & (1 << idxRem8)) ? 1 : 0);
	id_y = 2 * id_y + ((0xcc & (1 << idxRem8)) ? 1 : 0);
	id_z = 2 * id_z + ((0xf0 & (1 << idxRem8)) ? 1 : 0);

	exp_indices[idx] = 4 * resxy * id_z + 2 * res.x * id_y + id_x;
	exp_values[idx] = (val & (1 << idxRem8)) >> idxRem8;
	//	exp_values[idx] = (idxRem8 == 0 && val != 0) ? 1 : 0;
}

void Grid::ComputeContour (unsigned int * & contour, unsigned int & contour_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 2) / block_dim.x) + 1, ((res_[1] - 2) / block_dim.y) + 1,
	                 ((res_[2] - 2) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	TagVoxelsContourByErosion <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, grid_gpu_.res,
	 tagged_voxels, 0xff);
	//	TagVoxelsContourSymmetric<<<grid_dim, block_dim>>>
	//		(grid_gpu_.voxels, grid_gpu_.res,
	//		 tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	unsigned int * exp_contour = NULL;
	unsigned char * exp_values = NULL;
	cudaMalloc (&exp_values, 8 * contour_size * sizeof (unsigned char));
	cudaMalloc (&exp_contour, 8 * contour_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_exp_contour (exp_contour);
	thrust::device_ptr<unsigned char> devptr_exp_values (exp_values);
	thrust::device_vector<unsigned int>::iterator exp_old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator exp_new_end;

	time1 = GET_TIME ();
	ExpandIndicesAndValuesBy8 <<< (8 * contour_size / block_size) + 1, block_size >>>
	(exp_contour, exp_values, 8 * contour_size, contour,
	 //		 grid_gpu_.voxels,
	 tagged_voxels,
	 grid_gpu_.res);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	exp_new_end = thrust::copy_if (devptr_exp_contour,
	                               devptr_exp_contour + 8 * contour_size,
	                               devptr_exp_values,
	                               devptr_contour,
	                               IsNonZero ());
	contour_size = exp_new_end - exp_old_end;

	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " second pass "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
	FreeGPUResource (&exp_values);
	FreeGPUResource (&exp_contour);
}

__global__ void ComputeMortonCodesBase (unsigned int * indices,
                                        unsigned int indices_size,
                                        uint3 indices_res,
                                        unsigned int * morton_codes) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= indices_size)
		return;

	// Compute 3D indices from linear indices
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = indices[idx];
	resxy = indices_res.x * indices_res.y;
	id_z = key / resxy;
	remainder = key % resxy;
	id_y = remainder / (indices_res.x);
	id_x = remainder % (indices_res.x);

	// Compute 3D morton codes from 3D indices
	unsigned int morton = EncodeMorton3 (id_x, id_y, id_z);
	morton_codes[idx] = morton;
}

void Grid::ComputeContourMCM (unsigned int * & contour,
                              unsigned int & contour_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 2) / block_dim.x) + 1, ((res_[1] - 2) / block_dim.y) + 1,
	                 ((res_[2] - 2) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	TagVoxelsContourMCM <<< grid_dim, block_dim,
	                    ((block_dim.x + 1) * (block_dim.y + 1) * (block_dim.z + 1))*sizeof (unsigned char) >>>
	                    (grid_gpu_.voxels, grid_gpu_.res,
	                     tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	// Sort the contour in Morton order
	unsigned int * sorted_morton = NULL;
	cudaMalloc (&sorted_morton, contour_size * sizeof (unsigned int));
	time1 = GET_TIME ();
	ComputeMortonCodesBase <<< (contour_size / block_size) + 1, block_size >>>
	(contour, contour_size, grid_gpu_.res,
	 sorted_morton);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes for the conservative contour computed in "
	          << time2 - time1 << " ms." << std::endl;

	thrust::device_ptr<unsigned int> devptr_keys (sorted_morton);
	thrust::device_ptr<unsigned int> devptr_values (contour);
	time1 = GET_TIME ();
	thrust::sort_by_key (devptr_keys,
	                     devptr_keys + contour_size,
	                     devptr_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes of the conservative contour sorted in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&sorted_morton);

	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
}
/*
 * ComputeContourBase extract a contour searching for a
 * 6-connected contour at base resolution
 */
void Grid::ComputeContourBase (unsigned int * & contour,
                               unsigned int & contour_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 2) / block_dim.x) + 1, ((res_[1] - 2) / block_dim.y) + 1,
	                 ((res_[2] - 2) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	TagVoxelsContourBase <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, grid_gpu_.res,
	 tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	// Sort the contour in Morton order
	unsigned int * sorted_morton = NULL;
	cudaMalloc (&sorted_morton, contour_size * sizeof (unsigned int));
	time1 = GET_TIME ();
	ComputeMortonCodesBase <<< (contour_size / block_size) + 1, block_size >>>
	(contour, contour_size, grid_gpu_.res,
	 sorted_morton);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes for the conservative contour computed in "
	          << time2 - time1 << " ms." << std::endl;

	thrust::device_ptr<unsigned int> devptr_keys (sorted_morton);
	thrust::device_ptr<unsigned int> devptr_values (contour);
	time1 = GET_TIME ();
	thrust::sort_by_key (devptr_keys,
	                     devptr_keys + contour_size,
	                     devptr_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes of the conservative contour sorted in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&sorted_morton);

	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
}

void Grid::ComputeContourBaseTight (unsigned int * & contour,
                                    unsigned int & contour_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 1) / block_dim.x) + 1, ((res_[1] - 1) / block_dim.y) + 1,
	                 ((res_[2] - 1) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	TagVoxelsContourBaseTight <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, grid_gpu_.res,
	 tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	// Sort the contour in Morton order
	unsigned int * sorted_morton = NULL;
	cudaMalloc (&sorted_morton, contour_size * sizeof (unsigned int));
	time1 = GET_TIME ();
	ComputeMortonCodesBase <<< (contour_size / block_size) + 1, block_size >>>
	(contour, contour_size, grid_gpu_.res,
	 sorted_morton);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes for the conservative contour computed in "
	          << time2 - time1 << " ms." << std::endl;

	thrust::device_ptr<unsigned int> devptr_keys (sorted_morton);
	thrust::device_ptr<unsigned int> devptr_values (contour);
	time1 = GET_TIME ();
	thrust::sort_by_key (devptr_keys,
	                     devptr_keys + contour_size,
	                     devptr_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes of the conservative contour sorted in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&sorted_morton);
	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
}

void Grid::ComputeContourBaseConservative (unsigned int * & contour,
        unsigned int & contour_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 1) / block_dim.x) + 1, ((res_[1] - 1) / block_dim.y) + 1,
	                 ((res_[2] - 1) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	TagVoxelsContourBaseConservative <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, grid_gpu_.res,
	 tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	std::cout << "[Compute Contour] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	// Sort the contour in Morton order
	unsigned int * sorted_morton = NULL;
	cudaMalloc (&sorted_morton, contour_size * sizeof (unsigned int));
	time1 = GET_TIME ();
	ComputeMortonCodesBase <<< (contour_size / block_size) + 1, block_size >>>
	(contour, contour_size, grid_gpu_.res,
	 sorted_morton);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes for the conservative contour computed in "
	          << time2 - time1 << " ms." << std::endl;

	thrust::device_ptr<unsigned int> devptr_keys (sorted_morton);
	thrust::device_ptr<unsigned int> devptr_values (contour);
	time1 = GET_TIME ();
	thrust::sort_by_key (devptr_keys,
	                     devptr_keys + contour_size,
	                     devptr_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour] : " << contour_size
	          << " morton codes of the conservative contour sorted in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&sorted_morton);

	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
}
__global__ void FillValuesWRTIndices (const unsigned int * indices,
                                      unsigned int size,
                                      const unsigned char * values,
                                      unsigned char * values_of_indices) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	unsigned int value_index = indices[idx];
	unsigned char value_of_index = values[value_index];
	values_of_indices[idx] = value_of_index;
}

void Grid::ComputeContourPacked (unsigned int * & contour,
                                 unsigned char *& contour_values,
                                 unsigned int & contour_size) {
	ComputeContourPacked (contour, contour_values, contour_size, GRID_3D_26C);
}

void Grid::ComputeContourPacked (unsigned int * & contour,
                                 unsigned char *& contour_values,
                                 unsigned int & contour_size,
                                 GridConnectivity contour_type) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 2) / block_dim.x) + 1, ((res_[1] - 2) / block_dim.y) + 1,
	                 ((res_[2] - 2) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	unsigned int * init_contour = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));
	cudaMalloc (&init_contour, grid_size * sizeof (unsigned int));

	time1 = GET_TIME ();
	switch (contour_type) {
	case GRID_3D_26C:
		TagVoxelsContourByErosion <<< grid_dim, block_dim>>>
		(grid_gpu_.voxels, grid_gpu_.res,
		 tagged_voxels, 0xff);
		break;
	case GRID_3D_6C:
		TagVoxelsContourByErosion <<< grid_dim, block_dim>>>
		(grid_gpu_.voxels, grid_gpu_.res,
		 tagged_voxels, 0xff);
		break;
	}
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Compute Contour Packed] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	// Compaction of the tagged full grid
	cudaMalloc (&contour, grid_size * sizeof (unsigned int));
	thrust::device_ptr<unsigned int> devptr_contour (contour);
	thrust::device_ptr<unsigned int> devptr_init_contour (init_contour);
	thrust::device_ptr<unsigned char> devptr_tagged_voxels (tagged_voxels);
	thrust::device_vector<unsigned int>::iterator old_end (devptr_contour);
	thrust::device_vector<unsigned int>::iterator new_end;
	unsigned int block_size = 512;
	contour_size = 0;

	time1 = GET_TIME ();
	FillCanonicalIndices <<< (grid_size / block_size) + 1, block_size >>>
	(init_contour, grid_size);
	cudaDeviceSynchronize ();
	CheckCUDAError ();

	new_end = thrust::copy_if (devptr_init_contour,
	                           devptr_init_contour + grid_size,
	                           devptr_tagged_voxels,
	                           devptr_contour,
	                           IsNonZero ());
	contour_size = new_end - old_end;
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour Packed] : "
	          << contour_size << " tagged contour voxels compacted by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	// Allocate and copy the computed contour in a more
	// tight memory bound
	unsigned int * tight_memory_contour = NULL;
	cudaMalloc (&tight_memory_contour, contour_size * sizeof (unsigned int));
	cudaMemcpy (tight_memory_contour, contour,
	            contour_size * sizeof (unsigned int),
	            cudaMemcpyDeviceToDevice);
	FreeGPUResource (&contour);
	contour = tight_memory_contour;

	// Associate an occupation pattern for each contour cell found
	cudaMalloc (&contour_values, contour_size * sizeof (unsigned char));
	time1 = GET_TIME ();
	FillValuesWRTIndices <<< (contour_size / block_size) + 1, block_size >>>
	(contour, contour_size,
	 tagged_voxels,
	 //		 grid_gpu_.voxels,
	 contour_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Compute Contour Packed] : "
	          << "fill contour occupation pattern in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&init_contour);
	FreeGPUResource (&tagged_voxels);
}

void Grid::TransformToContour () {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	// Start by tagging a full grid with contour cells
	block_dim = dim3 (8, 8, 8);
	grid_dim = dim3 (((res_[0] - 2) / block_dim.x) + 1, ((res_[1] - 2) / block_dim.y) + 1,
	                 ((res_[2] - 2) / block_dim.z) + 1);

	unsigned char * tagged_voxels = NULL;
	cudaMalloc (&tagged_voxels, grid_size * sizeof (unsigned char));
	cudaMemset (tagged_voxels, 0, grid_size * sizeof (unsigned char));

	time1 = GET_TIME ();
	TagVoxelsContourByErosion <<< grid_dim, block_dim>>>
	(grid_gpu_.voxels, grid_gpu_.res,
	 tagged_voxels, 0xff);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Transform To Contour] "
	          << "voxels tagged in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&grid_gpu_.voxels);

	grid_gpu_.voxels = tagged_voxels;
}

void Grid::MipmapMeshing2x2x2 (int max_mipmap_depth) {
	if (max_mipmap_depth >= 9)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth],
		              res_[0], cell_size_,
		              "mipmap_d9.off");
	if (max_mipmap_depth >= 8)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 1],
		              res_[0] / 2, 2 * cell_size_,
		              "mipmap_d8.off");
	if (max_mipmap_depth >= 7)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 2],
		              res_[0] / 4, 4 * cell_size_,
		              "mipmap_d7.off");
	if (max_mipmap_depth >= 6)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 3],
		              res_[0] / 8, 8 * cell_size_,
		              "mipmap_d6.off");
	if (max_mipmap_depth >= 5)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 4],
		              res_[0] / 16, 16 * cell_size_,
		              "mipmap_d5.off");
	if (max_mipmap_depth >= 4)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 5],
		              res_[0] / 32, 32 * cell_size_,
		              "mipmap_d4.off");
	if (max_mipmap_depth >= 2)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 6],
		              res_[0] / 64, 64 * cell_size_,
		              "mipmap_d3.off");
	if (max_mipmap_depth >= 1)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 7],
		              res_[0] / 128, 128 * cell_size_,
		              "mipmap_d2.off");
	if (max_mipmap_depth >= 0)
		Meshing2x2x2 (grid_gpu_.mipmap[max_mipmap_depth - 8],
		              res_[0] / 256, 256 * cell_size_,
		              "mipmap_d1.off");
}

void Grid::Meshing2x2x2 (unsigned char * voxels, unsigned int base_res,
                         float base_cell_size, const std::string & filename) {
	double time1, time2;
	uint3 base_res3 = make_uint3 (base_res, base_res, base_res);
	unsigned int num_cells_2x2x2 = 2 * base_res * 2 * base_res * 2 * base_res;

	dim3 block_dim, grid_dim;
	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 (((2 * base_res) / block_dim.x) + 1, ((2 * base_res) / block_dim.y) + 1,
	                 ((2 * base_res) / block_dim.z) + 1);

	float * voxels_2x2x2 = NULL;
	cudaMalloc (&voxels_2x2x2, num_cells_2x2x2 * sizeof (float));
	CheckCUDAError ();

	time1 = GET_TIME ();
	ConvertVoxelGridByBit <<< grid_dim, block_dim>>>
	(voxels,
	 base_res3,
	 voxels_2x2x2);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Convert Voxel Grid] "
	          << "convert by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	Vec3f bboxMCM = bbox_.min();

	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), 0.5f * base_cell_size,
	                                    0.5f * base_cell_size, 0.5f * base_cell_size,
	                                    2 * base_res, 2 * base_res, 2 * base_res);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (voxels_2x2x2, 0.5f, 0.5 * NAN_EVAL);
	mesher.saveMesh (filename.c_str ());
	time2 = GET_TIME ();

	//	FreeGPUResource (&voxels_2x2x2);
}
