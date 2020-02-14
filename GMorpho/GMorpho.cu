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

#include <fstream>
#include <sstream>
#include <cfloat>

#include <queue>

#include <eigen3/Eigen/Dense>

#include <cuda_profiler_api.h>

// THRUST related header for morton sorting and compaction
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "cuda_math.h"
#include "MarchingCubesMesher.h"
#include "GMorpho.h"

using namespace MorphoGraphics;

void GMorpho::print (const std::string & msg) {
	std::cout << "[GMorpho] : " << msg << std::endl;
}

void GMorpho::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if (err != cudaSuccess) {
		GMorpho::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw GMorpho::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

template<typename T>
void GMorpho::FreeGPUResource (T ** res) {
	if (*res != NULL) {
		cudaFree (*res);
		*res = NULL;
	}
}

void GMorpho::ShowGPUMemoryUsage () {
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);
	used = total - avail;
	std::cout << "[GMorpho] : Device memory used: " << (float)used / 1e6
	          << " of " << (float)total / 1e6 << std::endl;
}

GMorpho::GMorpho () {
	global_warp_counter_ = NULL;
	grid_2x2x2_uint_ = NULL;
	mcm_contour_indices_ = NULL;
	mcm_contour_values_ = NULL;
	mcm_contour_non_empty_ = NULL;
	mcm_compact_contour_values_ = NULL;
	mcm_compact_contour_indices_ = NULL;
	mcm_contour_neigh_indices_ = NULL;
	mcm_compact_contour_neigh_morpho_centroids_ = NULL;
	mcm_compact_contour_neigh_indices_ = NULL;
	mcm_vertices = NULL;
	mcm_normals = NULL;
	mcm_bilateral_normals = NULL;
	mcm_triangles = NULL;

	//	//Enable peer access between participating GPUs:
	//	cudaSetDevice (0);
	//	cudaDeviceEnablePeerAccess (1, 0);
	//	cudaSetDevice (1);
	//	cudaDeviceEnablePeerAccess (0, 0);
	//	cudaSetDevice (0);
	//	cudaSetDevice (1);
}

GMorpho::~GMorpho () {

}

struct Node {
	int depth;
	unsigned int idx;
};

__device__ Node make_node (int _depth, unsigned int _idx) {
	Node node;
	node.idx = _idx;
	node.depth = _depth;
	return node;
}

struct NodeTex {
	short depth;
	ushort3 id;
};

__device__ NodeTex make_node (short _depth,
                              short _idx, short _idy, short _idz) {
	NodeTex node;
	node.id.x = _idx;
	node.id.y = _idy;
	node.id.z = _idz;
	node.depth = _depth;
	return node;
}

__device__ NodeTex make_node (short _depth,
                              const ushort3 & _id) {
	NodeTex node;
	node.id.x = _id.x;
	node.id.y = _id.y;
	node.id.z = _id.z;
	node.depth = _depth;
	return node;
}

__device__ bool AABBAABBIntersect (const float3 & aabb_min0,
                                   const float3 & aabb_max0,
                                   const float3 & aabb_min1,
                                   const float3 & aabb_max1) {
	bool overlap_x = aabb_min0.x <= aabb_max1.x && aabb_max0.x >= aabb_min1.x;
	bool overlap_y = aabb_min0.y <= aabb_max1.y && aabb_max0.y >= aabb_min1.y;
	bool overlap_z = aabb_min0.z <= aabb_max1.z && aabb_max0.z >= aabb_min1.z;
	bool overlap = overlap_x && overlap_y && overlap_z;
	return overlap;
}

__device__ unsigned char ComputeNodeValue (const Node & node,
        cudaTextureObject_t mipmap_tex) {
	//	unsigned int mipmap_node_offset = 1 << 3*node.depth;
	//	mipmap_node_offset -= 1;
	//	mipmap_node_offset = mipmap_node_offset/7;
	//	mipmap_node_offset += node.idx;
	//	unsigned char * mipmap = morton_mipmap + mipmap_node_offset;
	//	return *mipmap;

	float3 node_coords = make_float3 (DecodeMorton3X (node.idx),
	                                  DecodeMorton3Y (node.idx),
	                                  DecodeMorton3Z (node.idx));

	return tex3DLod<unsigned char> (mipmap_tex,
	                                node_coords.x, node_coords.y, node_coords.z,
	                                (int)node.depth);
}

__device__ unsigned char ComputeNodeValue (const Node & node,
        unsigned char * morton_mipmap) {
	unsigned int mipmap_node_offset = 1 << 3 * node.depth;
	mipmap_node_offset -= 1;
	mipmap_node_offset = mipmap_node_offset / 7;
	mipmap_node_offset += node.idx;
	unsigned char * mipmap = morton_mipmap + mipmap_node_offset;
	return *mipmap;
}

__device__ bool HasCubeSEOverlap (const float3 & query_min,
                                  const float3 & query_max,
                                  const float3 & morton_coords) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 aabb_min = morton_coords;
	float3 aabb_max = aabb_min + make_float3 (1.f,
	                  1.f,
	                  1.f);
	node_has_overlap = AABBAABBIntersect (query_min, query_max,
	                                      aabb_min, aabb_max);
	return node_has_overlap;
}

__device__ bool HasSphereSEOverlap (const float3 & query_min,
                                    const float3 & query_max,
                                    const ushort3 & coords) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 aabb_min = make_float3 (coords.x, coords.y, coords.z);
	float3 aabb_max = aabb_min + make_float3 (1.f,
	                  1.f,
	                  1.f);
	float3 query = 0.5f * (query_min + query_max);
	float sq_se_size = 0.5f * (query_max.x - query_min.x);
	sq_se_size *= sq_se_size;
	//	node_has_overlap = AABBAABBIntersect (query_min, query_max,
	//																				aabb_min, aabb_max);
	node_has_overlap = SphereAABBSqDistance (query, aabb_min, aabb_max) < (sq_se_size);
	return node_has_overlap;
}

/*
 * Main code use this function :
 * HasCubeSEOverlap (const float3 & query_min, const float3 & query_max, const ushort3 & coords);
 */

__device__ bool HasRotCubeSEOverlap (const float3 & query_min,
                                     const float3 & query_max,
                                     const float4 & quat,
                                     const ushort3 & coords,
                                     float cell_size) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float4 q_center = make_float4 (0.5f * (query_max.x - query_min.x) + query_min.x,
	                               0.5f * (query_max.y - query_min.y) + query_min.y,
	                               0.5f * (query_max.z - query_min.z) + query_min.z, 0.f);
	float4 q_coords = make_float4 (cell_size * (coords.x + 0.5f),
	                               cell_size * (coords.y + 0.5f),
	                               cell_size * (coords.z + 0.5f), 0.f);
	float4 q_query = q_coords - q_center;
	float4 q_rot_query = hamilton (conj (quat), hamilton (q_query, quat));
	q_rot_query = q_rot_query + q_center;
	float3 rot_query = make_float3 (q_rot_query.x, q_rot_query.y, q_rot_query.z);
	//	float3 rot_query = make_float3 (coords.x, coords.y, coords.z);
	node_has_overlap = SphereAABBSqDistance (rot_query,
	                   query_min, query_max) < (cell_size * cell_size * 1.5f * 1.5f);


	return node_has_overlap;
}

__device__ bool HasRotCubeSEOverlap (const float3 & b_x,
                                     const float3 & b_y,
                                     const float3 & b_z,
                                     const float3 & t,
                                     float se_size,
                                     float cell_size) {
	// CASE 1 : L = Ax
	if (fabs (t.x) > (cell_size + se_size * (fabs (b_x.x) + fabs (b_y.x) + fabs (b_z.x))))
		return false;
	// CASE 2 : L = Ay
	if (fabs (t.y) > (cell_size + se_size * (fabs (b_x.y) + fabs (b_y.y) + fabs (b_z.y))))
		return false;
	// CASE 3 : L = Az
	if (fabs (t.z) > (cell_size + se_size * (fabs (b_x.z) + fabs (b_y.z) + fabs (b_z.z))))
		return false;
	// CASE 4 : L = Bx
	if (fabs (dotProduct (t, b_x)) > (cell_size * (fabs (b_x.x) + fabs (b_x.y) + fabs (b_x.z)) + se_size))
		return false;
	// CASE 5 : L = By
	if (fabs (dotProduct (t, b_y)) > (cell_size * (fabs (b_y.x) + fabs (b_y.y) + fabs (b_y.z)) + se_size))
		return false;
	// CASE 6 : L = Bz
	if (fabs (dotProduct (t, b_z)) > (cell_size * (fabs (b_z.x) + fabs (b_z.y) + fabs (b_z.z)) + se_size))
		return false;
	// CASE 7 : L = Ax ^ Bx
	if (fabs (t.z * b_x.y - t.y * b_x.z) > cell_size * (fabs (b_x.z) + fabs (b_x.y))
	        + se_size * (fabs (b_z.x) + fabs (b_y.x)))
		return false;
	// CASE 8 : L = Ax ^ By
	if (fabs (t.z * b_y.y - t.y * b_y.z) > cell_size * (fabs (b_y.z) + fabs (b_y.y))
	        + se_size * (fabs (b_z.x) + fabs (b_x.x)))
		return false;
	// CASE 9 : L = Ax ^ Bz
	if (fabs (t.z * b_z.y - t.y * b_z.z) > cell_size * (fabs (b_z.z) + fabs (b_z.y))
	        + se_size * (fabs (b_y.x) + fabs (b_x.x)))
		return false;
	// CASE 10 : L = Ay ^ Bx
	if (fabs (t.x * b_x.z - t.z * b_x.x) > cell_size * (fabs (b_x.z) + fabs (b_x.x))
	        + se_size * (fabs (b_z.y) + fabs (b_y.y)))
		return false;
	// CASE 11 : L = Ay ^ By
	if (fabs (t.x * b_y.z - t.z * b_y.x) > cell_size * (fabs (b_y.z) + fabs (b_y.x))
	        + se_size * (fabs (b_z.y) + fabs (b_x.y)))
		return false;
	// CASE 12 : L = Ay ^ Bz
	if (fabs (t.x * b_z.z - t.z * b_z.x) > cell_size * (fabs (b_z.z) + fabs (b_z.x))
	        + se_size * (fabs (b_y.y) + fabs (b_x.y)))
		return false;
	// CASE 13 : L = Az ^ Bx
	if (fabs (t.y * b_x.x - t.x * b_x.y) > cell_size * (fabs (b_x.y) + fabs (b_x.x))
	        + se_size * (fabs (b_z.z) + fabs (b_y.z)))
		return false;
	// CASE 14 : L = Az ^ By
	if (fabs (t.y * b_y.x - t.x * b_y.y) > cell_size * (fabs (b_y.y) + fabs (b_y.x))
	        + se_size * (fabs (b_z.z) + fabs (b_x.z)))
		return false;
	// CASE 15 : L = Az ^ Bz
	if (fabs (t.y * b_z.x - t.x * b_z.y) > cell_size * (fabs (b_z.y) + fabs (b_z.x))
	        + se_size * (fabs (b_y.z) + fabs (b_x.z)))
		return false;
	return true;
}

__device__ bool HasRotCubeSEOverlap (const float3 & query_min,
                                     const float3 & query_max,
                                     const float4 & quat,
                                     const ushort3 & coords,
                                     float cell_size,
                                     float se_size) {
	bool node_has_overlap;
	float3 se_center = make_float3 (0.5f * (query_max.x - query_min.x) + query_min.x,
	                                0.5f * (query_max.y - query_min.y) + query_min.y,
	                                0.5f * (query_max.z - query_min.z) + query_min.z);
	float3 node_center = make_float3 (cell_size * (coords.x + 0.5f),
	                                  cell_size * (coords.y + 0.5f),
	                                  cell_size * (coords.z + 0.5f));

	float3 b_x, b_y, b_z, t;
	float3 e_x, e_y, e_z;

	e_x = make_float3 (1.f, 0.f, 0.f);
	e_y = make_float3 (0.f, 1.f, 0.f);
	e_z = make_float3 (0.f, 0.f, 1.f);
	b_x = rotate (quat, e_x);
	b_y = rotate (quat, e_y);
	b_z = rotate (quat, e_z);

	t = se_center - node_center;

	node_has_overlap = HasRotCubeSEOverlap (b_x, b_y, b_z, t, se_size, 0.5f * cell_size);
	return node_has_overlap;
}

__device__ bool HasRotCubeSEOverlap (const float3 & se_center,
                                     const float3 & b_x,
                                     const float3 & b_y,
                                     const float3 & b_z,
                                     const ushort3 & coords,
                                     float cell_size,
                                     float se_size) {
	bool node_has_overlap;
	float3 node_center = make_float3 (cell_size * (coords.x + 0.5f),
	                                  cell_size * (coords.y + 0.5f),
	                                  cell_size * (coords.z + 0.5f));
	float3 t = se_center - node_center;

	node_has_overlap = HasRotCubeSEOverlap (b_x, b_y, b_z, t, se_size, 0.5f * cell_size);
	return node_has_overlap;
}

__device__ bool HasCubeSEOverlap (const float3 & query_min,
                                  const float3 & query_max,
                                  const ushort3 & coords) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 aabb_min = make_float3 (coords.x, coords.y, coords.z);
	float3 aabb_max = aabb_min + make_float3 (1.f,
	                  1.f,
	                  1.f);
	node_has_overlap = AABBAABBIntersect (query_min, query_max,
	                                      aabb_min, aabb_max);
	return node_has_overlap;
}

__device__ bool HasCubeSEOverlap (const float3 & query,
                                  const float3 & morton_coords,
                                  float sq_se_size) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 aabb_min = morton_coords;
	float3 aabb_max = aabb_min + make_float3 (1.f, 1.f, 1.f);

	//	aabb_min = aabb_min - make_float3 (se_size, se_size, se_size);
	//	aabb_max = aabb_max + make_float3 (se_size, se_size, se_size);
	//
	//	node_has_overlap = (aabb_min.x < query.x) && (query.x < aabb_max.x);
	//	node_has_overlap = node_has_overlap &&
	//		((aabb_min.y < query.y) && (query.y < aabb_max.y));
	//	node_has_overlap = node_has_overlap &&
	//		((aabb_min.z < query.z) && (query.z < aabb_max.z));

	node_has_overlap = SphereAABBSqDistance (query, aabb_min, aabb_max) < (sq_se_size);
	return node_has_overlap;
}

//#define TEST_NO_ILP
#define TEST_ILP

__device__ bool HasCubeSEOverlap (const float3 & query,
                                  const ushort3 & coords,
                                  float sq_se_size) {
	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 aabb_min = make_float3 (coords.x, coords.y, coords.z);
	float3 aabb_max = aabb_min + make_float3 (1.f, 1.f, 1.f);

	//	aabb_min = aabb_min - make_float3 (se_size, se_size, se_size);
	//	aabb_max = aabb_max + make_float3 (se_size, se_size, se_size);
	//
	//	node_has_overlap = (aabb_min.x < query.x) && (query.x < aabb_max.x);
	//	node_has_overlap = node_has_overlap &&
	//		((aabb_min.y < query.y) && (query.y < aabb_max.y));
	//	node_has_overlap = node_has_overlap &&
	//		((aabb_min.z < query.z) && (query.z < aabb_max.z));

	node_has_overlap = SphereAABBSqDistance (query, aabb_min, aabb_max) < (sq_se_size);
	return node_has_overlap;
}

__device__ void HasCubeSEOverlap (const float3 & query,
                                  const float3 & morton_coords,
                                  float & sq_inner_se_size,
                                  float & sq_outer_se_size,
                                  bool & node_has_inner_overlap,
                                  bool & node_has_outer_overlap) {
	// Check if the node overlap the structuring element
	float3 aabb_min = morton_coords;
	float3 aabb_max = aabb_min + make_float3 (1.f, 1.f, 1.f);
	float sq_dist = SphereAABBSqDistance (query, aabb_min, aabb_max);
	node_has_inner_overlap = sq_dist < sq_inner_se_size;
	node_has_outer_overlap = sq_dist < sq_outer_se_size;
}

__device__ bool HasSphereSEOverlap (const float3 & query,
                                    const float3 & morton_coords,
                                    float sq_se_size) {
	//	bool node_has_overlap;
	// Check if the node overlap the structuring element
	float3 se_center = morton_coords + make_float3 (0.5f,
	                   0.5f,
	                   0.5f);
	return distanceS (se_center, query) < sq_se_size;
}

inline __device__ void ComputeNodeSE (const Node & node,
                                      int max_res,
                                      const float3 & min,
                                      const float3 & max,
                                      float3 & query_min,
                                      float3 & query_max) {
	float cell_size = max_res >> node.depth;
	query_min = (1.f / cell_size) * min;
	query_max = (1.f / cell_size) * max;
}

inline __device__ void ComputeNodeSE (int node_depth,
                                      int max_res,
                                      const float3 & min,
                                      const float3 & max,
                                      float3 & query_min,
                                      float3 & query_max) {
	float cell_size = max_res >> node_depth;
	query_min = (1.f / cell_size) * min;
	query_max = (1.f / cell_size) * max;
}

inline __device__ float3 Z8Coordinates (const float3 & base,
                                        unsigned int mask) {
	// (0x02 | 0x08 | 0x20 | 0x80)
	// (0x04 | 0x08 | 0x40 | 0x80)
	// (0x00 | 0xf0)
	return base +
	       make_float3 ((0xaa & mask) ? 1.f : 0.f,
	                    (0xcc & mask) ? 1.f : 0.f,
	                    (0xf0 & mask) ? 1.f : 0.f);
}

inline __device__ ushort3 Z8Coordinates (const ushort3 & base,
        unsigned int mask) {
	// (0x02 | 0x08 | 0x20 | 0x80)
	// (0x04 | 0x08 | 0x40 | 0x80)
	// (0x00 | 0xf0)
	ushort3 result;
	result.x = (0xaa & mask) ? 1 : 0;
	result.y = (0xcc & mask) ? 1 : 0;
	result.z = (0xf0 & mask) ? 1 : 0;

	result.x += base.x;
	result.y += base.y;
	result.z += base.z;

	return result;
}

inline __device__ float3 ComputeCoordinates (const float3 & base,
        int k) {
	unsigned int coords_x = 0x02 | 0x08 | 0x20 | 0x80;
	unsigned int coords_y = 0x04 | 0x08 | 0x40 | 0x80;
	unsigned int coords_z = 0x00 | 0xf0;
	unsigned int mask_k = (1 << k);

	return base +
	       make_float3 ((coords_x & mask_k) >> k, (coords_y & mask_k) >> k,
	                    (coords_z & mask_k) >> k);

}

__device__ void Push8SubTreesCube (Node & node, Node *& stack_ptr,
                                   const float3 & min,
                                   float radius,
                                   int max_depth,
                                   unsigned char * morton_mipmap) {
	// Find the 8 sub-trees that thightly intersect the Structuring Element
	int min_depth = max_depth - ceilf (log2f (2 * radius));
	min_depth--;
	float min_cell_size = (1 << (max_depth - min_depth));
	float3 se_min;
	se_min.x = floorf (min.x / min_cell_size);
	se_min.y = floorf (min.y / min_cell_size);
	se_min.z = floorf (min.z / min_cell_size);

	//	printf("2*radius : %f\n", 2*radius);
	//	printf("ceilf (..) : %f\n", ceilf (log2f (2*radius)));
	//	printf("min_depth : %i\n", min_depth);
	//	printf("min_cell_size : %f\n", min_cell_size);
	//	printf("se_min : %i %i %i\n", se_min.x, se_min.y, se_min.z);

	Node sub_tree = make_node (min_depth, 0);

	for (int k = 0; k < 8; k++) {
		float3 coords = ComputeCoordinates (se_min, k);
		sub_tree.idx = EncodeMorton3 (coords.x, coords.y, coords.z);
		bool has_geometry = ComputeNodeValue (sub_tree, morton_mipmap) != 0x00;
		if (has_geometry &&
		        coords.x >= 0 && coords.y >= 0 && coords.z >= 0
		        && coords.x < 512 && coords.y < 512 && coords.z < 512
		   ) {
			if (k != 7)
				*stack_ptr++ = sub_tree;
			node = sub_tree;
		}
	}
}

__global__ void ComputeCanonicalIndices (unsigned int * indices,
        unsigned int size) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		indices[idx] = idx;
}

struct IsNotMorphoCentroid {
	__host__ __device__ bool operator() (const unsigned int u) {
		return (u == 0xffffffff);
	}
};

__global__ void DilateBySphereStackless (GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= res || id_y >= res || id_z >= res)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * dilation_voxels = dilation_grid.voxels;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * input_mipmap = input_grid.morton_mipmap[1];

	if (input_voxels[i] == 0xff)
		return;

	float sq_radius = radius * radius;
	float3 center = make_float3 (2 * id_x, 2 * id_y, 2 * id_z);

	// Mipmap implicit traversal
	int max_mipmap_depth = input_grid.max_mipmap_depth;
	int node_depth = 1;
	unsigned int node_idx = 0;
	bool input_has_overlap = false;

	while (node_depth >= 1) {
		// Check if the node contains geometry
		unsigned char * mipmap = input_mipmap + ((1 << 3 * (node_depth - 1)) - 1) / 7;
		unsigned int mipmap_node_idx = node_idx >> 3;
		unsigned char mipmap_node_mask = 0x01 << (node_idx % 8);
		unsigned char mipmap_node_val = mipmap[mipmap_node_idx];
		bool node_has_geometry = (mipmap_node_val & mipmap_node_mask) != 0x00;
		bool node_is_leaf = (node_depth == max_mipmap_depth);
		bool node_has_overlap = false;

		if (node_has_geometry) {
			// Check if the node overlap the structuring element
			float3 aabb_min = make_float3 (DecodeMorton3X (node_idx),
			                               DecodeMorton3Y (node_idx),
			                               DecodeMorton3Z (node_idx));
			float3 aabb_max = aabb_min + make_float3 (1.f, 1.f, 1.f);
			int cell_size = 512 >> node_depth;
			aabb_min = cell_size * aabb_min;
			aabb_max = cell_size * aabb_max;
			node_has_overlap = SphereAABBIntersect (center, sq_radius,
			                                        aabb_min, aabb_max);
		}

		if (node_is_leaf) {
			if (node_has_overlap) {
				input_has_overlap = true;
				break;
			}
		} else if (node_has_overlap) {
			node_depth++;
			node_idx = node_idx << 3;
			continue;
		}

		node_idx++;
		int up = __clz (__brev (node_idx)) / 3;
		node_depth = node_depth - up;
		node_idx = node_idx >> 3 * up;
	}

	if (input_has_overlap)
		dilation_voxels[i] = 0xff;
}

__device__ unsigned int myAtomicOr (unsigned char * address, unsigned char val) {
	unsigned int * base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214,  0x3240,  0x3410,  0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, orOp, new_;

	old = *base_address;

	do {
		assumed = old;
		orOp = ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)) | val;
		new_ = __byte_perm(old, orOp, sel);

		if (new_ == old)
			break;

		old = atomicCAS(base_address, assumed, new_);

	} while (assumed != old);

	//	return old;
	return new_;
}

__device__ unsigned int myAtomicAnd (unsigned char * address, unsigned char val) {
	unsigned int * base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214,  0x3240,  0x3410,  0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, andOp, new_;

	old = *base_address;

	do {
		assumed = old;
		andOp = ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440)) & val;
		new_ = __byte_perm(old, andOp, sel);

		if (new_ == old)
			break;

		old = atomicCAS(base_address, assumed, new_);

	} while (assumed != old);

	//	return old;
	return new_;
}


__global__ void DilateByCubeByMipmap (GridGPU input_grid,
                                      GridGPU dilation_grid,
                                      float radius) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= 2 * data_res.x || id_y >= 2 * data_res.y || id_z >= 2 * data_res.z)
		return;

	unsigned int i = res * res * (id_z / 2) + res * (id_y / 2) + (id_x / 2);
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	radius /= (2.f * res);
	radius *= id_y;
	//	radius *= id_z;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	unsigned char * input_mipmap = input_grid.morton_mipmap[1];
	float3 * morton_coords = input_grid.morton_coords;
	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	Node stack[64];
	Node * stack_ptr = stack;
	*stack_ptr++ = make_node (-1, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	Node node = make_node (0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		int has_geometry_overlaps = ComputeNodeValue (node, input_mipmap);

		Node node_child = make_node (node.depth + 1, node.idx << 3);
		float3 base_node_coords = morton_coords[node_child.idx];
		float3 node_se_min, node_se_max;
		ComputeNodeSE (node_child, max_res, se_min, se_max, node_se_min, node_se_max);
		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_coords);

			if (node_has_overlap && node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*stack_ptr++ = node_child;
			} else if (node_has_overlap && node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*stack_ptr++ = make_node (-1, 0);
			}
			node_child.idx++;
		}
		// We pop the next node to process
		node = *--stack_ptr;
	} while (node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tidx = threadIdx.x % 2;
	unsigned char tidy = threadIdx.y % 2;
	unsigned char tidz = threadIdx.z % 2;
	unsigned char dilation_pos = 1 << (4 * tidz + 2 * tidy + tidx);

	// Shared Memory Initialization
	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
		shared_dilation_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                      blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)] = 0x00;
	}
	__syncthreads ();


	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                                   blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)], dilation_pos);

	__syncthreads ();

	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
		dilation_voxels[i] = shared_dilation_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                     blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)];
	}
}

#define DILATION_BASE_ILP2_BLOCK_SIZE 128

__global__ void DilateByCubeByTexMipmapBaseILP2 (GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	__shared__ bool local_overlap[8 * DILATION_BASE_ILP2_BLOCK_SIZE];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	//	radius /= (2.f*res);
	//	radius *= (2.f*id_y);
	//	radius += 2;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	const int tidx = blockDim.x * blockDim.y * threadIdx.z +
	                 blockDim.x * threadIdx.y + threadIdx.x;
	//	const int tidx = 0;
	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);
		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (base_tex_node_coords.x,
		                               base_tex_node_coords.y,
		                               base_tex_node_coords.z);
		//		bool local_overlap[8];

		//		overlap[12*tidx + 0] = node_se_max.x >= (b_coords.x);
		//		overlap[12*tidx + 1] = node_se_max.x >= (b_coords.x + 1.f);
		//		overlap[12*tidx + 2] = node_se_max.y >= (b_coords.y);
		//		overlap[12*tidx + 3] = node_se_max.y >= (b_coords.y + 1.f);
		//		overlap[12*tidx + 4] = node_se_max.z >= (b_coords.z);
		//		overlap[12*tidx + 5] = node_se_max.z >= (b_coords.z + 1.f);
		//		overlap[12*tidx + 6] = node_se_min.x <= (b_coords.x + 1.f);
		//		overlap[12*tidx + 7] = node_se_min.x <= (b_coords.x + 2.f);
		//		overlap[12*tidx + 8] = node_se_min.y <= (b_coords.y + 1.f);
		//		overlap[12*tidx + 9] = node_se_min.y <= (b_coords.y + 2.f);
		//		overlap[12*tidx + 10] = node_se_min.z <= (b_coords.z + 1.f);
		//		overlap[12*tidx + 11] = node_se_min.z <= (b_coords.z + 2.f);

		local_overlap[8 * tidx + 0] = (node_se_max.x >= (b_coords.x))
		                              && (node_se_max.y >= (b_coords.y))
		                              && (node_se_max.z >= (b_coords.z))
		                              && (node_se_min.x <= (b_coords.x + 1.f))
		                              && (node_se_min.y <= (b_coords.y + 1.f))
		                              && (node_se_min.z <= (b_coords.z + 1.f));

		local_overlap[8 * tidx + 1] = (node_se_max.x >= (b_coords.x + 1.f))
		                              && (node_se_max.y >= (b_coords.y))
		                              && (node_se_max.z >= (b_coords.z))
		                              && (node_se_min.x <= (b_coords.x + 2.f))
		                              && (node_se_min.y <= (b_coords.y + 1.f))
		                              && (node_se_min.z <= (b_coords.z + 1.f));

		local_overlap[8 * tidx + 2] = (node_se_max.x >= (b_coords.x))
		                              && (node_se_max.y >= (b_coords.y + 1.f))
		                              && (node_se_max.z >= (b_coords.z))
		                              && (node_se_min.x <= (b_coords.x + 1.f))
		                              && (node_se_min.y <= (b_coords.y + 2.f))
		                              && (node_se_min.z <= (b_coords.z + 1.f));

		local_overlap[8 * tidx + 3] = (node_se_max.x >= (b_coords.x + 1.f))
		                              && (node_se_max.y >= (b_coords.y + 1.f))
		                              && (node_se_max.z >= (b_coords.z))
		                              && (node_se_min.x <= (b_coords.x + 2.f))
		                              && (node_se_min.y <= (b_coords.y + 2.f))
		                              && (node_se_min.z <= (b_coords.z + 1.f));

		local_overlap[8 * tidx + 4] = (node_se_max.x >= (b_coords.x))
		                              && (node_se_max.y >= (b_coords.y))
		                              && (node_se_max.z >= (b_coords.z + 1.f))
		                              && (node_se_min.x <= (b_coords.x + 1.f))
		                              && (node_se_min.y <= (b_coords.y + 1.f))
		                              && (node_se_min.z <= (b_coords.z + 2.f));

		local_overlap[8 * tidx + 5] = (node_se_max.x >= (b_coords.x + 1.f))
		                              && (node_se_max.y >= (b_coords.y))
		                              && (node_se_max.z >= (b_coords.z + 1.f))
		                              && (node_se_min.x <= (b_coords.x + 2.f))
		                              && (node_se_min.y <= (b_coords.y + 1.f))
		                              && (node_se_min.z <= (b_coords.z + 2.f));

		local_overlap[8 * tidx + 6] = (node_se_max.x >= (b_coords.x))
		                              && (node_se_max.y >= (b_coords.y + 1.f))
		                              && (node_se_max.z >= (b_coords.z + 1.f))
		                              && (node_se_min.x <= (b_coords.x + 1.f))
		                              && (node_se_min.y <= (b_coords.y + 2.f))
		                              && (node_se_min.z <= (b_coords.z + 2.f));

		local_overlap[8 * tidx + 7] = (node_se_max.x >= (b_coords.x + 1.f))
		                              && (node_se_max.y >= (b_coords.y + 1.f))
		                              && (node_se_max.z >= (b_coords.z + 1.f))
		                              && (node_se_min.x <= (b_coords.x + 2.f))
		                              && (node_se_min.y <= (b_coords.y + 2.f))
		                              && (node_se_min.z <= (b_coords.z + 2.f));

		//		local_overlap[0] =
		//			overlap[12*tidx + 0] && overlap[12*tidx + 2] && overlap[12*tidx + 4] &&
		//			overlap[12*tidx + 6] && overlap[12*tidx + 8] && overlap[12*tidx + 10];
		//
		//		local_overlap[1] =
		//			overlap[12*tidx + 1] && overlap[12*tidx + 2] && overlap[12*tidx + 4] &&
		//			overlap[12*tidx + 7] && overlap[12*tidx + 8] && overlap[12*tidx + 10];
		//
		//		local_overlap[2] =
		//			overlap[12*tidx + 0] && overlap[12*tidx + 3] && overlap[12*tidx + 4] &&
		//			overlap[12*tidx + 6] && overlap[12*tidx + 9] && overlap[12*tidx + 10];
		//
		//		local_overlap[2] =
		//			overlap[12*tidx + 1] && overlap[12*tidx + 3] && overlap[12*tidx + 4] &&
		//			overlap[12*tidx + 7] && overlap[12*tidx + 9] && overlap[12*tidx + 10];
		//
		//		local_overlap[3] =
		//			overlap[12*tidx + 0] && overlap[12*tidx + 2] && overlap[12*tidx + 5] &&
		//			overlap[12*tidx + 6] && overlap[12*tidx + 8] && overlap[12*tidx + 11];
		//
		//		local_overlap[4] =
		//			overlap[12*tidx + 1] && overlap[12*tidx + 2] && overlap[12*tidx + 5] &&
		//			overlap[12*tidx + 7] && overlap[12*tidx + 8] && overlap[12*tidx + 11];
		//
		//		local_overlap[5] =
		//			overlap[12*tidx + 0] && overlap[12*tidx + 3] && overlap[12*tidx + 5] &&
		//			overlap[12*tidx + 6] && overlap[12*tidx + 9] && overlap[12*tidx + 11];
		//
		//		local_overlap[6] =
		//			overlap[12*tidx + 1] && overlap[12*tidx + 3] && overlap[12*tidx + 5] &&
		//			overlap[12*tidx + 7] && overlap[12*tidx + 9] && overlap[12*tidx + 11];

		if (local_overlap[8 * tidx + 0]
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 1]
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 2]
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 3]
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 4]
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 5]
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 6]
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (local_overlap[8 * tidx + 7]
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		//		overlap[8*tidx + 0] = 0xc0;
		//		overlap[8*tidx + 1] = 0xc0;
		//		overlap[8*tidx + 2] = 0xc0;
		//		overlap[8*tidx + 3] = 0xc0;
		//		overlap[8*tidx + 4] = 0xc0;
		//		overlap[8*tidx + 5] = 0xc0;
		//		overlap[8*tidx + 6] = 0xc0;
		//		overlap[8*tidx + 7] = 0xc0;
		//
		//		overlap[8*tidx + 0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[8*tidx + 1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[8*tidx + 2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[8*tidx + 3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[8*tidx + 4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[8*tidx + 5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[8*tidx + 6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[8*tidx + 7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//
		//		overlap[8*tidx + 0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[8*tidx + 1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[8*tidx + 2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[8*tidx + 3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[8*tidx + 4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[8*tidx + 5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[8*tidx + 6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[8*tidx + 7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//
		//		overlap[8*tidx + 0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[8*tidx + 1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[8*tidx + 2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[8*tidx + 3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[8*tidx + 4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[8*tidx + 5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[8*tidx + 6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[8*tidx + 7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//
		//		overlap[8*tidx + 0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[8*tidx + 1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[8*tidx + 2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[8*tidx + 3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[8*tidx + 4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[8*tidx + 5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[8*tidx + 6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[8*tidx + 7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//
		//		overlap[8*tidx + 0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[8*tidx + 1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[8*tidx + 2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[8*tidx + 3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[8*tidx + 4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[8*tidx + 5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[8*tidx + 6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[8*tidx + 7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//
		//		overlap[8*tidx + 0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//		overlap[8*tidx + 1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//		overlap[8*tidx + 2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//		overlap[8*tidx + 3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//		overlap[8*tidx + 4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//		overlap[8*tidx + 5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//		overlap[8*tidx + 6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//		overlap[8*tidx + 7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;


		//		overlap[0] = 0xc0;
		//		overlap[0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[1] = 0xc0;
		//		overlap[1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[2] = 0xc0;
		//		overlap[2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[3] = 0xc0;
		//		overlap[3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[4] = 0xc0;
		//		overlap[4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[5] = 0xc0;
		//		overlap[5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;


		//		overlap[6] = 0xc0;
		//		overlap[6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[7] = 0xc0;
		//		overlap[7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;


		//		if (overlap[8*tidx + 0] == 0xff
		//				&& ((has_geometry_overlaps & (1)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x,
		//																			base_tex_node_coords.y,
		//																			base_tex_node_coords.z);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 1] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 1)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x + 1,
		//																			base_tex_node_coords.y,
		//																			base_tex_node_coords.z);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 2] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 2)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x,
		//																			base_tex_node_coords.y + 1,
		//																			base_tex_node_coords.z);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 3] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 3)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x + 1,
		//																			base_tex_node_coords.y + 1,
		//																			base_tex_node_coords.z);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 4] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 4)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x,
		//																			base_tex_node_coords.y,
		//																			base_tex_node_coords.z + 1);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 5] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 5)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x + 1,
		//																			base_tex_node_coords.y,
		//																			base_tex_node_coords.z + 1);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 6] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 6)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x,
		//																			base_tex_node_coords.y + 1,
		//																			base_tex_node_coords.z + 1);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}
		//
		//		if (overlap[8*tidx + 7] == 0xff
		//				&& ((has_geometry_overlaps & (1 << 7)) != 0)) {
		//			if ((tex_node.depth + 1) != max_depth) {
		//				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
		//																			base_tex_node_coords.x + 1,
		//																			base_tex_node_coords.y + 1,
		//																			base_tex_node_coords.z + 1);
		//			} else {
		//				dilation_occup = 0xff;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//			}
		//		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}



#define DILATION_BASE_ILP1_BLOCK_SIZE 128

__global__ void DilateByCubeByTexMipmapBaseILP1 (GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	__shared__ unsigned char overlap[8 * DILATION_BASE_ILP1_BLOCK_SIZE];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	const int tidx = blockDim.x * blockDim.y * threadIdx.z +
	                 blockDim.x * threadIdx.y + threadIdx.x;
	//	const int tidx = 0;
	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);
		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (base_tex_node_coords.x,
		                               base_tex_node_coords.y,
		                               base_tex_node_coords.z);
		overlap[8 * tidx + 0] = 0xc0;
		overlap[8 * tidx + 1] = 0xc0;
		overlap[8 * tidx + 2] = 0xc0;
		overlap[8 * tidx + 3] = 0xc0;
		overlap[8 * tidx + 4] = 0xc0;
		overlap[8 * tidx + 5] = 0xc0;
		overlap[8 * tidx + 6] = 0xc0;
		overlap[8 * tidx + 7] = 0xc0;

		overlap[8 * tidx + 0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;

		overlap[8 * tidx + 0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;

		overlap[8 * tidx + 0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;

		overlap[8 * tidx + 0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;

		overlap[8 * tidx + 0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;

		overlap[8 * tidx + 0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;


		//		overlap[0] = 0xc0;
		//		overlap[0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[1] = 0xc0;
		//		overlap[1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[2] = 0xc0;
		//		overlap[2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[3] = 0xc0;
		//		overlap[3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[4] = 0xc0;
		//		overlap[4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[5] = 0xc0;
		//		overlap[5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[6] = 0xc0;
		//		overlap[6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[7] = 0xc0;
		//		overlap[7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;

		if (overlap[8 * tidx + 0] == 0xff
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 1] == 0xff
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 2] == 0xff
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 3] == 0xff
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 4] == 0xff
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 5] == 0xff
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 6] == 0xff
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 7] == 0xff
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__device__ float4 InterpolateQuaternionField (const float4 * quat_field,
        unsigned int quat_field_res,
        unsigned int id_x,
        unsigned int id_y,
        unsigned int id_z,
        unsigned int res,
        const float4 group_quats[],
        int group_quats_size) {

	float4 quat_interpol;
	float4 q000, q100, q110, q010, q001, q101, q111, q011;

	//	int scale_res = res/quat_field_res;

	//	int id_qf = (id_x/scale_res) + (id_y/scale_res)*(quat_field_res)
	//		+ (id_z/scale_res)*(quat_field_res*quat_field_res);

	float3 id3_normalized = make_float3 (((float)id_x) / ((float) res),
	                                     ((float)id_y) / ((float) res),
	                                     ((float)id_z) / ((float) res));

	float3 id3f_qf = make_float3 (((float)quat_field_res) * (id3_normalized.x),
	                              ((float)quat_field_res) * (id3_normalized.y),
	                              ((float)quat_field_res) * (id3_normalized.z));

	int3 id3_qf = make_int3 (__float2int_rz (id3f_qf.x), __float2int_rz (id3f_qf.y), __float2int_rz (id3f_qf.z));

	int id_qf = id3_qf.x + quat_field_res * id3_qf.y + quat_field_res * quat_field_res * id3_qf.z;

	int off_x_qf = 1, off_y_qf = quat_field_res, off_z_qf = quat_field_res * quat_field_res;

	q000 = quat_field [id_qf];
	q100 = quat_field [id_qf + off_x_qf];
	q110 = quat_field [id_qf + off_x_qf + off_y_qf];
	q010 = quat_field [id_qf + off_y_qf];
	q001 = quat_field [id_qf + off_z_qf];
	q101 = quat_field [id_qf + off_x_qf + off_z_qf];
	q111 = quat_field [id_qf + off_x_qf + off_y_qf + off_z_qf];
	q011 = quat_field [id_qf + off_y_qf + off_z_qf];

	float4 q00, q01, q10, q11, q0, q1;
	//	float tx = ((float)(id_x%scale_res))/((float) scale_res);
	//	float ty = ((float)(id_y%scale_res))/((float) scale_res);
	//	float tz = ((float)(id_z%scale_res))/((float) scale_res);

	float tx = id3f_qf.x - ((float)id3_qf.x);
	float ty = id3f_qf.y - ((float)id3_qf.y);
	float tz = id3f_qf.z - ((float)id3_qf.z);

	q00 = interpolate_cube_group (q000, q100, tx, gcube_quats, group_quats_size);
	q01 = interpolate_cube_group (q001, q101, tx, gcube_quats, group_quats_size);
	q10 = interpolate_cube_group (q010, q110, tx, gcube_quats, group_quats_size);
	q11 = interpolate_cube_group (q011, q111, tx, gcube_quats, group_quats_size);

	//	q00 = normalize (q00);
	//	q01 = normalize (q01);
	//	q10 = normalize (q10);
	//	q11 = normalize (q11);

	q0 = interpolate_cube_group (q00, q10, ty, gcube_quats, group_quats_size);
	q1 = interpolate_cube_group (q01, q11, ty, gcube_quats, group_quats_size);

	//	q0 = normalize (q0);
	//	q1 = normalize (q1);

	quat_interpol = interpolate_cube_group (q0, q1, tz,
	                                        gcube_quats, group_quats_size);

	//	float4 gcube_quats0 = make_float4 (0, 0, 0, 1);
	//	quat_interpol = interpolate_cube_group (q000, q100, 0.1f, gcube_quats);
	//	quat_interpol = hamilton (q000, gcube_quats0) + hamilton (q100, gcube_quats0);
	//	quat_interpol = hamilton (q000, group_quats[0]) + hamilton (q100, group_quats[0]);
	//	quat_interpol = q000 + q100;

	//	quat_interpol = normalize (quat_interpol);

	//	quat_interpol = (1.f/8.f)*(q000 + q100 + q110 + q010 + q001 + q101 + q111 + q011);

	return quat_interpol;
}

__global__ void DilateByRotCubeByTexMipmapBase (GridGPU input_grid,
        float4 * quat_field,
        uint3 quat_field_res,
        GridGPU dilation_grid) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	// Rotated Cube from quaternion field
	float4 quat = InterpolateQuaternionField (quat_field, quat_field_res.x,
	              id_x, id_y, id_z, res, gcube_quats,
	              0);
	//	float4 quat = make_float4 (0.290464, -0.0938969, 0.00821934, 0.952232);
	//	float4 quat = make_float4 (0.167596, 0.167596, 0.167596, 0.95694);
	//	float4 quat = make_float4 (0.f, 0.f, 0.f, 0.99999f);
	quat = normalize (conj (quat));
	//	quat = normalize (quat);

	//	float3 se_center = make_float3 (0.5f*(se_max.x - se_min.x) + se_min.x,
	//																	0.5f*(se_max.y - se_min.y) + se_min.y,
	//																	0.5f*(se_max.z - se_min.z) + se_min.z);
	float3 se_center = make_float3 (2 * id_x, 2 * id_y, 2 * id_z);
	float3 b_x = rotate (quat, make_float3 (1.f, 0.f, 0.f));
	float3 b_y = rotate (quat, make_float3 (0.f, 1.f, 0.f));
	float3 b_z = rotate (quat, make_float3 (0.f, 0.f, 1.f));


	//	radius /= (2.f*res);
	//	radius *= (2.f*id_y);
	//	radius += 2;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	float radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	//	float3 se_min = make_float3 (2*id_x - radius, 2*id_y - radius, 2*id_z - radius);
	//	float3 se_max = make_float3 (2*id_x + radius, 2*id_y + radius, 2*id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		//		float3 node_se_min, node_se_max;
		//		ComputeNodeSE (tex_node.depth + 1,
		//									 max_res, se_min, se_max, node_se_min, node_se_max);
		//		node_se_min = se_min;
		//		node_se_max = se_max;
		float node_cell_size = max_res >> (tex_node.depth + 1);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			//			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			//				HasCubeSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasRotCubeSEOverlap (se_center, b_x, b_y, b_z, child_tex_node_coords,
			                                node_cell_size, radius);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__global__ void DilateByCubeByTexMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	//	radius /= (2.f*res);
	//	radius *= (2.f*id_y);
	//	radius += 2;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	float radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__global__ void DilateByCubeByMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	//	radius /= (2.f*res);
	//	radius *= (2.f*id_y);
	//	radius += 2;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	unsigned char * input_mipmap = input_grid.morton_mipmap[1];
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	Node stack[64];
	Node * stack_ptr = stack;
	*stack_ptr++ = make_node (-1, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	Node node;
	//	Push8SubTreesCube (node, stack_ptr, se_min, radius, max_depth,
	//											 input_mipmap);
	node = make_node (0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		int has_geometry_overlaps = ComputeNodeValue (node, input_mipmap);

		Node node_child = make_node (node.depth + 1, node.idx << 3);
		float3 base_node_coords = make_float3 (DecodeMorton3X (node_child.idx),
		                                       DecodeMorton3Y (node_child.idx),
		                                       DecodeMorton3Z (node_child.idx));
		float3 node_se_min, node_se_max;
		ComputeNodeSE (node_child, max_res, se_min, se_max, node_se_min, node_se_max);
		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_coords);

			if (node_has_overlap && node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*stack_ptr++ = node_child;
			} else if (node_has_overlap && node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*stack_ptr++ = make_node (-1, 0);
				//				return;
			}
			node_child.idx++;
		}
		// We pop the next node to process
		node = *--stack_ptr;
	} while (node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__global__ void DilateByCubeByTexMipmap (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	//	radius /= (2.f*res);
	//	radius *= id_y;
	//	radius += 2;
	unsigned char uchar_radius;
	unsigned long long mask_size = 0x7f;
	uchar_radius = mask_size & (data >> (7 * (threadIdx.x % 8)));
	radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char dilation_fine = (1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_dilation_occup[threadIdx.x / 8] = 0x00;
	}
	__syncthreads ();

	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[threadIdx.x / 8], dilation_fine);
	__syncthreads ();

	if (tid == 0) {
		dilation_voxels[i] = shared_dilation_occup[threadIdx.x / 8];
	}
}

__global__ void DilateByCubeByTexMipmapBaseILP1 (GridGPU input_grid,
        GridGPU dilation_grid) {
	__shared__ unsigned char overlap[8 * DILATION_BASE_ILP1_BLOCK_SIZE];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	float radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	const int tidx = blockDim.x * blockDim.y * threadIdx.z +
	                 blockDim.x * threadIdx.y + threadIdx.x;
	//	const int tidx = 0;
	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);
		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (base_tex_node_coords.x,
		                               base_tex_node_coords.y,
		                               base_tex_node_coords.z);
		overlap[8 * tidx + 0] = 0xc0;
		overlap[8 * tidx + 1] = 0xc0;
		overlap[8 * tidx + 2] = 0xc0;
		overlap[8 * tidx + 3] = 0xc0;
		overlap[8 * tidx + 4] = 0xc0;
		overlap[8 * tidx + 5] = 0xc0;
		overlap[8 * tidx + 6] = 0xc0;
		overlap[8 * tidx + 7] = 0xc0;

		overlap[8 * tidx + 0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		overlap[8 * tidx + 6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		overlap[8 * tidx + 7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;

		overlap[8 * tidx + 0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		overlap[8 * tidx + 6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		overlap[8 * tidx + 7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;

		overlap[8 * tidx + 0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		overlap[8 * tidx + 4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		overlap[8 * tidx + 7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;

		overlap[8 * tidx + 0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		overlap[8 * tidx + 6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		overlap[8 * tidx + 7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;

		overlap[8 * tidx + 0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		overlap[8 * tidx + 6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		overlap[8 * tidx + 7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;

		overlap[8 * tidx + 0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		overlap[8 * tidx + 4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		overlap[8 * tidx + 7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;


		//		overlap[0] = 0xc0;
		//		overlap[0] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[0] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[0] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[0] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[0] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[0] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[1] = 0xc0;
		//		overlap[1] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[1] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[1] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[1] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[1] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[1] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[2] = 0xc0;
		//		overlap[2] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[2] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[2] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[2] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[2] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[2] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[3] = 0xc0;
		//		overlap[3] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[3] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[3] |= node_se_max.z >= (b_coords.z) ? C2 : 0;
		//		overlap[3] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[3] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[3] |= node_se_min.z <= (b_coords.z + 1.f) ? C5 : 0;
		//
		//		overlap[4] = 0xc0;
		//		overlap[4] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[4] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[4] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[4] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[4] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[4] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[5] = 0xc0;
		//		overlap[5] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[5] |= node_se_max.y >= (b_coords.y) ? C1 : 0;
		//		overlap[5] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[5] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[5] |= node_se_min.y <= (b_coords.y + 1.f) ? C4 : 0;
		//		overlap[5] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[6] = 0xc0;
		//		overlap[6] |= node_se_max.x >= (b_coords.x) ? C0 : 0;
		//		overlap[6] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[6] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[6] |= node_se_min.x <= (b_coords.x + 1.f) ? C3 : 0;
		//		overlap[6] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[6] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;
		//
		//		overlap[7] = 0xc0;
		//		overlap[7] |= node_se_max.x >= (b_coords.x + 1.f) ? C0 : 0;
		//		overlap[7] |= node_se_max.y >= (b_coords.y + 1.f) ? C1 : 0;
		//		overlap[7] |= node_se_max.z >= (b_coords.z + 1.f) ? C2 : 0;
		//		overlap[7] |= node_se_min.x <= (b_coords.x + 2.f) ? C3 : 0;
		//		overlap[7] |= node_se_min.y <= (b_coords.y + 2.f) ? C4 : 0;
		//		overlap[7] |= node_se_min.z <= (b_coords.z + 2.f) ? C5 : 0;

		if (overlap[8 * tidx + 0] == 0xff
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 1] == 0xff
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 2] == 0xff
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 3] == 0xff
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 4] == 0xff
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 5] == 0xff
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 6] == 0xff
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (overlap[8 * tidx + 7] == 0xff
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__global__ void DilateByRotCubeByTexMipmap (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        float4 * quat_field,
        uint3 quat_field_res,
        GridGPU dilation_grid) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;
	unsigned char uchar_radius;
	unsigned long long mask_size = 0x7f;
	uchar_radius = mask_size & (data >> (7 * (threadIdx.x % 8)));
	float radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	//	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	//	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Rotated Cube from quaternion field
	float4 quat = InterpolateQuaternionField (quat_field, quat_field_res.x,
	              id_x, id_y, id_z, 2 * res, gcube_quats,
	              0);
	//	float4 quat = make_float4 (0.290464, -0.0938969, 0.00821934, 0.952232);
	//	float4 quat = make_float4 (0.167596, 0.167596, 0.167596, 0.95694);
	//	float4 quat = make_float4 (0.f, 0.f, 0.f, 0.99999f);
	quat = normalize (conj (quat));
	//	quat = normalize (quat);

	//	float3 se_center = make_float3 (0.5f*(se_max.x - se_min.x) + se_min.x,
	//																	0.5f*(se_max.y - se_min.y) + se_min.y,
	//																	0.5f*(se_max.z - se_min.z) + se_min.z);
	float3 se_center = make_float3 (id_x, id_y, id_z);
	float3 b_x = rotate (quat, make_float3 (1.f, 0.f, 0.f));
	float3 b_y = rotate (quat, make_float3 (0.f, 1.f, 0.f));
	float3 b_z = rotate (quat, make_float3 (0.f, 0.f, 1.f));

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float node_cell_size = max_res >> (tex_node.depth + 1);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasRotCubeSEOverlap (se_center, b_x, b_y, b_z, child_tex_node_coords,
			                                node_cell_size, radius);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char dilation_fine = (1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_dilation_occup[threadIdx.x / 8] = 0x00;
	}
	__syncthreads ();

	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[threadIdx.x / 8], dilation_fine);
	__syncthreads ();

	if (tid == 0) {
		dilation_voxels[i] = shared_dilation_occup[threadIdx.x / 8];
	}
}

__global__ void DilateByCubeByTexMipmap (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU dilation_grid) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;
	unsigned char uchar_radius;
	unsigned long long mask_size = 0x7f;
	uchar_radius = mask_size & (data >> (7 * (threadIdx.x % 8)));
	float radius = uchar_radius;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char dilation_fine = (1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_dilation_occup[threadIdx.x / 8] = 0x00;
	}
	__syncthreads ();

	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[threadIdx.x / 8], dilation_fine);
	__syncthreads ();

	if (tid == 0) {
		dilation_voxels[i] = shared_dilation_occup[threadIdx.x / 8];
	}
}

__global__ void DilateByCubeByMipmap (unsigned int * cells,
                                      unsigned int cells_size,
                                      GridGPU input_grid,
                                      GridGPU dilation_grid,
                                      float radius) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	//	radius /= (2.f*res);
	//	radius *= id_y;
	//	radius += 2;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	unsigned char * input_mipmap = input_grid.morton_mipmap[1];
	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	Node stack[64];
	Node * stack_ptr = stack;
	*stack_ptr++ = make_node (-1, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	Node node = make_node (0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		int has_geometry_overlaps = ComputeNodeValue (node, input_mipmap);

		Node node_child = make_node (node.depth + 1, node.idx << 3);
		float3 base_node_coords = make_float3 (DecodeMorton3X (node_child.idx),
		                                       DecodeMorton3Y (node_child.idx),
		                                       DecodeMorton3Z (node_child.idx));
		float3 node_se_min, node_se_max;
		ComputeNodeSE (node_child, max_res, se_min, se_max, node_se_min, node_se_max);
		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_se_min, node_se_max, child_coords);

			if (node_has_overlap && node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*stack_ptr++ = node_child;
			} else if (node_has_overlap && node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*stack_ptr++ = make_node (-1, 0);
				//				return;
			}
			node_child.idx++;
		}
		// We pop the next node to process
		node = *--stack_ptr;
	} while (node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char dilation_fine = (1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_dilation_occup[threadIdx.x / 8] = 0x00;
	}
	__syncthreads ();

	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[threadIdx.x / 8], dilation_fine);
	__syncthreads ();

	if (tid == 0) {
		dilation_voxels[i] = shared_dilation_occup[threadIdx.x / 8];
	}
}

__global__ void DilateBySphereByTexMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = dilation_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;

	if (input_voxels[i] == 0xff)
		return;

	//	radius /= (2.f*res);
	//	radius *= (2.f*id_y);
	//	radius += 2;

	unsigned long long mask_size = 0x7f;
	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	unsigned char uchar_radius;
	uchar_radius = mask_size & data;
	radius = uchar_radius;

	// Safety Measure
	radius++;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (2 * id_x - radius, 2 * id_y - radius, 2 * id_z - radius);
	float3 se_max = make_float3 (2 * id_x + radius, 2 * id_y + radius, 2 * id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasSphereSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (dilation_occup != 0x00)
		dilation_voxels[i] = dilation_occup;
}

__global__ void DilateBySphereByTexMipmap (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU dilation_grid,
        float radius) {
	extern __shared__ unsigned char shared_dilation_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
	                                  (float)id_x / res,
	                                  (float)id_y / res,
	                                  (float)id_z / res,
	                                  0);
	unsigned long long data;
	data = *reinterpret_cast<unsigned long long*>(&data_tex);

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	//	radius /= (2.f*res);
	//	radius *= id_y;
	//	radius += 2;
	unsigned char uchar_radius;
	unsigned long long mask_size = 0x7f;
	uchar_radius = mask_size & (data >> (7 * (threadIdx.x % 8)));
	radius = uchar_radius;

	// Safety Measure
	radius++;

	unsigned char * dilation_voxels = dilation_grid.voxels;
	float3 se_min = make_float3 (id_x - radius, id_y - radius, id_z - radius);
	float3 se_max = make_float3 (id_x + radius, id_y + radius, id_z + radius);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char dilation_occup = 0x00;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                            (float)tex_node.id.x / curr_res,
		                            (float)tex_node.id.y / curr_res,
		                            (float)tex_node.id.z / curr_res,
		                            max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		data = *reinterpret_cast<unsigned long long*>(&data_tex);

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 node_se_min, node_se_max;
		ComputeNodeSE (tex_node.depth + 1,
		               max_res, se_min, se_max, node_se_min, node_se_max);

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);

			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasSphereSEOverlap (node_se_min, node_se_max, child_tex_node_coords);

			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				dilation_occup = 0xff;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char dilation_fine = (1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_dilation_occup[threadIdx.x / 8] = 0x00;
	}
	__syncthreads ();

	if (dilation_occup != 0x00)
		myAtomicOr (&shared_dilation_occup[threadIdx.x / 8], dilation_fine);
	__syncthreads ();

	if (tid == 0) {
		dilation_voxels[i] = shared_dilation_occup[threadIdx.x / 8];
	}
}

void GMorpho::DilateByRotCubeMipmap (const ScaleField & scale_field,
                                     const FrameField & frame_field) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;
	ShowGPUMemoryUsage ();
#endif

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (scale_field);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);


	std::vector<float4> gcube_quats_vec_ (24);

	for (int i = 0; i < 24; i++)
		gcube_quats_vec_[i]  = make_float4 ( 0,      0,      0,      1);

	// Beam symmetry group
	gcube_quats_vec_[0]  = make_float4 ( 0,      0,      0,      1);
	gcube_quats_vec_[1]  = make_float4 ( 1,      0,      0,      0);
	gcube_quats_vec_[2]  = make_float4 ( 0,      1,      0,      0);
	gcube_quats_vec_[3]  = make_float4 ( 0,      0,      1,      0);
	gcube_quats_vec_[4] = make_float4 ( dCPI4,  dCPI4,  0,      0);
	gcube_quats_vec_[5] = make_float4 ( 0,      0,      dCPI4,  dCPI4);
	gcube_quats_vec_[6] = make_float4 ( 0,      0,     -dCPI4,  dCPI4);
	gcube_quats_vec_[7] = make_float4 (-dCPI4,  dCPI4,  0,      0);

	// Cube symmetry group
	gcube_quats_vec_[8]  = make_float4 ( dCPI4,  0,      0,      dCPI4);
	gcube_quats_vec_[9]  = make_float4 (-dCPI4,  0,      0,      dCPI4);
	gcube_quats_vec_[10]  = make_float4 ( 0,      dCPI4,  dCPI4,  0);
	gcube_quats_vec_[11]  = make_float4 ( 0,     -dCPI4,  dCPI4,  0);

	gcube_quats_vec_[12]  = make_float4 ( 0.5,    0.5,    0.5,    0.5);
	gcube_quats_vec_[13]  = make_float4 (-0.5,   -0.5,    0.5,    0.5);
	gcube_quats_vec_[14] = make_float4 (-0.5,    0.5,    0.5,   -0.5);
	gcube_quats_vec_[15] = make_float4 ( 0.5,   -0.5,    0.5,   -0.5);

	gcube_quats_vec_[16] = make_float4 ( 0.5,    0.5,    0.5,   -0.5);
	gcube_quats_vec_[17] = make_float4 ( 0.5,   -0.5,    0.5,    0.5);
	gcube_quats_vec_[18] = make_float4 ( 0,     -dCPI4,  0,      dCPI4);
	gcube_quats_vec_[19] = make_float4 ( dCPI4,  0,      dCPI4,  0);
	gcube_quats_vec_[20] = make_float4 (-0.5,   -0.5,    0.5,   -0.5);
	gcube_quats_vec_[21] = make_float4 (-0.5,    0.5,    0.5,    0.5);
	gcube_quats_vec_[22] = make_float4 ( 0,      dCPI4,  0,  dCPI4);
	gcube_quats_vec_[23] = make_float4 (-dCPI4,  0,      dCPI4,  0);


	//	gcube_quats_vec_[0]  = make_float4 ( 0,      0,      0,      1);
	//	gcube_quats_vec_[1]  = make_float4 ( 1,      0,      0,      0);
	////	gcube_quats_vec_[2]  = make_float4 ( dCPI4,  0,      0,      dCPI4);
	////	gcube_quats_vec_[3]  = make_float4 (-dCPI4,  0,      0,      dCPI4);
	//	gcube_quats_vec_[4]  = make_float4 ( 0,      1,      0,      0);
	//	gcube_quats_vec_[5]  = make_float4 ( 0,      0,      1,      0);
	////	gcube_quats_vec_[6]  = make_float4 ( 0,      dCPI4,  dCPI4,  0);
	////	gcube_quats_vec_[7]  = make_float4 ( 0,     -dCPI4,  dCPI4,  0);
	////
	////	gcube_quats_vec_[8]  = make_float4 ( 0.5,    0.5,    0.5,    0.5);
	////	gcube_quats_vec_[9]  = make_float4 (-0.5,   -0.5,    0.5,    0.5);
	//	gcube_quats_vec_[10] = make_float4 ( dCPI4,  dCPI4,  0,      0);
	//	gcube_quats_vec_[11] = make_float4 ( 0,      0,      dCPI4,  dCPI4);
	////	gcube_quats_vec_[12] = make_float4 (-0.5,    0.5,    0.5,   -0.5);
	////	gcube_quats_vec_[13] = make_float4 ( 0.5,   -0.5,    0.5,   -0.5);
	//	gcube_quats_vec_[14] = make_float4 ( 0,      0,     -dCPI4,  dCPI4);
	//	gcube_quats_vec_[15] = make_float4 (-dCPI4,  dCPI4,  0,      0);
	//
	////	gcube_quats_vec_[16] = make_float4 ( 0.5,    0.5,    0.5,   -0.5);
	////	gcube_quats_vec_[17] = make_float4 ( 0.5,   -0.5,    0.5,    0.5);
	////	gcube_quats_vec_[18] = make_float4 ( 0,     -dCPI4,  0,      dCPI4);
	////	gcube_quats_vec_[19] = make_float4 ( dCPI4,  0,      dCPI4,  0);
	////	gcube_quats_vec_[20] = make_float4 (-0.5,   -0.5,    0.5,   -0.5);
	////	gcube_quats_vec_[21] = make_float4 (-0.5,    0.5,    0.5,    0.5);
	////	gcube_quats_vec_[22] = make_float4 ( 0,      dCPI4,  0,  dCPI4);
	////	gcube_quats_vec_[23] = make_float4 (-dCPI4,  0,      dCPI4,  0);

	cudaMemcpyToSymbol (gcube_quats, &(gcube_quats_vec_[0].x),
	                    24 * 4 * sizeof (float));
	CheckCUDAError ();


	//	cudaMemcpyFromSymbol (&(gcube_quats_vec[0]_.x), gcube_quats,
	//												24*4*sizeof (float));
	//	CheckCUDAError ();
	//	for (int i = 0; i < 24; i++) {
	//		std::cout << "gcube_quats[" << i << "] : "
	//			<< gcube_quats_vec_[i].x << " "
	//			<< gcube_quats_vec_[i].y << " "
	//			<< gcube_quats_vec_[i].z << " "
	//			<< gcube_quats_vec_[i].w
	//			<< std::endl;
	//	}

	time1 = GET_TIME ();
	cudaProfilerStart ();
	//	DilateByCubeByTexMipmapBaseILP1<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu ());
	//	DilateByCubeByTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu ());
	DilateByRotCubeByTexMipmapBase <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 (float4 *)frame_field.quats_pong (),
	 make_uint3 (frame_field.res ()[0],
	             frame_field.res ()[1],
	             frame_field.res ()[2]),
	 dilation_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	cudaProfilerStop ();
	CheckCUDAError ();
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;
#endif


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	time1 = GET_TIME ();
	//	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	//																					dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	        dilation_contour_base_size);
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "contour in "
	          << time2 - time1 << " ms." << std::endl;
#endif

	// Run a Dilation at full resolution only on the
	// contour cells computed at base resolution
	block_size = 128;
	time1 = GET_TIME ();
	cudaProfilerStart ();
	//	DilateByCubeByTexMipmap<<<(8*dilation_contour_base_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(dilation_contour_base, dilation_contour_base_size,
	//			 input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu ());
	DilateByRotCubeByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                           block_size,
	                           (block_size / 8)*sizeof (unsigned char) >>>
	                           (dilation_contour_base, dilation_contour_base_size,
	                            input_grid_.grid_gpu (),
	                            (float4 *)frame_field.quats_pong (),
	                            make_uint3 (frame_field.res ()[0], frame_field.res ()[1], frame_field.res ()[2]),
	                            dilation_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	cudaProfilerStop ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::DilateByCubeMipmap (const ScaleField & scale_field) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;
	ShowGPUMemoryUsage ();
#endif

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (scale_field);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	cudaProfilerStart ();
	//	DilateByCubeByTexMipmapBaseILP1<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu ());
	DilateByCubeByTexMipmapBase <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu ());
	//	DilateByRotCubeByTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	cudaProfilerStop ();
	CheckCUDAError ();
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;
#endif


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	time1 = GET_TIME ();
	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	                                        dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	//																								 dilation_contour_base_size);
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "contour in "
	          << time2 - time1 << " ms." << std::endl;
#endif

	// Run a Dilation at full resolution only on the
	// contour cells computed at base resolution
	block_size = 128;
	time1 = GET_TIME ();
	cudaProfilerStart ();
	DilateByCubeByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                        block_size,
	                        (block_size / 8)*sizeof (unsigned char) >>>
	                        (dilation_contour_base, dilation_contour_base_size,
	                         input_grid_.grid_gpu (),
	                         dilation_grid_.grid_gpu ());
	//	DilateByRotCubeByTexMipmap<<<(8*dilation_contour_base_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(dilation_contour_base, dilation_contour_base_size,
	//			 input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu ());
	//	cudaDeviceSynchronize ();
	cudaProfilerStop ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::DilateByCubeMipmap (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;
	ShowGPUMemoryUsage ();
#endif

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (se_size);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	DilateByCubeByTexMipmapBaseILP1 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;
#endif


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	time1 = GET_TIME ();
	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	                                        dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	//																								 dilation_contour_base_size);
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "contour in "
	          << time2 - time1 << " ms." << std::endl;
#endif

	// Run a Dilation at full resolution only on the
	// contour cells computed at base resolution
	block_size = 128;
	time1 = GET_TIME ();
	DilateByCubeByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                        block_size,
	                        (block_size / 8)*sizeof (unsigned char) >>>
	                        (dilation_contour_base, dilation_contour_base_size,
	                         input_grid_.grid_gpu (),
	                         dilation_grid_.grid_gpu (),
	                         floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::DilateByMipmap (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;
	ShowGPUMemoryUsage ();
#endif

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (se_size);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	//	DilateByCubeByMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	//	DilateByCubeByTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	time2 = GET_TIME ();
	//	std::cout << "[Dilate] : " << "base dilation in "
	//		<< time2 - time1 << " ms." << std::endl;

	DilateByCubeByTexMipmapBaseILP1 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));

	//	DilateBySphereByTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;
#endif


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	time1 = GET_TIME ();
	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	                                        dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	//																								 dilation_contour_base_size);
	time2 = GET_TIME ();
#ifdef GMORPHO_DEBUG
	std::cout << "[Dilate] : " << "contour in "
	          << time2 - time1 << " ms." << std::endl;
#endif

	//	unsigned int * dilation_contour_base_host = NULL;/*{{{*/
	//	dilation_contour_base_host = new unsigned int[dilation_contour_base_size];
	//	cudaMemcpy (dilation_contour_base_host, dilation_contour_base,
	//							dilation_contour_base_size*sizeof (unsigned int),
	//							cudaMemcpyDeviceToHost);
	//	SaveCubeList ("dilation_contour_base.ply",
	//								dilation_contour_base_host, dilation_contour_base_size,
	//								bbox_, res_, cell_size_);
	//	free (dilation_contour_base_host);/*}}}*/


	//	// Run a Dilation at full res
	//	block_dim = dim3 (8, 4, 4);
	//	grid_dim = dim3 ((2*data_res_[0]/block_dim.x)+1,
	//									 (2*data_res_[1]/block_dim.y)+1,
	//									 (2*data_res_[2]/block_dim.z)+1);
	//	time1 = GET_TIME ();
	//	DilateByCubeByMipmap<<<grid_dim, block_dim,
	//		(block_dim.x*block_dim.y*block_dim.z/8)*sizeof (unsigned char)>>>
	//			(input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu (),
	//			 floor (2.f*se_size/cell_size_));
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	time2 = GET_TIME ();
	//	std::cout << "[Dilate] : " << "dilation in "
	//		<< time2 - time1 << " ms." << std::endl;

	//	cudaMemset (dilation_grid_.grid_gpu ().voxels, 0x00,
	//							res_[0]*res_[1]*res_[2]*sizeof (unsigned char));

	block_size = 128;
	time1 = GET_TIME ();
	DilateByCubeByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                        block_size,
	                        (block_size / 8)*sizeof (unsigned char) >>>
	                        (dilation_contour_base, dilation_contour_base_size,
	                         input_grid_.grid_gpu (),
	                         dilation_grid_.grid_gpu (),
	                         floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	//	block_size = 128;
	//	time1 = GET_TIME ();
	//	DilateBySphereByTexMipmap<<<(8*dilation_contour_base_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(dilation_contour_base, dilation_contour_base_size,
	//			 input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu (),
	//			 floor (2.f*se_size/cell_size_));
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	time2 = GET_TIME ();
	//	std::cout << "[Dilate] : " << "dilation in "
	//		<< time2 - time1 << " ms." << std::endl;

	//	block_size = 128;
	//	time1 = GET_TIME ();
	//	DilateByCubeByMipmap<<<(8*dilation_contour_base_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(dilation_contour_base, dilation_contour_base_size,
	//			 input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu (),
	//			 floor (2.f*se_size/cell_size_));
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	time2 = GET_TIME ();
	//	std::cout << "[Dilate] : " << "dilation in "
	//		<< time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::DilateBySphereMipmap (const ScaleField & scale_field) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;

	float se_size = scale_field.global_scale ();

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (scale_field);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	DilateBySphereByTexMipmapBase <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	                                        dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	//																								 dilation_contour_base_size);

	block_size = 128;
	time1 = GET_TIME ();
	DilateBySphereByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                          block_size,
	                          (block_size / 8)*sizeof (unsigned char) >>>
	                          (dilation_contour_base, dilation_contour_base_size,
	                           input_grid_.grid_gpu (),
	                           dilation_grid_.grid_gpu (),
	                           floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::DilateBySphereMipmap (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	std::cout << "[Dilate] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;

	// Prepare Mipmap
	input_grid_.BuildTexMipmaps (se_size);

	// Run a Dilation at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	DilateBySphereByTexMipmapBase <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "base dilation in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * dilation_contour_base = NULL;
	unsigned int dilation_contour_base_size = 0;
	dilation_grid_.ComputeContourBaseTight (dilation_contour_base,
	                                        dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBase (dilation_contour_base,
	//																		 dilation_contour_base_size);
	//	dilation_grid_.ComputeContourBaseConservative (dilation_contour_base,
	//																								 dilation_contour_base_size);

	block_size = 128;
	time1 = GET_TIME ();
	DilateBySphereByTexMipmap <<< (8 * dilation_contour_base_size / block_size) + 1,
	                          block_size,
	                          (block_size / 8)*sizeof (unsigned char) >>>
	                          (dilation_contour_base, dilation_contour_base_size,
	                           input_grid_.grid_gpu (),
	                           dilation_grid_.grid_gpu (),
	                           floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Dilate] : " << "dilation in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour_base);
	ShowGPUMemoryUsage ();
}

__global__ void ErodeBySphereMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid,
        float radius) {

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	radius -= 1.f;
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned char * input_mipmap = dilation_contour_grid.morton_mipmap[1];
	//	float3 query = make_float3 (2*id_x, 2*id_y, 2*id_z);
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	Node stack[64];
	Node * stack_ptr = stack;
	*stack_ptr++ = make_node (-1, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	Node node = make_node (0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		int has_geometry_overlaps = ComputeNodeValue (node, input_mipmap);

		Node node_child = make_node (node.depth + 1, node.idx << 3);
		float3 base_node_coords = make_float3 (DecodeMorton3X (node_child.idx),
		                                       DecodeMorton3Y (node_child.idx),
		                                       DecodeMorton3Z (node_child.idx));
		float cell_size = max_res >> node_child.depth;
		float3 node_query = (1.f / cell_size) * query;
		float sq_node_se_size = radius / cell_size;
		sq_node_se_size *= sq_node_se_size;
		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_query, child_coords, sq_node_se_size);

			if (node_has_overlap && node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*stack_ptr++ = node_child;
			} else if (node_has_overlap && node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				*stack_ptr++ = make_node (-1, 0);
				//				return;
			}
			node_child.idx++;
		}
		// We pop the next node to process
		node = *--stack_ptr;
	} while (node.depth != -1);

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}

__global__ void ErodeBySphereMipmapFull (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU erosion_grid,
        unsigned int * morpho_centroids_grid,
        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned char * input_mipmap = input_grid.morton_mipmap[1];
	float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	Node stack[64];
	Node * stack_ptr = stack;
	*stack_ptr++ = make_node (-1, 0xffffffff);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	Node node = make_node (0, 0);

	do {
		// First retrieve the current node childs geometry occupation
		// pattern.
		int has_geometry_overlaps = ComputeNodeValue (node, input_mipmap);

		Node node_child = make_node (node.depth + 1, node.idx << 3);
		float3 base_node_coords = make_float3 (DecodeMorton3X (node_child.idx),
		                                       DecodeMorton3Y (node_child.idx),
		                                       DecodeMorton3Z (node_child.idx));
		float cell_size = max_res >> node_child.depth;
		float3 node_query = (1.f / cell_size) * query;
		float sq_node_se_size = radius / cell_size;
		sq_node_se_size *= sq_node_se_size;

		for (int k = 0; k < 8; k++) {
			unsigned int mask_k = (1 << k);
			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k);

			node_has_overlap = node_has_overlap &&
			                   HasCubeSEOverlap (node_query, child_coords, sq_node_se_size);

			if (node_has_overlap && node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*stack_ptr++ = node_child;
			} else if (node_has_overlap && node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				//				*stack_ptr++ = make_node (-1, node_child.idx);
				*stack_ptr++ = make_node (-1, 0);
				//				return;
			}
			node_child.idx++;
		}
		// We pop the next node to process
		node = *--stack_ptr;
	} while (node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char erosion_fine = ~(1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_erosion_occup[threadIdx.x / 8] = 0xff;
	}
	__syncthreads ();

	if (erosion_occup == 0x00)
		myAtomicAnd (&shared_erosion_occup[threadIdx.x / 8], erosion_fine);
	__syncthreads ();

	if (tid == 0) {
		erosion_voxels[i] = shared_erosion_occup[threadIdx.x / 8];
	}

	//	if (id_x == 189 && id_y == 111 && id_z == 46 && node.idx != 0xffffffff) {
	//		printf ("node : %i %i\n", node.idx, 4*resxy*(id_z) + 2*res*(id_y) + id_x);
	//	}
	//	if (node.idx != 0xffffffff)
	//		morpho_centroids_grid[4*resxy*(id_z) + 2*res*(id_y) + id_x] = node.idx;
}

__global__ void ErodeBySphereTexDualMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid,
        float radius) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		float3 node_coords = make_float3 (tex_node.id.x,
		                                  tex_node.id.y,
		                                  tex_node.id.z);
		float tex_depth = 8 - tex_node.depth;
		int reg_depth = tex_node.depth;
		unsigned int curr_res = (1 << reg_depth);
		uint2 has_geometry_overlaps_tex = tex3DLod<uint2> (dilation_contour_grid.tex_mipmap,
		                                  node_coords.x / curr_res,
		                                  node_coords.y / curr_res,
		                                  node_coords.z / curr_res,
		                                  tex_depth);
		int has_geometry_overlaps = has_geometry_overlaps_tex.y >> 24;
		unsigned long long data = has_geometry_overlaps_tex.y;
		data = data << 32;
		data |= has_geometry_overlaps_tex.x;
		unsigned long long mask_size = 0x7f;
		//		if (thread_idx == 0) {
		//			printf ("x%.8x x%.8x | x%.8x\n",
		//							has_geometry_overlaps_tex.y,
		//							has_geometry_overlaps_tex.x,
		//							has_geometry_overlaps);
		//		}

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float3 base_node_coords = make_float3 (base_tex_node_coords.x,
		                                       base_tex_node_coords.y,
		                                       base_tex_node_coords.z);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;

		//		for (int k = 0; k < 8; k++) {
		//			unsigned char uchar_radius = mask_size & data;
		//			float sq_node_se_size = uchar_radius - 1;
		//			sq_node_se_size /= cell_size;
		//			sq_node_se_size *= sq_node_se_size;
		//			unsigned int mask_k = (1 << k);
		//			float3 child_coords = Z8Coordinates (base_node_coords, mask_k);
		//			bool node_has_overlap = (has_geometry_overlaps & mask_k);
		//
		//			node_has_overlap = node_has_overlap &&
		//				HasCubeSEOverlap (node_query, child_coords, sq_node_se_size);
		//
		//			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords,
		//																										 mask_k);
		//			NodeTex tex_node_child = make_node (tex_node.depth + 1,
		//																					child_tex_node_coords.x,
		//																					child_tex_node_coords.y,
		//																					child_tex_node_coords.z);
		//
		//			if (node_has_overlap && tex_node_child.depth != max_depth) {
		//				// If the node overlap the SE and IS NOT a leaf
		//				// we push it into the stack
		//				*tex_stack_ptr++ = tex_node_child;
		//			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
		//				// If the node overlap the SE and IS a leaf
		//				// we splat the voxel and return from the thread
		//				//				dilation_voxels[i] = 0xff;
		//				erosion_occup = 0x00;
		//				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
		//				//				return;
		//			}
		//			data = data >> 7;
		//		}

		for (int k = 0; k < 8; k++) {
			unsigned char uchar_radius = mask_size & data;
			float sq_node_inner_se_size = 0.f;
			float sq_node_outer_se_size = uchar_radius;
			sq_node_outer_se_size /= cell_size;
			sq_node_inner_se_size = fmaxf (0.f, sq_node_outer_se_size - 1.73205080757f);
			sq_node_outer_se_size *= sq_node_outer_se_size;
			sq_node_inner_se_size *= sq_node_inner_se_size;

			//			if (blockIdx.x == 1 && threadIdx.x == 36 && k == 0) {
			//				printf ("count : %i se_inner_size %f se_outer_size %f\n", counter,
			//								sqrt(sq_node_inner_se_size), sqrt (sq_node_outer_se_size));
			//			}

			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords,
			                                mask_k);
			float3 child_coords = make_float3 (child_tex_node_coords.x,
			                                   child_tex_node_coords.y,
			                                   child_tex_node_coords.z);
			bool node_has_geometry_overlap, node_has_outer_overlap, node_has_inner_overlap;

			node_has_geometry_overlap = (has_geometry_overlaps & mask_k);

			HasCubeSEOverlap (node_query, child_coords,
			                  sq_node_inner_se_size, sq_node_outer_se_size,
			                  node_has_inner_overlap, node_has_outer_overlap);

			//			node_has_inner_overlap = false;
			node_has_inner_overlap = node_has_inner_overlap && node_has_geometry_overlap;
			node_has_outer_overlap = node_has_outer_overlap && node_has_geometry_overlap;

			NodeTex tex_node_child = make_node (tex_node.depth + 1,
			                                    child_tex_node_coords.x,
			                                    child_tex_node_coords.y,
			                                    child_tex_node_coords.z);

			if (node_has_inner_overlap) {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			} else if (node_has_outer_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_outer_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				//				return;
			}
			data = data >> 7;
		}

		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	//	// Shared Memory Access by atomics
	//	unsigned char tid = threadIdx.x%8;
	//	unsigned char erosion_fine = ~(1 << tid);
	//
	//	// Shared Memory Initialization
	//	if (tid == 0) {
	//		shared_erosion_occup[threadIdx.x/8] = 0xff;
	//	}
	//	__syncthreads ();
	//
	//	if (erosion_occup == 0x00)
	//		myAtomicAnd (&shared_erosion_occup[threadIdx.x/8], erosion_fine);
	//	__syncthreads ();
	//
	//	if (tid == 0) {
	//		erosion_voxels[i] = shared_erosion_occup[threadIdx.x/8];
	//	}

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}

#define EROSION_BASE_ILP2_BLOCK_SIZE 64
__global__ void ErodeBySphereTexMipmapBaseILP2 (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid,
        float radius) {
	float accum[2];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = dilation_contour_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	//	const int tidx = blockDim.x*blockDim.y*threadIdx.z +
	//		blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		uint2 data_tex = tex3DLod<uint2> (dilation_contour_grid.tex_mipmap,
		                                  (float)tex_node.id.x / curr_res,
		                                  (float)tex_node.id.y / curr_res,
		                                  (float)tex_node.id.z / curr_res,
		                                  max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		unsigned char uchar_radius[8];
		uchar_radius[0] = 0x7f & data;
		uchar_radius[1] = 0x7f & (data >> 7);
		uchar_radius[2] = 0x7f & (data >> 14);
		uchar_radius[3] = 0x7f & (data >> 21);
		uchar_radius[4] = 0x7f & (data >> 28);
		uchar_radius[5] = 0x7f & (data >> 35);
		uchar_radius[6] = 0x7f & (data >> 42);
		uchar_radius[7] = 0x7f & (data >> 49);

		float sq_node_se_size[8];
		sq_node_se_size[0] = square ((float)(uchar_radius[0] - 1) / cell_size);
		sq_node_se_size[1] = square ((float)(uchar_radius[1] - 1) / cell_size);
		sq_node_se_size[2] = square ((float)(uchar_radius[2] - 1) / cell_size);
		sq_node_se_size[3] = square ((float)(uchar_radius[3] - 1) / cell_size);
		sq_node_se_size[4] = square ((float)(uchar_radius[4] - 1) / cell_size);
		sq_node_se_size[5] = square ((float)(uchar_radius[5] - 1) / cell_size);
		sq_node_se_size[6] = square ((float)(uchar_radius[6] - 1) / cell_size);
		sq_node_se_size[7] = square ((float)(uchar_radius[7] - 1) / cell_size);

		float sq_d[8];
		sq_d[0] = 0.f; sq_d[1] = 0.f; sq_d[2] = 0.f; sq_d[3] = 0.f;
		sq_d[4] = 0.f; sq_d[5] = 0.f; sq_d[6] = 0.f; sq_d[7] = 0.f;

		// X dimension and min
		accum[0] = node_query.x < b_coords.x ?
		           square (node_query.x - b_coords.x) : 0.f;
		accum[1] = node_query.x < (b_coords.x + 1.f) ?
		           square (node_query.x - b_coords.x - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[1];
		sq_d[2] += accum[0];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[1];
		sq_d[6] += accum[0];
		sq_d[7] += accum[1];

		// X dimension and max
		accum[0] = node_query.x > (b_coords.x + 1.f) ?
		           square (node_query.x - b_coords.x - 1.f) : 0.f;
		accum[1] = node_query.x > (b_coords.x + 2.f) ?
		           square (node_query.x - b_coords.x - 2.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[1];
		sq_d[2] += accum[0];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[1];
		sq_d[6] += accum[0];
		sq_d[7] += accum[1];

		// Y dimension and min
		accum[0] = node_query.y < b_coords.y ?
		           square (node_query.y - b_coords.y) : 0.f;
		accum[1] = node_query.y < (b_coords.y + 1.f) ?
		           square (node_query.y - b_coords.y - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[1];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[0];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Y dimension and max
		accum[0] = node_query.y > (b_coords.y + 1.f) ?
		           square (node_query.y - b_coords.y - 1.f) : 0.f;
		accum[1] = node_query.y > (b_coords.y + 2.f) ?
		           square (node_query.y - b_coords.y - 2.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[1];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[0];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Z dimension and min
		accum[0] = node_query.z < b_coords.z ?
		           square (node_query.z - b_coords.z) : 0.f;
		accum[1] = node_query.z < (b_coords.z + 1.f) ?
		           square (node_query.z - b_coords.z - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[0];
		sq_d[3] += accum[0];
		sq_d[4] += accum[1];
		sq_d[5] += accum[1];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Z dimension and max
		accum[0] = node_query.z > (b_coords.z + 1.f) ?
		           square (node_query.z - b_coords.z - 1.f) : 0.f;
		accum[1] = node_query.z > (b_coords.z + 2.f) ?
		           square (node_query.z - b_coords.z - 2.f) : 0.f;


		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[0];
		sq_d[3] += accum[0];
		sq_d[4] += accum[1];
		sq_d[5] += accum[1];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		//		ushort3 base_tex_node_coords = make_ushort3 (2*tex_node.id.x,
		//																								 2*tex_node.id.y,
		//																								 2*tex_node.id.z);
		if (sq_d[0] < sq_node_se_size[0]
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[1] < sq_node_se_size[1]
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[2] < sq_node_se_size[2]
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[3] < sq_node_se_size[3]
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[4] < sq_node_se_size[4]
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[5] < sq_node_se_size[5]
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[6] < sq_node_se_size[6]
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[7] < sq_node_se_size[7]
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}


		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}

__global__ void ErodeBySphereTexMipmapBaseILP2 (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid) {
	float accum[2];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = dilation_contour_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	//	const int tidx = blockDim.x*blockDim.y*threadIdx.z +
	//		blockDim.x*threadIdx.y + threadIdx.x;
	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		uint2 data_tex = tex3DLod<uint2> (dilation_contour_grid.tex_mipmap,
		                                  (float)tex_node.id.x / curr_res,
		                                  (float)tex_node.id.y / curr_res,
		                                  (float)tex_node.id.z / curr_res,
		                                  max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		unsigned char uchar_radius[8];
		uchar_radius[0] = 0x7f & data;
		uchar_radius[1] = 0x7f & (data >> 7);
		uchar_radius[2] = 0x7f & (data >> 14);
		uchar_radius[3] = 0x7f & (data >> 21);
		uchar_radius[4] = 0x7f & (data >> 28);
		uchar_radius[5] = 0x7f & (data >> 35);
		uchar_radius[6] = 0x7f & (data >> 42);
		uchar_radius[7] = 0x7f & (data >> 49);

		float sq_node_se_size[8];
		sq_node_se_size[0] = square ((float)(uchar_radius[0] - 1) / cell_size);
		sq_node_se_size[1] = square ((float)(uchar_radius[1] - 1) / cell_size);
		sq_node_se_size[2] = square ((float)(uchar_radius[2] - 1) / cell_size);
		sq_node_se_size[3] = square ((float)(uchar_radius[3] - 1) / cell_size);
		sq_node_se_size[4] = square ((float)(uchar_radius[4] - 1) / cell_size);
		sq_node_se_size[5] = square ((float)(uchar_radius[5] - 1) / cell_size);
		sq_node_se_size[6] = square ((float)(uchar_radius[6] - 1) / cell_size);
		sq_node_se_size[7] = square ((float)(uchar_radius[7] - 1) / cell_size);

		float sq_d[8];
		sq_d[0] = 0.f; sq_d[1] = 0.f; sq_d[2] = 0.f; sq_d[3] = 0.f;
		sq_d[4] = 0.f; sq_d[5] = 0.f; sq_d[6] = 0.f; sq_d[7] = 0.f;

		// X dimension and min
		accum[0] = node_query.x < b_coords.x ?
		           square (node_query.x - b_coords.x) : 0.f;
		accum[1] = node_query.x < (b_coords.x + 1.f) ?
		           square (node_query.x - b_coords.x - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[1];
		sq_d[2] += accum[0];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[1];
		sq_d[6] += accum[0];
		sq_d[7] += accum[1];

		// X dimension and max
		accum[0] = node_query.x > (b_coords.x + 1.f) ?
		           square (node_query.x - b_coords.x - 1.f) : 0.f;
		accum[1] = node_query.x > (b_coords.x + 2.f) ?
		           square (node_query.x - b_coords.x - 2.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[1];
		sq_d[2] += accum[0];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[1];
		sq_d[6] += accum[0];
		sq_d[7] += accum[1];

		// Y dimension and min
		accum[0] = node_query.y < b_coords.y ?
		           square (node_query.y - b_coords.y) : 0.f;
		accum[1] = node_query.y < (b_coords.y + 1.f) ?
		           square (node_query.y - b_coords.y - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[1];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[0];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Y dimension and max
		accum[0] = node_query.y > (b_coords.y + 1.f) ?
		           square (node_query.y - b_coords.y - 1.f) : 0.f;
		accum[1] = node_query.y > (b_coords.y + 2.f) ?
		           square (node_query.y - b_coords.y - 2.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[1];
		sq_d[3] += accum[1];
		sq_d[4] += accum[0];
		sq_d[5] += accum[0];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Z dimension and min
		accum[0] = node_query.z < b_coords.z ?
		           square (node_query.z - b_coords.z) : 0.f;
		accum[1] = node_query.z < (b_coords.z + 1.f) ?
		           square (node_query.z - b_coords.z - 1.f) : 0.f;

		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[0];
		sq_d[3] += accum[0];
		sq_d[4] += accum[1];
		sq_d[5] += accum[1];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		// Z dimension and max
		accum[0] = node_query.z > (b_coords.z + 1.f) ?
		           square (node_query.z - b_coords.z - 1.f) : 0.f;
		accum[1] = node_query.z > (b_coords.z + 2.f) ?
		           square (node_query.z - b_coords.z - 2.f) : 0.f;


		sq_d[0] += accum[0];
		sq_d[1] += accum[0];
		sq_d[2] += accum[0];
		sq_d[3] += accum[0];
		sq_d[4] += accum[1];
		sq_d[5] += accum[1];
		sq_d[6] += accum[1];
		sq_d[7] += accum[1];

		//		ushort3 base_tex_node_coords = make_ushort3 (2*tex_node.id.x,
		//																								 2*tex_node.id.y,
		//																								 2*tex_node.id.z);
		if (sq_d[0] < sq_node_se_size[0]
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[1] < sq_node_se_size[1]
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[2] < sq_node_se_size[2]
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[3] < sq_node_se_size[3]
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[4] < sq_node_se_size[4]
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[5] < sq_node_se_size[5]
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[6] < sq_node_se_size[6]
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[7] < sq_node_se_size[7]
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}


		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}
#define EROSION_BASE_ILP1_BLOCK_SIZE 64

__global__ void ErodeBySphereTexMipmapBaseILP1 (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid,
        float radius) {
	__shared__ float accum[2][6 * EROSION_BASE_ILP1_BLOCK_SIZE];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = dilation_contour_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	const int tidx = blockDim.x * blockDim.y * threadIdx.z +
	                 blockDim.x * threadIdx.y + threadIdx.x;
	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		uint2 data_tex = tex3DLod<uint2> (dilation_contour_grid.tex_mipmap,
		                                  (float)tex_node.id.x / curr_res,
		                                  (float)tex_node.id.y / curr_res,
		                                  (float)tex_node.id.z / curr_res,
		                                  max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

		// k = 0 -> 0 0 0
		// k = 1 -> 1 0 0
		// k = 2 -> 0 1 0
		// k = 3 -> 1 1 0
		// k = 4 -> 0 0 1
		// k = 5 -> 1 0 1
		// k = 6 -> 0 1 1
		// k = 7 -> 1 1 1
		float3 b_coords = make_float3 (base_tex_node_coords.x,
		                               base_tex_node_coords.y,
		                               base_tex_node_coords.z);
		unsigned char uchar_radius[8];
		uchar_radius[0] = 0x7f & data;
		uchar_radius[1] = 0x7f & (data >> 7);
		uchar_radius[2] = 0x7f & (data >> 14);
		uchar_radius[3] = 0x7f & (data >> 21);
		uchar_radius[4] = 0x7f & (data >> 28);
		uchar_radius[5] = 0x7f & (data >> 35);
		uchar_radius[6] = 0x7f & (data >> 42);
		uchar_radius[7] = 0x7f & (data >> 49);

		float sq_node_se_size[8];
		sq_node_se_size[0] = square ((float)(uchar_radius[0] - 1) / cell_size);
		sq_node_se_size[1] = square ((float)(uchar_radius[1] - 1) / cell_size);
		sq_node_se_size[2] = square ((float)(uchar_radius[2] - 1) / cell_size);
		sq_node_se_size[3] = square ((float)(uchar_radius[3] - 1) / cell_size);
		sq_node_se_size[4] = square ((float)(uchar_radius[4] - 1) / cell_size);
		sq_node_se_size[5] = square ((float)(uchar_radius[5] - 1) / cell_size);
		sq_node_se_size[6] = square ((float)(uchar_radius[6] - 1) / cell_size);
		sq_node_se_size[7] = square ((float)(uchar_radius[7] - 1) / cell_size);

		float sq_d[8];
		sq_d[0] = 0.f; sq_d[1] = 0.f; sq_d[2] = 0.f; sq_d[3] = 0.f;
		sq_d[4] = 0.f; sq_d[5] = 0.f; sq_d[6] = 0.f; sq_d[7] = 0.f;

		// X dimension and min
		accum[0][6 * tidx + 0] = node_query.x < b_coords.x ?
		                         square (node_query.x - b_coords.x) : 0.f;
		accum[1][6 * tidx + 0] = node_query.x < (b_coords.x + 1.f) ?
		                         square (node_query.x - b_coords.x - 1.f) : 0.f;
		// X dimension and max
		accum[0][6 * tidx + 1] = node_query.x > (b_coords.x + 1.f) ?
		                         square (node_query.x - b_coords.x - 1.f) : 0.f;
		accum[1][6 * tidx + 1] = node_query.x > (b_coords.x + 2.f) ?
		                         square (node_query.x - b_coords.x - 2.f) : 0.f;
		// Y dimension and min
		accum[0][6 * tidx + 2] = node_query.y < b_coords.y ?
		                         square (node_query.y - b_coords.y) : 0.f;
		accum[1][6 * tidx + 2] = node_query.y < (b_coords.y + 1.f) ?
		                         square (node_query.y - b_coords.y - 1.f) : 0.f;
		// Y dimension and max
		accum[0][6 * tidx + 3] = node_query.y > (b_coords.y + 1.f) ?
		                         square (node_query.y - b_coords.y - 1.f) : 0.f;
		accum[1][6 * tidx + 3] = node_query.y > (b_coords.y + 2.f) ?
		                         square (node_query.y - b_coords.y - 2.f) : 0.f;
		// Z dimension and min
		accum[0][6 * tidx + 4] = node_query.z < b_coords.z ?
		                         square (node_query.z - b_coords.z) : 0.f;
		accum[1][6 * tidx + 4] = node_query.z < (b_coords.z + 1.f) ?
		                         square (node_query.z - b_coords.z - 1.f) : 0.f;
		// Z dimension and max
		accum[0][6 * tidx + 5] = node_query.z > (b_coords.z + 1.f) ?
		                         square (node_query.z - b_coords.z - 1.f) : 0.f;
		accum[1][6 * tidx + 5] = node_query.z > (b_coords.z + 2.f) ?
		                         square (node_query.z - b_coords.z - 2.f) : 0.f;

		sq_d[0] += accum[0][6 * tidx + 0];
		sq_d[1] += accum[1][6 * tidx + 0];
		sq_d[2] += accum[0][6 * tidx + 0];
		sq_d[3] += accum[1][6 * tidx + 0];
		sq_d[4] += accum[0][6 * tidx + 0];
		sq_d[5] += accum[1][6 * tidx + 0];
		sq_d[6] += accum[0][6 * tidx + 0];
		sq_d[7] += accum[1][6 * tidx + 0];

		sq_d[0] += accum[0][6 * tidx + 1];
		sq_d[1] += accum[1][6 * tidx + 1];
		sq_d[2] += accum[0][6 * tidx + 1];
		sq_d[3] += accum[1][6 * tidx + 1];
		sq_d[4] += accum[0][6 * tidx + 1];
		sq_d[5] += accum[1][6 * tidx + 1];
		sq_d[6] += accum[0][6 * tidx + 1];
		sq_d[7] += accum[1][6 * tidx + 1];

		sq_d[0] += accum[0][6 * tidx + 2];
		sq_d[1] += accum[0][6 * tidx + 2];
		sq_d[2] += accum[1][6 * tidx + 2];
		sq_d[3] += accum[1][6 * tidx + 2];
		sq_d[4] += accum[0][6 * tidx + 2];
		sq_d[5] += accum[0][6 * tidx + 2];
		sq_d[6] += accum[1][6 * tidx + 2];
		sq_d[7] += accum[1][6 * tidx + 2];

		sq_d[0] += accum[0][6 * tidx + 3];
		sq_d[1] += accum[0][6 * tidx + 3];
		sq_d[2] += accum[1][6 * tidx + 3];
		sq_d[3] += accum[1][6 * tidx + 3];
		sq_d[4] += accum[0][6 * tidx + 3];
		sq_d[5] += accum[0][6 * tidx + 3];
		sq_d[6] += accum[1][6 * tidx + 3];
		sq_d[7] += accum[1][6 * tidx + 3];

		sq_d[0] += accum[0][6 * tidx + 4];
		sq_d[1] += accum[0][6 * tidx + 4];
		sq_d[2] += accum[0][6 * tidx + 4];
		sq_d[3] += accum[0][6 * tidx + 4];
		sq_d[4] += accum[1][6 * tidx + 4];
		sq_d[5] += accum[1][6 * tidx + 4];
		sq_d[6] += accum[1][6 * tidx + 4];
		sq_d[7] += accum[1][6 * tidx + 4];

		sq_d[0] += accum[0][6 * tidx + 5];
		sq_d[1] += accum[0][6 * tidx + 5];
		sq_d[2] += accum[0][6 * tidx + 5];
		sq_d[3] += accum[0][6 * tidx + 5];
		sq_d[4] += accum[1][6 * tidx + 5];
		sq_d[5] += accum[1][6 * tidx + 5];
		sq_d[6] += accum[1][6 * tidx + 5];
		sq_d[7] += accum[1][6 * tidx + 5];

		if (sq_d[0] < sq_node_se_size[0]
		        && ((has_geometry_overlaps & (1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[1] < sq_node_se_size[1]
		        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[2] < sq_node_se_size[2]
		        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[3] < sq_node_se_size[3]
		        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[4] < sq_node_se_size[4]
		        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[5] < sq_node_se_size[5]
		        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[6] < sq_node_se_size[6]
		        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}

		if (sq_d[7] < sq_node_se_size[7]
		        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
			if ((tex_node.depth + 1) != max_depth) {
				*tex_stack_ptr++ = make_node (tex_node.depth + 1,
				                              base_tex_node_coords.x + 1,
				                              base_tex_node_coords.y + 1,
				                              base_tex_node_coords.z + 1);
			} else {
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			}
		}


		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}

__global__ void ErodeBySphereTexMipmapBase (GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU dilation_contour_grid,
        GridGPU erosion_grid,
        float radius) {
	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	uint3 data_res = input_grid.data_res;
	unsigned int res = erosion_grid.res.x;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (2 * id_x + 1, 2 * id_y + 1, 2 * id_z + 1);
	int max_depth = dilation_contour_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	//	const int tidx = threadIdx.x;
	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		unsigned int curr_res = (1 << tex_node.depth);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		uint2 data_tex = tex3DLod<uint2> (dilation_contour_grid.tex_mipmap,
		                                  (float)tex_node.id.x / curr_res,
		                                  (float)tex_node.id.y / curr_res,
		                                  (float)tex_node.id.z / curr_res,
		                                  max_depth - tex_node.depth - 1);
		int has_geometry_overlaps = data_tex.y >> 24;
		unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);
		unsigned long long mask_size = 0x7f;

		for (int k = 0; k < 8; k++) {
			unsigned char uchar_radius = mask_size & data;
			float sq_node_se_size = square ((float)(uchar_radius - 1) / cell_size);
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_query, child_tex_node_coords, sq_node_se_size);

			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);
			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				//				return;
			}
			data = data >> 7;
		}
		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	if (erosion_occup == 0x00)
		erosion_voxels[i] = erosion_occup;
}

#define WARP_SIZE 32
#define EROSION_ILP_BLOCK_SIZE 128
#define EROSION_BLOCK_SIZE 256

__global__ void ErodeBySphereTexMipmapPersistentWarpILP (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU erosion_grid,
        int * global_warp_counter) {
	__shared__ unsigned char shared_erosion_occup[EROSION_ILP_BLOCK_SIZE / 8];
	__shared__ float accum[2][6 * EROSION_ILP_BLOCK_SIZE];
	/*
	 * Variables valid accross all traversals
	 */
	NodeTex tex_stack[64];
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned int res = input_grid.res.x;
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = threadIdx.x;
	const int tidq = tidx / 8;
	const unsigned char tidr = tidx % 8;

	do {
		/*
		 * Fetch Warp Data: 4 x (8 cells)
		 */
		unsigned int cell_idx = cells[thread_idx / 8];
		// Compute 3D indices from linear indices
		unsigned int remainder, resxy, id_x, id_y, id_z;
		resxy = res * res;
		id_z = cell_idx / resxy;
		remainder = cell_idx % resxy;
		id_y = remainder / res;
		id_x = remainder % res;

		id_x *= 2; id_y *= 2; id_z *= 2;
		id_x += (0xaa & (1 << (tidr))) ? 1 : 0;
		id_y += (0xcc & (1 << (tidr))) ? 1 : 0;
		id_z += (0xf0 & (1 << (tidr))) ? 1 : 0;

		float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
		unsigned char erosion_occup = 0xff;

		/*
		 * Traversal Part
		 */
		// Allocate traversal stack from thread-local memory,
		// and push VOID to indicate that there are no postponed nodes.
		NodeTex * tex_stack_ptr = tex_stack;
		*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

		// Traverse nodes starting from the 8 sub-trees that thightly
		// intersect the Structuring Element
		NodeTex tex_node = make_node (0, 0, 0, 0);

		do {
			// First retrieve the current node childs geometry occupation
			// pattern.
			unsigned int curr_res = (1 << tex_node.depth);
			float cell_size = max_res >> (tex_node.depth + 1);
			float3 node_query = (1.f / cell_size) * query;
			ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
			                               2 * tex_node.id.y,
			                               2 * tex_node.id.z);
			uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
			                                  (float)tex_node.id.x / curr_res,
			                                  (float)tex_node.id.y / curr_res,
			                                  (float)tex_node.id.z / curr_res,
			                                  max_depth - tex_node.depth - 1);
			unsigned char has_geometry_overlaps = data_tex.y >> 24;
			unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

			// k = 0 -> 0 0 0
			// k = 1 -> 1 0 0
			// k = 2 -> 0 1 0
			// k = 3 -> 1 1 0
			// k = 4 -> 0 0 1
			// k = 5 -> 1 0 1
			// k = 6 -> 0 1 1
			// k = 7 -> 1 1 1
			float3 b_coords = make_float3 (base_tex_node_coords.x,
			                               base_tex_node_coords.y,
			                               base_tex_node_coords.z);
			unsigned char uchar_radius[8];
			uchar_radius[0] = 0x7f & data;
			uchar_radius[1] = 0x7f & (data >> 7);
			uchar_radius[2] = 0x7f & (data >> 14);
			uchar_radius[3] = 0x7f & (data >> 21);
			uchar_radius[4] = 0x7f & (data >> 28);
			uchar_radius[5] = 0x7f & (data >> 35);
			uchar_radius[6] = 0x7f & (data >> 42);
			uchar_radius[7] = 0x7f & (data >> 49);

			float sq_node_se_size[8];
			sq_node_se_size[0] = square ((float)(uchar_radius[0] - 1) / cell_size);
			sq_node_se_size[1] = square ((float)(uchar_radius[1] - 1) / cell_size);
			sq_node_se_size[2] = square ((float)(uchar_radius[2] - 1) / cell_size);
			sq_node_se_size[3] = square ((float)(uchar_radius[3] - 1) / cell_size);
			sq_node_se_size[4] = square ((float)(uchar_radius[4] - 1) / cell_size);
			sq_node_se_size[5] = square ((float)(uchar_radius[5] - 1) / cell_size);
			sq_node_se_size[6] = square ((float)(uchar_radius[6] - 1) / cell_size);
			sq_node_se_size[7] = square ((float)(uchar_radius[7] - 1) / cell_size);

			float sq_d[8];
			sq_d[0] = 0.f; sq_d[1] = 0.f; sq_d[2] = 0.f; sq_d[3] = 0.f;
			sq_d[4] = 0.f; sq_d[5] = 0.f; sq_d[6] = 0.f; sq_d[7] = 0.f;

			// X dimension and min
			accum[0][6 * tidx + 0] = node_query.x < b_coords.x ?
			                         square (node_query.x - b_coords.x) : 0.f;
			accum[1][6 * tidx + 0] = node_query.x < (b_coords.x + 1.f) ?
			                         square (node_query.x - b_coords.x - 1.f) : 0.f;
			// X dimension and max
			accum[0][6 * tidx + 1] = node_query.x > (b_coords.x + 1.f) ?
			                         square (node_query.x - b_coords.x - 1.f) : 0.f;
			accum[1][6 * tidx + 1] = node_query.x > (b_coords.x + 2.f) ?
			                         square (node_query.x - b_coords.x - 2.f) : 0.f;
			// Y dimension and min
			accum[0][6 * tidx + 2] = node_query.y < b_coords.y ?
			                         square (node_query.y - b_coords.y) : 0.f;
			accum[1][6 * tidx + 2] = node_query.y < (b_coords.y + 1.f) ?
			                         square (node_query.y - b_coords.y - 1.f) : 0.f;
			// Y dimension and max
			accum[0][6 * tidx + 3] = node_query.y > (b_coords.y + 1.f) ?
			                         square (node_query.y - b_coords.y - 1.f) : 0.f;
			accum[1][6 * tidx + 3] = node_query.y > (b_coords.y + 2.f) ?
			                         square (node_query.y - b_coords.y - 2.f) : 0.f;
			// Z dimension and min
			accum[0][6 * tidx + 4] = node_query.z < b_coords.z ?
			                         square (node_query.z - b_coords.z) : 0.f;
			accum[1][6 * tidx + 4] = node_query.z < (b_coords.z + 1.f) ?
			                         square (node_query.z - b_coords.z - 1.f) : 0.f;
			// Z dimension and max
			accum[0][6 * tidx + 5] = node_query.z > (b_coords.z + 1.f) ?
			                         square (node_query.z - b_coords.z - 1.f) : 0.f;
			accum[1][6 * tidx + 5] = node_query.z > (b_coords.z + 2.f) ?
			                         square (node_query.z - b_coords.z - 2.f) : 0.f;

			sq_d[0] += accum[0][6 * tidx + 0];
			sq_d[1] += accum[1][6 * tidx + 0];
			sq_d[2] += accum[0][6 * tidx + 0];
			sq_d[3] += accum[1][6 * tidx + 0];
			sq_d[4] += accum[0][6 * tidx + 0];
			sq_d[5] += accum[1][6 * tidx + 0];
			sq_d[6] += accum[0][6 * tidx + 0];
			sq_d[7] += accum[1][6 * tidx + 0];

			sq_d[0] += accum[0][6 * tidx + 1];
			sq_d[1] += accum[1][6 * tidx + 1];
			sq_d[2] += accum[0][6 * tidx + 1];
			sq_d[3] += accum[1][6 * tidx + 1];
			sq_d[4] += accum[0][6 * tidx + 1];
			sq_d[5] += accum[1][6 * tidx + 1];
			sq_d[6] += accum[0][6 * tidx + 1];
			sq_d[7] += accum[1][6 * tidx + 1];

			sq_d[0] += accum[0][6 * tidx + 2];
			sq_d[1] += accum[0][6 * tidx + 2];
			sq_d[2] += accum[1][6 * tidx + 2];
			sq_d[3] += accum[1][6 * tidx + 2];
			sq_d[4] += accum[0][6 * tidx + 2];
			sq_d[5] += accum[0][6 * tidx + 2];
			sq_d[6] += accum[1][6 * tidx + 2];
			sq_d[7] += accum[1][6 * tidx + 2];

			sq_d[0] += accum[0][6 * tidx + 3];
			sq_d[1] += accum[0][6 * tidx + 3];
			sq_d[2] += accum[1][6 * tidx + 3];
			sq_d[3] += accum[1][6 * tidx + 3];
			sq_d[4] += accum[0][6 * tidx + 3];
			sq_d[5] += accum[0][6 * tidx + 3];
			sq_d[6] += accum[1][6 * tidx + 3];
			sq_d[7] += accum[1][6 * tidx + 3];

			sq_d[0] += accum[0][6 * tidx + 4];
			sq_d[1] += accum[0][6 * tidx + 4];
			sq_d[2] += accum[0][6 * tidx + 4];
			sq_d[3] += accum[0][6 * tidx + 4];
			sq_d[4] += accum[1][6 * tidx + 4];
			sq_d[5] += accum[1][6 * tidx + 4];
			sq_d[6] += accum[1][6 * tidx + 4];
			sq_d[7] += accum[1][6 * tidx + 4];

			sq_d[0] += accum[0][6 * tidx + 5];
			sq_d[1] += accum[0][6 * tidx + 5];
			sq_d[2] += accum[0][6 * tidx + 5];
			sq_d[3] += accum[0][6 * tidx + 5];
			sq_d[4] += accum[1][6 * tidx + 5];
			sq_d[5] += accum[1][6 * tidx + 5];
			sq_d[6] += accum[1][6 * tidx + 5];
			sq_d[7] += accum[1][6 * tidx + 5];

			//			if (thread_idx == 0) {
			//				for (int k = 0; k < 8; k++) {
			//					printf ("id : %i %i %i k : %i sq_d: %f\n",
			//									tex_node.id.x, tex_node.id.y, tex_node.id.z,
			//									k, sq_d[k]);
			//				}
			//			}

			if (sq_d[0] < sq_node_se_size[0]
			        && ((has_geometry_overlaps & (1)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[1] < sq_node_se_size[1]
			        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[2] < sq_node_se_size[2]
			        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[3] < sq_node_se_size[3]
			        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[4] < sq_node_se_size[4]
			        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[5] < sq_node_se_size[5]
			        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[6] < sq_node_se_size[6]
			        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[7] < sq_node_se_size[7]
			        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			// We pop the next node to process
			tex_node = *--tex_stack_ptr;
		} while (tex_node.depth != -1);

		// Shared Memory Access by atomics
		unsigned char erosion_fine = ~(1 << tidr);

		// Shared Memory Initialization
		if (tidr == 0) {
			shared_erosion_occup[tidq] = 0xff;
		}
		//		unsigned int sync_group_mask = 0xff << (8*tidr);
		if (erosion_occup == 0x00)
			myAtomicAnd (&shared_erosion_occup[tidq], erosion_fine);

		if (tidr == 0) {
			erosion_voxels[cell_idx] = shared_erosion_occup[tidq];
		}

		/*
		 * Set the new 4 x (8 cells) starting index for the warp
		 */
		//		thread_idx = atomicAdd (global_warp_counter, WARP_SIZE) + (tidx%WARP_SIZE);
		if (tidx % WARP_SIZE == 0)
			thread_idx = atomicAdd (global_warp_counter, WARP_SIZE);

		thread_idx = __shfl (thread_idx, 0) + tidx % WARP_SIZE;

		if (thread_idx >= 8 * cells_size)
			return;
	} while (true);
}


__global__ void ErodeBySphereTexMipmapPersistentWarpILP (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU erosion_grid,
        int * global_warp_counter,
        float radius) {
	__shared__ unsigned char shared_erosion_occup[EROSION_ILP_BLOCK_SIZE / 8];
	__shared__ float accum[2][6 * EROSION_ILP_BLOCK_SIZE];
	/*
	 * Variables valid accross all traversals
	 */
	NodeTex tex_stack[64];
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned int res = input_grid.res.x;
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = threadIdx.x;
	const int tidq = tidx / 8;
	const unsigned char tidr = tidx % 8;

	do {
		/*
		 * Fetch Warp Data: 4 x (8 cells)
		 */
		unsigned int cell_idx = cells[thread_idx / 8];
		// Compute 3D indices from linear indices
		unsigned int remainder, resxy, id_x, id_y, id_z;
		resxy = res * res;
		id_z = cell_idx / resxy;
		remainder = cell_idx % resxy;
		id_y = remainder / res;
		id_x = remainder % res;

		id_x *= 2; id_y *= 2; id_z *= 2;
		id_x += (0xaa & (1 << (tidr))) ? 1 : 0;
		id_y += (0xcc & (1 << (tidr))) ? 1 : 0;
		id_z += (0xf0 & (1 << (tidr))) ? 1 : 0;

		float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
		unsigned char erosion_occup = 0xff;

		/*
		 * Traversal Part
		 */
		// Allocate traversal stack from thread-local memory,
		// and push VOID to indicate that there are no postponed nodes.
		NodeTex * tex_stack_ptr = tex_stack;
		*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

		// Traverse nodes starting from the 8 sub-trees that thightly
		// intersect the Structuring Element
		NodeTex tex_node = make_node (0, 0, 0, 0);

		do {
			// First retrieve the current node childs geometry occupation
			// pattern.
			unsigned int curr_res = (1 << tex_node.depth);
			float cell_size = max_res >> (tex_node.depth + 1);
			float3 node_query = (1.f / cell_size) * query;
			ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
			                               2 * tex_node.id.y,
			                               2 * tex_node.id.z);
			uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
			                                  (float)tex_node.id.x / curr_res,
			                                  (float)tex_node.id.y / curr_res,
			                                  (float)tex_node.id.z / curr_res,
			                                  max_depth - tex_node.depth - 1);
			unsigned char has_geometry_overlaps = data_tex.y >> 24;
			unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

			// k = 0 -> 0 0 0
			// k = 1 -> 1 0 0
			// k = 2 -> 0 1 0
			// k = 3 -> 1 1 0
			// k = 4 -> 0 0 1
			// k = 5 -> 1 0 1
			// k = 6 -> 0 1 1
			// k = 7 -> 1 1 1
			float3 b_coords = make_float3 (base_tex_node_coords.x,
			                               base_tex_node_coords.y,
			                               base_tex_node_coords.z);
			unsigned char uchar_radius[8];
			uchar_radius[0] = 0x7f & data;
			uchar_radius[1] = 0x7f & (data >> 7);
			uchar_radius[2] = 0x7f & (data >> 14);
			uchar_radius[3] = 0x7f & (data >> 21);
			uchar_radius[4] = 0x7f & (data >> 28);
			uchar_radius[5] = 0x7f & (data >> 35);
			uchar_radius[6] = 0x7f & (data >> 42);
			uchar_radius[7] = 0x7f & (data >> 49);

			float sq_node_se_size[8];
			sq_node_se_size[0] = square ((float)(uchar_radius[0] - 1) / cell_size);
			sq_node_se_size[1] = square ((float)(uchar_radius[1] - 1) / cell_size);
			sq_node_se_size[2] = square ((float)(uchar_radius[2] - 1) / cell_size);
			sq_node_se_size[3] = square ((float)(uchar_radius[3] - 1) / cell_size);
			sq_node_se_size[4] = square ((float)(uchar_radius[4] - 1) / cell_size);
			sq_node_se_size[5] = square ((float)(uchar_radius[5] - 1) / cell_size);
			sq_node_se_size[6] = square ((float)(uchar_radius[6] - 1) / cell_size);
			sq_node_se_size[7] = square ((float)(uchar_radius[7] - 1) / cell_size);

			float sq_d[8];
			sq_d[0] = 0.f; sq_d[1] = 0.f; sq_d[2] = 0.f; sq_d[3] = 0.f;
			sq_d[4] = 0.f; sq_d[5] = 0.f; sq_d[6] = 0.f; sq_d[7] = 0.f;

			// X dimension and min
			accum[0][6 * tidx + 0] = node_query.x < b_coords.x ?
			                         square (node_query.x - b_coords.x) : 0.f;
			accum[1][6 * tidx + 0] = node_query.x < (b_coords.x + 1.f) ?
			                         square (node_query.x - b_coords.x - 1.f) : 0.f;
			// X dimension and max
			accum[0][6 * tidx + 1] = node_query.x > (b_coords.x + 1.f) ?
			                         square (node_query.x - b_coords.x - 1.f) : 0.f;
			accum[1][6 * tidx + 1] = node_query.x > (b_coords.x + 2.f) ?
			                         square (node_query.x - b_coords.x - 2.f) : 0.f;
			// Y dimension and min
			accum[0][6 * tidx + 2] = node_query.y < b_coords.y ?
			                         square (node_query.y - b_coords.y) : 0.f;
			accum[1][6 * tidx + 2] = node_query.y < (b_coords.y + 1.f) ?
			                         square (node_query.y - b_coords.y - 1.f) : 0.f;
			// Y dimension and max
			accum[0][6 * tidx + 3] = node_query.y > (b_coords.y + 1.f) ?
			                         square (node_query.y - b_coords.y - 1.f) : 0.f;
			accum[1][6 * tidx + 3] = node_query.y > (b_coords.y + 2.f) ?
			                         square (node_query.y - b_coords.y - 2.f) : 0.f;
			// Z dimension and min
			accum[0][6 * tidx + 4] = node_query.z < b_coords.z ?
			                         square (node_query.z - b_coords.z) : 0.f;
			accum[1][6 * tidx + 4] = node_query.z < (b_coords.z + 1.f) ?
			                         square (node_query.z - b_coords.z - 1.f) : 0.f;
			// Z dimension and max
			accum[0][6 * tidx + 5] = node_query.z > (b_coords.z + 1.f) ?
			                         square (node_query.z - b_coords.z - 1.f) : 0.f;
			accum[1][6 * tidx + 5] = node_query.z > (b_coords.z + 2.f) ?
			                         square (node_query.z - b_coords.z - 2.f) : 0.f;

			sq_d[0] += accum[0][6 * tidx + 0];
			sq_d[1] += accum[1][6 * tidx + 0];
			sq_d[2] += accum[0][6 * tidx + 0];
			sq_d[3] += accum[1][6 * tidx + 0];
			sq_d[4] += accum[0][6 * tidx + 0];
			sq_d[5] += accum[1][6 * tidx + 0];
			sq_d[6] += accum[0][6 * tidx + 0];
			sq_d[7] += accum[1][6 * tidx + 0];

			sq_d[0] += accum[0][6 * tidx + 1];
			sq_d[1] += accum[1][6 * tidx + 1];
			sq_d[2] += accum[0][6 * tidx + 1];
			sq_d[3] += accum[1][6 * tidx + 1];
			sq_d[4] += accum[0][6 * tidx + 1];
			sq_d[5] += accum[1][6 * tidx + 1];
			sq_d[6] += accum[0][6 * tidx + 1];
			sq_d[7] += accum[1][6 * tidx + 1];

			sq_d[0] += accum[0][6 * tidx + 2];
			sq_d[1] += accum[0][6 * tidx + 2];
			sq_d[2] += accum[1][6 * tidx + 2];
			sq_d[3] += accum[1][6 * tidx + 2];
			sq_d[4] += accum[0][6 * tidx + 2];
			sq_d[5] += accum[0][6 * tidx + 2];
			sq_d[6] += accum[1][6 * tidx + 2];
			sq_d[7] += accum[1][6 * tidx + 2];

			sq_d[0] += accum[0][6 * tidx + 3];
			sq_d[1] += accum[0][6 * tidx + 3];
			sq_d[2] += accum[1][6 * tidx + 3];
			sq_d[3] += accum[1][6 * tidx + 3];
			sq_d[4] += accum[0][6 * tidx + 3];
			sq_d[5] += accum[0][6 * tidx + 3];
			sq_d[6] += accum[1][6 * tidx + 3];
			sq_d[7] += accum[1][6 * tidx + 3];

			sq_d[0] += accum[0][6 * tidx + 4];
			sq_d[1] += accum[0][6 * tidx + 4];
			sq_d[2] += accum[0][6 * tidx + 4];
			sq_d[3] += accum[0][6 * tidx + 4];
			sq_d[4] += accum[1][6 * tidx + 4];
			sq_d[5] += accum[1][6 * tidx + 4];
			sq_d[6] += accum[1][6 * tidx + 4];
			sq_d[7] += accum[1][6 * tidx + 4];

			sq_d[0] += accum[0][6 * tidx + 5];
			sq_d[1] += accum[0][6 * tidx + 5];
			sq_d[2] += accum[0][6 * tidx + 5];
			sq_d[3] += accum[0][6 * tidx + 5];
			sq_d[4] += accum[1][6 * tidx + 5];
			sq_d[5] += accum[1][6 * tidx + 5];
			sq_d[6] += accum[1][6 * tidx + 5];
			sq_d[7] += accum[1][6 * tidx + 5];

			//			if (thread_idx == 0) {
			//				for (int k = 0; k < 8; k++) {
			//					printf ("id : %i %i %i k : %i sq_d: %f\n",
			//									tex_node.id.x, tex_node.id.y, tex_node.id.z,
			//									k, sq_d[k]);
			//				}
			//			}

			if (sq_d[0] < sq_node_se_size[0]
			        && ((has_geometry_overlaps & (1)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[1] < sq_node_se_size[1]
			        && ((has_geometry_overlaps & (1 << 1)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[2] < sq_node_se_size[2]
			        && ((has_geometry_overlaps & (1 << 2)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[3] < sq_node_se_size[3]
			        && ((has_geometry_overlaps & (1 << 3)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[4] < sq_node_se_size[4]
			        && ((has_geometry_overlaps & (1 << 4)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[5] < sq_node_se_size[5]
			        && ((has_geometry_overlaps & (1 << 5)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[6] < sq_node_se_size[6]
			        && ((has_geometry_overlaps & (1 << 6)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			if (sq_d[7] < sq_node_se_size[7]
			        && ((has_geometry_overlaps & (1 << 7)) != 0)) {
				if ((tex_node.depth + 1) != max_depth) {
					*tex_stack_ptr++ = make_node (tex_node.depth + 1,
					                              base_tex_node_coords.x + 1,
					                              base_tex_node_coords.y + 1,
					                              base_tex_node_coords.z + 1);
				} else {
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				}
			}

			// We pop the next node to process
			tex_node = *--tex_stack_ptr;
		} while (tex_node.depth != -1);

		// Shared Memory Access by atomics
		unsigned char erosion_fine = ~(1 << tidr);

		// Shared Memory Initialization
		if (tidr == 0) {
			shared_erosion_occup[tidq] = 0xff;
		}
		//		unsigned int sync_group_mask = 0xff << (8*tidr);
		if (erosion_occup == 0x00)
			myAtomicAnd (&shared_erosion_occup[tidq], erosion_fine);

		if (tidr == 0) {
			erosion_voxels[cell_idx] = shared_erosion_occup[tidq];
		}

		/*
		 * Set the new 4 x (8 cells) starting index for the warp
		 */
		//		thread_idx = atomicAdd (global_warp_counter, WARP_SIZE) + (tidx%WARP_SIZE);
		if (tidx % WARP_SIZE == 0)
			thread_idx = atomicAdd (global_warp_counter, WARP_SIZE);

		thread_idx = __shfl (thread_idx, 0) + tidx % WARP_SIZE;

		if (thread_idx >= 8 * cells_size)
			return;
	} while (true);
}

__global__ void ErodeBySphereTexMipmapPersistentWarp (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU erosion_grid,
        int * global_warp_counter,
        float radius) {
	__shared__ unsigned char shared_erosion_occup[EROSION_BLOCK_SIZE / 8];
	/*
	 * Variables valid accross all traversals
	 */
	NodeTex tex_stack[64];
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned int res = input_grid.res.x;
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = threadIdx.x;
	const int tidq = tidx / 8;
	const unsigned char tidr = tidx % 8;

	do {
		/*
		 * Fetch Warp Data: 4 x (8 cells)
		 */
		unsigned int cell_idx = cells[thread_idx / 8];
		// Compute 3D indices from linear indices
		unsigned int remainder, resxy, id_x, id_y, id_z;
		resxy = res * res;
		id_z = cell_idx / resxy;
		remainder = cell_idx % resxy;
		id_y = remainder / res;
		id_x = remainder % res;

		id_x *= 2; id_y *= 2; id_z *= 2;
		id_x += (0xaa & (1 << (tidr))) ? 1 : 0;
		id_y += (0xcc & (1 << (tidr))) ? 1 : 0;
		id_z += (0xf0 & (1 << (tidr))) ? 1 : 0;

		float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
		unsigned char erosion_occup = 0xff;

		/*
		 * Traversal Part
		 */
		// Allocate traversal stack from thread-local memory,
		// and push VOID to indicate that there are no postponed nodes.
		NodeTex * tex_stack_ptr = tex_stack;
		*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

		// Traverse nodes starting from the 8 sub-trees that thightly
		// intersect the Structuring Element
		NodeTex tex_node = make_node (0, 0, 0, 0);

		do {
			// First retrieve the current node childs geometry occupation
			// pattern.
			unsigned int curr_res = (1 << tex_node.depth);
			float cell_size = max_res >> (tex_node.depth + 1);
			float3 node_query = (1.f / cell_size) * query;
			ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
			                               2 * tex_node.id.y,
			                               2 * tex_node.id.z);
			uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
			                                  (float)tex_node.id.x / curr_res,
			                                  (float)tex_node.id.y / curr_res,
			                                  (float)tex_node.id.z / curr_res,
			                                  max_depth - tex_node.depth - 1);
			unsigned char has_geometry_overlaps = data_tex.y >> 24;
			unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);

			for (int k = 0; k < 8; k++) {
				unsigned char mask_k = (1 << k);
				unsigned char uchar_radius = 0x7f & data;
				ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
				float sq_node_se_size = square ((float)(uchar_radius - 1) / cell_size);
				bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
				                        HasCubeSEOverlap (node_query, child_tex_node_coords, sq_node_se_size);

				NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);
				if (node_has_overlap && tex_node_child.depth != max_depth) {
					// If the node overlap the SE and IS NOT a leaf
					// we push it into the stack
					*tex_stack_ptr++ = tex_node_child;
				} else if (node_has_overlap && tex_node_child.depth == max_depth) {
					// If the node overlap the SE and IS a leaf
					// we splat the voxel and return from the thread
					//				dilation_voxels[i] = 0xff;
					erosion_occup = 0x00;
					*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
					//				return;
				}
				data = data >> 7;
			}

			// We pop the next node to process
			tex_node = *--tex_stack_ptr;
		} while (tex_node.depth != -1);

		// Shared Memory Access by atomics
		unsigned char erosion_fine = ~(1 << tidr);

		// Shared Memory Initialization
		if (tidr == 0) {
			shared_erosion_occup[tidq] = 0xff;
		}
		//		unsigned int sync_group_mask = 0xff << (8*tidr);
		if (erosion_occup == 0x00)
			myAtomicAnd (&shared_erosion_occup[tidq], erosion_fine);

		if (tidr == 0) {
			erosion_voxels[cell_idx] = shared_erosion_occup[tidq];
		}

		/*
		 * Set the new 4 x (8 cells) starting index for the warp
		 */
		//		thread_idx = atomicAdd (global_warp_counter, WARP_SIZE) + (tidx%WARP_SIZE);
		if (tidx % WARP_SIZE == 0)
			thread_idx = atomicAdd (global_warp_counter, WARP_SIZE);

		thread_idx = __shfl (thread_idx, 0) + tidx % WARP_SIZE;

		if (thread_idx >= 8 * cells_size)
			return;
	} while (true);
}

__global__ void ErodeBySphereTexMipmap (unsigned int * cells,
                                        unsigned int cells_size,
                                        GridGPU input_grid,
                                        GridGPU erosion_grid,
                                        unsigned int * morpho_centroids_grid,
                                        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	int max_depth = input_grid.max_mipmap_level;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	unsigned int counter = 0;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		float tex_depth = 8 - tex_node.depth;
		unsigned int curr_res = (1 << tex_node.depth);
		uint2 data_tex = tex3DLod<uint2> (input_grid.tex_mipmap,
		                                  (float)tex_node.id.x / curr_res,
		                                  (float)tex_node.id.y / curr_res,
		                                  (float)tex_node.id.z / curr_res,
		                                  tex_depth);
		int has_geometry_overlaps = data_tex.y >> 24;
		unsigned long long data = *reinterpret_cast<unsigned long long*>(&data_tex);
		unsigned long long mask_size = 0x7f;
		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;

		for (int k = 0; k < 8; k++) {
			unsigned char uchar_radius = mask_size & data;
			float sq_node_se_size = square ((float)(uchar_radius - 1) / cell_size);
			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords, mask_k);
			bool node_has_overlap = (has_geometry_overlaps & mask_k) &&
			                        HasCubeSEOverlap (node_query, child_tex_node_coords, sq_node_se_size);

			NodeTex tex_node_child = make_node (tex_node.depth + 1, child_tex_node_coords);
			if (node_has_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				//				return;
			}
			data = data >> 7;
		}
		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char erosion_fine = ~(1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_erosion_occup[threadIdx.x / 8] = 0xff;
	}
	__syncthreads ();

	if (erosion_occup == 0x00)
		myAtomicAnd (&shared_erosion_occup[threadIdx.x / 8], erosion_fine);
	__syncthreads ();

	if (tid == 0) {
		erosion_voxels[i] = shared_erosion_occup[threadIdx.x / 8];
	}
}

__global__ void ErodeBySphereTexDualMipmap (unsigned int * cells,
        unsigned int cells_size,
        GridGPU input_grid,
        GridGPU erosion_grid,
        unsigned int * morpho_centroids_grid,
        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	radius -= 1.f;

	unsigned char * erosion_voxels = erosion_grid.voxels;
	float3 query = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	int max_depth = input_grid.max_mipmap_depth;
	int max_res = 1 << max_depth;
	unsigned char erosion_occup = 0xff;

	// Allocate traversal stack from thread-local memory,
	// and push VOID to indicate that there are no postponed nodes.
	NodeTex tex_stack[64];
	NodeTex * tex_stack_ptr = tex_stack;
	*tex_stack_ptr++ = make_node (-1, 0, 0, 0);

	// Traverse nodes starting from the 8 sub-trees that thightly
	// intersect the Structuring Element
	NodeTex tex_node = make_node (0, 0, 0, 0);

	unsigned int counter = 0;
	//	bool early_found = false;
	do {
		counter++;
		// First retrieve the current node childs geometry occupation
		// pattern.
		float tex_depth = 8 - tex_node.depth;
		unsigned int curr_res = (1 << tex_node.depth);
		int4 has_geometry_overlaps_tex = tex3DLod<int4> (input_grid.tex_dual_mipmap,
		                                 (float)tex_node.id.x / curr_res,
		                                 (float)tex_node.id.y / curr_res,
		                                 (float)tex_node.id.z / curr_res,
		                                 tex_depth);
		ulonglong2 data_tex = *reinterpret_cast<ulonglong2*> (&has_geometry_overlaps_tex);
		unsigned long long data = data_tex.y;
		int has_geometry_overlaps = data >> 56;
		unsigned long long mask_size = 0x7f;

		ushort3 base_tex_node_coords = make_ushort3 (2 * tex_node.id.x,
		                               2 * tex_node.id.y,
		                               2 * tex_node.id.z);
		float cell_size = max_res >> (tex_node.depth + 1);
		float3 node_query = (1.f / cell_size) * query;

		for (int k = 0; k < 8; k++) {
			unsigned char uchar_radius = mask_size & data;
			float sq_node_inner_se_size = 0.f;
			float sq_node_outer_se_size = uchar_radius;
			sq_node_outer_se_size /= cell_size;
			sq_node_inner_se_size = fmaxf (0.f, sq_node_outer_se_size - 1.73205080757f);
			sq_node_outer_se_size *= sq_node_outer_se_size;
			sq_node_inner_se_size *= sq_node_inner_se_size;

			if (blockIdx.x == 1 && threadIdx.x == 36 && k == 0) {
				printf ("count : %i se_inner_size %f se_outer_size %f\n", counter,
				        sqrt(sq_node_inner_se_size), sqrt (sq_node_outer_se_size));
			}

			unsigned int mask_k = (1 << k);
			ushort3 child_tex_node_coords = Z8Coordinates (base_tex_node_coords,
			                                mask_k);
			float3 child_coords = make_float3 (child_tex_node_coords.x,
			                                   child_tex_node_coords.y,
			                                   child_tex_node_coords.z);
			bool node_has_geometry_overlap, node_has_outer_overlap, node_has_inner_overlap;

			node_has_geometry_overlap = (has_geometry_overlaps & mask_k);

			HasCubeSEOverlap (node_query, child_coords,
			                  sq_node_inner_se_size, sq_node_outer_se_size,
			                  node_has_inner_overlap, node_has_outer_overlap);

			//			node_has_inner_overlap = false;
			node_has_inner_overlap = node_has_inner_overlap && node_has_geometry_overlap;
			node_has_outer_overlap = node_has_outer_overlap && node_has_geometry_overlap;

			NodeTex tex_node_child = make_node (tex_node.depth + 1,
			                                    child_tex_node_coords.x,
			                                    child_tex_node_coords.y,
			                                    child_tex_node_coords.z);

			if (node_has_inner_overlap) {
				//				early_found = true;
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
			} else if (node_has_outer_overlap && tex_node_child.depth != max_depth) {
				// If the node overlap the SE and IS NOT a leaf
				// we push it into the stack
				*tex_stack_ptr++ = tex_node_child;
			} else if (node_has_outer_overlap && tex_node_child.depth == max_depth) {
				// If the node overlap the SE and IS a leaf
				// we splat the voxel and return from the thread
				//				dilation_voxels[i] = 0xff;
				erosion_occup = 0x00;
				*tex_stack_ptr++ = make_node (-1, 0, 0, 0);
				//				return;
			}
			data = data >> 7;
		}
		// We pop the next node to process
		tex_node = *--tex_stack_ptr;
	} while (tex_node.depth != -1);

	//	if (blockIdx.x == 1) {
	//		printf ("i : %i count : %i early : %i\n", threadIdx.x, counter, early_found);
	//	}
	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char erosion_fine = ~(1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_erosion_occup[threadIdx.x / 8] = 0xff;
	}
	__syncthreads ();

	if (erosion_occup == 0x00)
		myAtomicAnd (&shared_erosion_occup[threadIdx.x / 8], erosion_fine);
	__syncthreads ();

	if (tid == 0) {
		erosion_voxels[i] = shared_erosion_occup[threadIdx.x / 8];
	}

}

__global__ void ComputeMCMData (unsigned int * cells,
                                unsigned int cells_size,
                                GridGPU input_grid,
                                unsigned int * mcm_indices,
                                unsigned char * mcm_values,
                                unsigned int * mcm_non_empty) {
	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx];
	//	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	//	id_x *= 2; id_y *= 2; id_z *= 2;
	//	id_x += (0xaa & (1 << (threadIdx.x%8))) ? 1 : 0;
	//	id_y += (0xcc & (1 << (threadIdx.x%8))) ? 1 : 0;
	//	id_z += (0xf0 & (1 << (threadIdx.x%8))) ? 1 : 0;

	unsigned char * voxels = input_grid.voxels;
	unsigned char val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7;
	unsigned char v0, v1, v2, v3, v4, v5, v6, v7;

	val_0 = 0; val_1 = 0; val_2 = 0; val_3 = 0;
	val_4 = 0; val_5 = 0; val_6 = 0; val_7 = 0;

	v0 = voxels[resxy * (id_z) + res * (id_y) + (id_x)];
	v1 = voxels[resxy * (id_z) + res * (id_y) + (id_x + 1)];
	v2 = voxels[resxy * (id_z) + res * (id_y + 1) + (id_x)];
	v3 = voxels[resxy * (id_z) + res * (id_y + 1) + (id_x + 1)];
	v4 = voxels[resxy * (id_z + 1) + res * (id_y) + (id_x)];
	v5 = voxels[resxy * (id_z + 1) + res * (id_y) + (id_x + 1)];
	v6 = voxels[resxy * (id_z + 1) + res * (id_y + 1) + (id_x)];
	v7 = voxels[resxy * (id_z + 1) + res * (id_y + 1) + (id_x + 1)];

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

	unsigned int ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7;
	ind_0 = 4 * resxy * (2 * id_z) + 2 * res * (2 * id_y) + (2 * id_x);
	ind_1 = 4 * resxy * (2 * id_z) + 2 * res * (2 * id_y) + (2 * id_x + 1);
	ind_2 = 4 * resxy * (2 * id_z) + 2 * res * (2 * id_y + 1) + (2 * id_x);
	ind_3 = 4 * resxy * (2 * id_z) + 2 * res * (2 * id_y + 1) + (2 * id_x + 1);
	ind_4 = 4 * resxy * (2 * id_z + 1) + 2 * res * (2 * id_y) + (2 * id_x);
	ind_5 = 4 * resxy * (2 * id_z + 1) + 2 * res * (2 * id_y) + (2 * id_x + 1);
	ind_6 = 4 * resxy * (2 * id_z + 1) + 2 * res * (2 * id_y + 1) + (2 * id_x);
	ind_7 = 4 * resxy * (2 * id_z + 1) + 2 * res * (2 * id_y + 1) + (2 * id_x + 1);

	mcm_indices[8 * thread_idx + 0] = ind_0;
	mcm_indices[8 * thread_idx + 1] = ind_1;
	mcm_indices[8 * thread_idx + 2] = ind_2;
	mcm_indices[8 * thread_idx + 3] = ind_3;
	mcm_indices[8 * thread_idx + 4] = ind_4;
	mcm_indices[8 * thread_idx + 5] = ind_5;
	mcm_indices[8 * thread_idx + 6] = ind_6;
	mcm_indices[8 * thread_idx + 7] = ind_7;

	mcm_values[8 * thread_idx + 0] = val_0;
	mcm_values[8 * thread_idx + 1] = val_1;
	mcm_values[8 * thread_idx + 2] = val_2;
	mcm_values[8 * thread_idx + 3] = val_3;
	mcm_values[8 * thread_idx + 4] = val_4;
	mcm_values[8 * thread_idx + 5] = val_5;
	mcm_values[8 * thread_idx + 6] = val_6;
	mcm_values[8 * thread_idx + 7] = val_7;

	mcm_non_empty[8 * thread_idx + 0] = (val_0 != 0) && (val_0 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 1] = (val_1 != 0) && (val_1 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 2] = (val_2 != 0) && (val_2 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 3] = (val_3 != 0) && (val_3 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 4] = (val_4 != 0) && (val_4 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 5] = (val_5 != 0) && (val_5 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 6] = (val_6 != 0) && (val_6 != 255) ? 1 : 0;
	mcm_non_empty[8 * thread_idx + 7] = (val_7 != 0) && (val_7 != 255) ? 1 : 0;
}

__global__ void CompactMCMData (unsigned char * mcm_values,
                                unsigned char * mcm_compact_values,
                                unsigned int * mcm_indices,
                                unsigned int * mcm_compact_indices,
                                unsigned int * compact_neigh_morpho_centroids,
                                unsigned int * morpho_centroids_grid,
                                unsigned int * mcm_non_empty_scan,
                                unsigned int num_mcm_cells,
                                uint3 res
                               ) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < num_mcm_cells)
		if (mcm_values[i] != 0 && mcm_values[i] != 255) {
			//			unsigned int mcm_index = mcm_indices[i];
			unsigned int non_empty_index = mcm_non_empty_scan[i];
			mcm_compact_values[non_empty_index] = mcm_values[i];
			mcm_compact_indices[non_empty_index] = mcm_indices[i];

			//			// Compute 3D indices from linear indices
			//			unsigned int remainder, resxy, id_x, id_y, id_z, resx;
			//			resxy = 4*res.x*res.y;
			//			resx = 2*res.x;
			//			id_z = mcm_index/resxy;
			//			remainder = mcm_index % resxy;
			//			id_y = remainder/(resx);
			//			id_x = remainder % (resx);
			//
			//			unsigned int ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7;
			//			unsigned int c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7;
			//			ind_0 = resxy*(id_z) + resx*(id_y) + (id_x);
			//			ind_1 = resxy*(id_z) + resx*(id_y) + (id_x + 1);
			//			ind_2 = resxy*(id_z) + resx*(id_y + 1) + (id_x + 1);
			//			ind_3 = resxy*(id_z) + resx*(id_y + 1) + (id_x);
			//			ind_4 = resxy*(id_z + 1) + resx*(id_y) + (id_x);
			//			ind_5 = resxy*(id_z + 1) + resx*(id_y) + (id_x + 1);
			//			ind_6 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x + 1);
			//			ind_7 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x);
			//			c_0 = morpho_centroids_grid[ind_0];
			//			c_1 = morpho_centroids_grid[ind_1];
			//			c_2 = morpho_centroids_grid[ind_2];
			//			c_3 = morpho_centroids_grid[ind_3];
			//			c_4 = morpho_centroids_grid[ind_4];
			//			c_5 = morpho_centroids_grid[ind_5];
			//			c_6 = morpho_centroids_grid[ind_6];
			//			c_7 = morpho_centroids_grid[ind_7];
			//			compact_neigh_morpho_centroids[8*non_empty_index + 0] = c_0;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 1] = c_1;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 2] = c_2;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 3] = c_3;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 4] = c_4;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 5] = c_5;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 6] = c_6;
			//			compact_neigh_morpho_centroids[8*non_empty_index + 7] = c_7;

			//			if (non_empty_index == 10000) {
			//				for (int k = 0; k < 8; k++) {
			//					printf ("centroid : %i\n", morpho_centroids_grid[ind_0]);
			//				}
			//				printf ("value : x%.2x\n", mcm_values[i]);
			//				printf ("pos : %i %i %i %i\n", id_x, id_y, id_z, ind_0);
			//			}
		}
}

//__global__ void BuildMCMNeighborsX (unsigned int * indices,
//																		unsigned int * neigh_indices,
//																		unsigned int num_indices,
//																		uint3 res) {
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//	if (i >= num_indices)
//		return;
//
//	unsigned int index = indices[i];
//
//	unsigned int ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7;
//	ind_0 = resxy*(id_z) + resx*(id_y) + (id_x);
//	ind_1 = resxy*(id_z) + resx*(id_y) + (id_x + 1);
//	ind_2 = resxy*(id_z) + resx*(id_y + 1) + (id_x + 1);
//	ind_3 = resxy*(id_z) + resx*(id_y + 1) + (id_x);
//	ind_4 = resxy*(id_z + 1) + resx*(id_y) + (id_x);
//	ind_5 = resxy*(id_z + 1) + resx*(id_y) + (id_x + 1);
//	ind_6 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x + 1);
//	ind_7 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x);
//	neigh_indices[8*i + 0] = grid_indices[ind_0];
//	neigh_indices[8*i + 1] = grid_indices[ind_1];
//	neigh_indices[8*i + 2] = grid_indices[ind_2];
//	neigh_indices[8*i + 3] = grid_indices[ind_3];
//	neigh_indices[8*i + 4] = grid_indices[ind_4];
//	neigh_indices[8*i + 5] = grid_indices[ind_5];
//	neigh_indices[8*i + 6] = grid_indices[ind_6];
//	neigh_indices[8*i + 7] = grid_indices[ind_7];
//}

__global__ void SplatIndices (unsigned int * indices,
                              unsigned int * grid_indices,
                              unsigned int num_indices) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_indices)
		return;
	grid_indices[indices[i]] = i;
}

__global__ void BuildNeighbors (unsigned int * indices,
                                unsigned int * neigh_indices,
                                unsigned int * mapping_indices,
                                unsigned int * sorted_indices,
                                unsigned int num_indices,
                                uint3 res) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_indices)
		return;

	unsigned int index = indices[i];

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z, resx;
	resxy = 4 * res.x * res.y;
	resx = 2 * res.x;
	id_z = index / resxy;
	remainder = index % resxy;
	id_y = remainder / (resx);
	id_x = remainder % (resx);

	//	unsigned int j = 4*(id_z%2) + 2*(id_y%2) + id_x%2;
	unsigned int sorted_id;
	unsigned int local_sorted_indices[8];

	sorted_id = mapping_indices[i];

	local_sorted_indices[0] = sorted_indices[sorted_id + 0];
	local_sorted_indices[1] = sorted_indices[sorted_id + 1];
	local_sorted_indices[2] = sorted_indices[sorted_id + 2];
	local_sorted_indices[3] = sorted_indices[sorted_id + 3];
	local_sorted_indices[4] = sorted_indices[sorted_id + 4];
	local_sorted_indices[5] = sorted_indices[sorted_id + 5];
	local_sorted_indices[6] = sorted_indices[sorted_id + 6];
	local_sorted_indices[7] = sorted_indices[sorted_id + 7];

	unsigned int ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7;
	//	ind_0 = resxy*(id_z) + resx*(id_y) + (id_x);
	//	ind_1 = resxy*(id_z) + resx*(id_y) + (id_x + 1);
	//	ind_2 = resxy*(id_z) + resx*(id_y + 1) + (id_x + 1);
	//	ind_3 = resxy*(id_z) + resx*(id_y + 1) + (id_x);
	//	ind_4 = resxy*(id_z + 1) + resx*(id_y) + (id_x);
	//	ind_5 = resxy*(id_z + 1) + resx*(id_y) + (id_x + 1);
	//	ind_6 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x + 1);
	//	ind_7 = resxy*(id_z + 1) + resx*(id_y + 1) + (id_x);
	//
	ind_0 = res.x * res.y * (id_z / 2) + res.x * (id_y / 2) + (id_x / 2);
	ind_1 = res.x * res.y * (id_z / 2) + res.x * (id_y / 2) + ((id_x + 1) / 2);
	ind_2 = res.x * res.y * (id_z / 2) + res.x * ((id_y + 1) / 2) + ((id_x + 1) / 2);
	ind_3 = res.x * res.y * (id_z / 2) + res.x * ((id_y + 1) / 2) + (id_x / 2);
	ind_4 = res.x * res.y * ((id_z + 1) / 2) + res.x * (id_y / 2) + (id_x / 2);
	ind_5 = res.x * res.y * ((id_z + 1) / 2) + res.x * (id_y / 2) + ((id_x + 1) / 2);
	ind_6 = res.x * res.y * ((id_z + 1) / 2) + res.x * ((id_y + 1) / 2) + ((id_x + 1) / 2);
	ind_7 = res.x * res.y * ((id_z + 1) / 2) + res.x * ((id_y + 1) / 2) + (id_x / 2);

	ind_0 += 4 * (id_z % 2) + 2 * (id_y % 2) + (id_x % 2);
	ind_1 += 4 * (id_z % 2) + 2 * (id_y % 2) + ((id_x + 1) % 2);
	ind_2 += 4 * (id_z % 2) + 2 * ((id_y + 1) % 2) + ((id_x + 1) % 2);
	ind_3 += 4 * (id_z % 2) + 2 * ((id_y + 1) % 2) + (id_x % 2);
	ind_4 += 4 * ((id_z + 1) % 2) + 2 * (id_y % 2) + (id_x % 2);
	ind_5 += 4 * ((id_z + 1) % 2) + 2 * (id_y % 2) + ((id_x + 1) % 2);
	ind_6 += 4 * ((id_z + 1) % 2) + 2 * ((id_y + 1) % 2) + ((id_x + 1) % 2);
	ind_7 += 4 * ((id_z + 1) % 2) + 2 * ((id_y + 1) % 2) + (id_x % 2);

	neigh_indices[8 * i + 0] = local_sorted_indices[0] == ind_0 ? ind_0 : 0xffffffff;
	neigh_indices[8 * i + 1] = local_sorted_indices[1] == ind_1 ? ind_1 : 0xffffffff;
	neigh_indices[8 * i + 2] = local_sorted_indices[2] == ind_2 ? ind_2 : 0xffffffff;
	neigh_indices[8 * i + 3] = local_sorted_indices[3] == ind_3 ? ind_3 : 0xffffffff;
	neigh_indices[8 * i + 4] = local_sorted_indices[4] == ind_4 ? ind_4 : 0xffffffff;
	neigh_indices[8 * i + 5] = local_sorted_indices[5] == ind_5 ? ind_5 : 0xffffffff;
	neigh_indices[8 * i + 6] = local_sorted_indices[6] == ind_6 ? ind_6 : 0xffffffff;
	neigh_indices[8 * i + 7] = local_sorted_indices[7] == ind_7 ? ind_7 : 0xffffffff;
}

__global__ void BuildNeighbors (unsigned int * indices,
                                unsigned int * neigh_indices,
                                unsigned int * grid_indices,
                                unsigned int num_indices,
                                uint3 res) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_indices)
		return;

	unsigned int index = indices[i];

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z, resx;
	resxy = 4 * res.x * res.y;
	resx = 2 * res.x;
	id_z = index / resxy;
	remainder = index % resxy;
	id_y = remainder / (resx);
	id_x = remainder % (resx);

	unsigned int ind_0, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7;
	ind_0 = resxy * (id_z) + resx * (id_y) + (id_x);
	ind_1 = resxy * (id_z) + resx * (id_y) + (id_x + 1);
	ind_2 = resxy * (id_z) + resx * (id_y + 1) + (id_x + 1);
	ind_3 = resxy * (id_z) + resx * (id_y + 1) + (id_x);
	ind_4 = resxy * (id_z + 1) + resx * (id_y) + (id_x);
	ind_5 = resxy * (id_z + 1) + resx * (id_y) + (id_x + 1);
	ind_6 = resxy * (id_z + 1) + resx * (id_y + 1) + (id_x + 1);
	ind_7 = resxy * (id_z + 1) + resx * (id_y + 1) + (id_x);
	neigh_indices[8 * i + 0] = grid_indices[ind_0];
	neigh_indices[8 * i + 1] = grid_indices[ind_1];
	neigh_indices[8 * i + 2] = grid_indices[ind_2];
	neigh_indices[8 * i + 3] = grid_indices[ind_3];
	neigh_indices[8 * i + 4] = grid_indices[ind_4];
	neigh_indices[8 * i + 5] = grid_indices[ind_5];
	neigh_indices[8 * i + 6] = grid_indices[ind_6];
	neigh_indices[8 * i + 7] = grid_indices[ind_7];
}

__global__ void ComputeMeshNormals (float3 * positions,
                                    float3 * normals,
                                    uint3 * triangles,
                                    unsigned int num_vertices,
                                    unsigned int num_triangles) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_triangles)
		return;

	uint3 triangle = triangles[i];

	float3 p0 = positions[triangle.x];
	float3 p1 = positions[triangle.y];
	float3 p2 = positions[triangle.z];

	//	float weight, sumWeight;
	float3 N = crossProduct (p1 - p0, p2 - p0);
	atomicAdd (&normals[triangle.x].x, N.x);
	atomicAdd (&normals[triangle.x].y, N.y);
	atomicAdd (&normals[triangle.x].z, N.z);
	atomicAdd (&normals[triangle.y].x, N.x);
	atomicAdd (&normals[triangle.y].y, N.y);
	atomicAdd (&normals[triangle.y].z, N.z);
	atomicAdd (&normals[triangle.z].x, N.x);
	atomicAdd (&normals[triangle.z].y, N.y);
	atomicAdd (&normals[triangle.z].z, N.z);
}

__global__ void NormalizeMeshNormals (float3 * positions,
                                      float3 * normals,
                                      uint3 * triangles,
                                      unsigned int num_vertices,
                                      unsigned int num_triangles) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_vertices) {
		float3 normal = normals[i];
		normalize (normal);
		normals[i] = normal;
	}
}

__global__ void BilateralMeshFiltering (float3 * positions,
                                        float3 * normals,
                                        uint3 * triangles,
                                        float3 * bilateral_positions,
                                        float3 * bilateral_normals,
                                        float * bilateral_weights,
                                        unsigned int num_vertices,
                                        unsigned int num_triangles,
                                        float sigma_p, float sigma_n) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_triangles)
		return;

	//	float sigma_p = 2.f;
	float sq_sigma_p = sigma_p * sigma_p;
	//	float sigma_n = 0.2f;
	float sq_sigma_n = sigma_n * sigma_n;

	uint3 triangle = triangles[i];

	float3 n0 = normals[triangle.x];
	float3 n1 = normals[triangle.y];
	float3 n2 = normals[triangle.z];
	float3 p0 = positions[triangle.x];
	float3 p1 = positions[triangle.y];
	float3 p2 = positions[triangle.z];

	float weight, sumWeight;
	float3 sumN;

	weight = distanceS (p0, p1) / sq_sigma_p + distanceS (n0, n1) / sq_sigma_n;
	weight = exp (-weight);
	sumN = weight * n1;
	sumWeight = weight;
	weight = distanceS (p0, p2) / sq_sigma_p + distanceS (n0, n2) / sq_sigma_n;
	weight = exp (-weight);
	sumN = sumN + weight * n2;
	sumWeight += weight;
	atomicAdd (&bilateral_normals[triangle.x].x, sumN.x);
	atomicAdd (&bilateral_normals[triangle.x].y, sumN.y);
	atomicAdd (&bilateral_normals[triangle.x].z, sumN.z);

	weight = distanceS (p1, p0) / sq_sigma_p + distanceS (n1, n0) / sq_sigma_n;
	weight = exp (-weight);
	sumN = weight * n0;
	sumWeight = weight;
	weight = distanceS (p1, p2) / sq_sigma_p + distanceS (n1, n2) / sq_sigma_n;
	weight = exp (-weight);
	sumN = sumN + weight * n2;
	sumWeight += weight;
	atomicAdd (&bilateral_normals[triangle.y].x, sumN.x);
	atomicAdd (&bilateral_normals[triangle.y].y, sumN.y);
	atomicAdd (&bilateral_normals[triangle.y].z, sumN.z);

	weight = distanceS (p2, p0) / sq_sigma_p + distanceS (n2, n0) / sq_sigma_n;
	weight = exp (-weight);
	sumN = weight * n0;
	sumWeight = weight;
	weight = distanceS (p2, p1) / sq_sigma_p + distanceS (n2, n1) / sq_sigma_n;
	weight = exp (-weight);
	sumN = sumN + weight * n1;
	sumWeight += weight;
	atomicAdd (&bilateral_normals[triangle.z].x, sumN.x);
	atomicAdd (&bilateral_normals[triangle.z].y, sumN.y);
	atomicAdd (&bilateral_normals[triangle.z].z, sumN.z);
}

__global__ void BilateralMeshFilteringNormalize (float3 * positions,
        float3 * normals,
        uint3 * triangles,
        float3 * bilateral_positions,
        float3 * bilateral_normals,
        float * bilateral_weights,
        unsigned int num_vertices,
        unsigned int num_triangles) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_vertices) {
		float3 bilateral_normal = bilateral_normals[i];
		normalize (bilateral_normal);
		normals[i] = bilateral_normal;
	}
}

void GMorpho::ErodeAllocation () {
	if (global_warp_counter_ == NULL)
		cudaMalloc (&global_warp_counter_, sizeof (int));
}

void GMorpho::ErodeBySphereMipmap (const ScaleField & scale_field) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	//	unsigned int block_size;
	float se_size = scale_field.global_scale ();

	// Build a Mipmap on the dilation contour
	time1 = GET_TIME ();
	dilation_contour_grid_.CopyVoxelsFrom (dilation_grid_);
	dilation_contour_grid_.TransformToContour ();
	dilation_contour_grid_.BuildTexMipmaps (scale_field);
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "build mipmap on contour in "
	          << time2 - time1 << " ms." << std::endl;

	// Run an erosion at the base resolution
	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);
	time1 = GET_TIME ();
	ErodeBySphereTexMipmapBaseILP2 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "base erosion in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * closing_contour_base_tight = NULL;
	unsigned int closing_contour_base_tight_size = 0;
	//	closing_grid_.ComputeContourBaseTight (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);
	closing_grid_.ComputeContourBase (closing_contour_base_tight,
	                                  closing_contour_base_tight_size);
	//	closing_grid_.ComputeContourBaseConservative (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);

	// Run an Erosion at full resolution only on the
	// contour cells computed at base resolution
	//	block_size = 256;
	int num_blocks = 100;
	int host_global_warp_counter = num_blocks * EROSION_ILP_BLOCK_SIZE;

	//	if (global_warp_counter_ == NULL)
	//		cudaMalloc (&global_warp_counter_, sizeof (int));
	cudaMemcpy (global_warp_counter_, &host_global_warp_counter,
	            sizeof (int), cudaMemcpyHostToDevice);
	CheckCUDAError ();

	time1 = GET_TIME ();
	ErodeBySphereTexMipmapPersistentWarpILP <<< num_blocks, EROSION_ILP_BLOCK_SIZE>>>
	(closing_contour_base_tight,
	 closing_contour_base_tight_size,
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 global_warp_counter_);
	//	ErodeBySphereTexMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 NULL,
	//			 floor (2.f*se_size/cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "erosion in "
	          << time2 - time1 << " ms." << std::endl;
}

void GMorpho::ErodeBySphereMipmap (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	//	unsigned int block_size;

	// Build a Mipmap on the dilation contour
	time1 = GET_TIME ();
	dilation_contour_grid_.CopyVoxelsFrom (dilation_grid_);
	dilation_contour_grid_.TransformToContour ();
	dilation_contour_grid_.BuildTexMipmaps (se_size);
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "build mipmap on contour in "
	          << time2 - time1 << " ms." << std::endl;

	// Run an erosion at the base resolution
	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);
	time1 = GET_TIME ();
	ErodeBySphereTexMipmapBaseILP2 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "base erosion in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * closing_contour_base_tight = NULL;
	unsigned int closing_contour_base_tight_size = 0;
	//	closing_grid_.ComputeContourBaseTight (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);
	closing_grid_.ComputeContourBase (closing_contour_base_tight,
	                                  closing_contour_base_tight_size);
	//	closing_grid_.ComputeContourBaseConservative (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);

	// Run an Erosion at full resolution only on the
	// contour cells computed at base resolution
	//	block_size = 256;
	int num_blocks = 100;
	int host_global_warp_counter = num_blocks * EROSION_ILP_BLOCK_SIZE;

	if (global_warp_counter_ == NULL)
		cudaMalloc (&global_warp_counter_, sizeof (int));
	cudaMemcpy (global_warp_counter_, &host_global_warp_counter,
	            sizeof (int), cudaMemcpyHostToDevice);
	CheckCUDAError ();

	time1 = GET_TIME ();
	ErodeBySphereTexMipmapPersistentWarpILP <<< num_blocks, EROSION_ILP_BLOCK_SIZE>>>
	(closing_contour_base_tight,
	 closing_contour_base_tight_size,
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 global_warp_counter_,
	 floor (2.f * se_size / cell_size_));
	//	ErodeBySphereTexMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 NULL,
	//			 floor (2.f*se_size/cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "erosion in "
	          << time2 - time1 << " ms." << std::endl;

}

void GMorpho::ExtractClosingMeshAllocation () {

	bool is_allocated = false;

	// Allocate a 2x2x2 full grid of uints
	unsigned int grid_size = res_[0] * res_[1] * res_[2];
	if (grid_2x2x2_uint_ == NULL)
		is_allocated = false;
	else
		is_allocated = true;

	if (!is_allocated)
		cudaMalloc (&grid_2x2x2_uint_, 8 * grid_size * sizeof (unsigned int));

	if (!is_allocated) {
		cudaMalloc (&mcm_contour_indices_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_contour_values_, MAX_NUM_MCM_CELLS * sizeof (unsigned char));
		cudaMalloc (&mcm_contour_non_empty_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_values_, MAX_NUM_MCM_CELLS * sizeof (unsigned char));
		cudaMalloc (&mcm_compact_contour_indices_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_neigh_indices_, 8 * MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_neigh_morpho_centroids_, 8 * MAX_NUM_MCM_CELLS * sizeof (unsigned int));

		cudaMalloc (&mcm_vertices, MAX_V_SIZE * sizeof (float3));
		cudaMalloc (&mcm_normals, MAX_V_SIZE * sizeof (float3));
		cudaMalloc (&mcm_bilateral_normals, MAX_V_SIZE * sizeof (float3));
		cudaMalloc (&mcm_triangles, MAX_T_SIZE * sizeof (uint3));
	}
	ShowGPUMemoryUsage ();
}

void GMorpho::ExtractClosingMesh (Mesh & mesh, int num_bilateral_iters) {
	double time1, time2, timeMCM1, timeMCM2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	unsigned int * morpho_centroids_grid = NULL;
	//	unsigned int * morpho_centroids_grid = grid_2x2x2_uint_;
	//	cudaMemset (morpho_centroids_grid, 0xff, 8*grid_size*sizeof (unsigned int));

	// Allocate a 2x2x2 full grid of uints
	unsigned int grid_size = res_[0] * res_[1] * res_[2];

	timeMCM1 = GET_TIME ();
	// Compute a crossing contour on the base grid
	unsigned int * closing_contour_base = NULL;
	unsigned int closing_contour_base_size = 0;
	closing_grid_.ComputeContourMCM (closing_contour_base,
	                                 closing_contour_base_size);

	// Compute Marching Cube Data
	unsigned int * mcm_contour_grid_indices = grid_2x2x2_uint_;
	unsigned int num_mcm_cells = 8 * closing_contour_base_size;

	block_size = 256;
	time1 = GET_TIME ();
	ComputeMCMData <<< (closing_contour_base_size / block_size) + 1,
	               block_size >>>
	               (closing_contour_base, closing_contour_base_size,
	                closing_grid_.grid_gpu (),
	                mcm_contour_indices_,
	                mcm_contour_values_,
	                mcm_contour_non_empty_);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Marching Cube Data] : "
	          << "marching cube data computed in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	unsigned int last_value, last_value_scan;
	cudaMemcpy (&last_value,
	            mcm_contour_non_empty_ + num_mcm_cells - 1,
	            sizeof (unsigned int), cudaMemcpyDeviceToHost);

	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (mcm_contour_non_empty_),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty_
	                                + num_mcm_cells),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty_));
	cudaDeviceSynchronize ();
	cudaMemcpy (&last_value_scan,
	            mcm_contour_non_empty_ + num_mcm_cells - 1,
	            sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int num_mcm_contour_non_empty = last_value + last_value_scan;

	CompactMCMData <<< (num_mcm_cells / block_size) + 1, block_size >>>
	(mcm_contour_values_, mcm_compact_contour_values_,
	 mcm_contour_indices_, mcm_compact_contour_indices_,
	 mcm_compact_contour_neigh_morpho_centroids_,
	 morpho_centroids_grid,
	 mcm_contour_non_empty_, num_mcm_cells,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << num_mcm_contour_non_empty << " non empty cells compacted in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	cudaMemset (mcm_contour_grid_indices, 0xff, 8 * grid_size * sizeof (unsigned int));
	SplatIndices <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices_,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty);
	cudaDeviceSynchronize ();

	BuildNeighbors <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices_,
	 mcm_compact_contour_neigh_indices_,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << " neighbors computed in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&closing_contour_base);
	ShowGPUMemoryUsage ();

	Vec3f bboxMCM = bbox_.min();

	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), 0.5f * cell_size_,
	                                    0.5f * cell_size_, 0.5f * cell_size_,
	                                    2 * res_[0], 2 * res_[1], 2 * res_[2]);
	MarchingCubesMesher mesher (&grid_mcm);

	mesher.setDeviceMesh (mcm_vertices, mcm_normals, MAX_V_SIZE,
	                      mcm_triangles, MAX_T_SIZE);

	time1 = GET_TIME ();
	mesher.createMesh3D (mcm_compact_contour_values_,
	                     mcm_compact_contour_indices_,
	                     mcm_compact_contour_neigh_indices_,
	                     mcm_compact_contour_neigh_morpho_centroids_,
	                     num_mcm_contour_non_empty, 0.95f, 0.5 * NAN_EVAL);
	//	mesher.createMesh3D (closing_grid_.grid_gpu ().voxels, 128, 1e20);
	time2 = GET_TIME ();
	timeMCM2 = GET_TIME ();
	std::cout << "[Marching Cube] : "
	          << "mesh computed in "
	          << time2 - time1 << " ms." << std::endl;
	std::cout << "[Marching Cube] : "
	          << "mesh computed in "
	          << timeMCM2 - timeMCM1 << " ms." << std::endl;

	float3 * V;
	float3 * N;
	uint3 * T;
	unsigned int VSize;
	unsigned int TSize;

	//	time1 = GET_TIME ();
	mesher.getDeviceMesh (&V, &N, &VSize, &T, &TSize);

	float3 * bilateral_V = NULL;
	float3 * bilateral_N = NULL;
	float * bilateral_weights = NULL;

	bilateral_N = mcm_bilateral_normals;

	block_size = 512;
	time1 = GET_TIME ();
	cudaMemset (N, 0, VSize * sizeof (float3));
	ComputeMeshNormals <<< (TSize / block_size) + 1, block_size >>>
	(V, N, T, VSize, TSize);
	cudaDeviceSynchronize ();
	NormalizeMeshNormals <<< (VSize / block_size) + 1, block_size >>>
	(V, N, T, VSize, TSize);
	cudaDeviceSynchronize ();

	float sigma_p = 1.f;
	float sigma_n = 0.3f;
	for (int iter = 0; iter < (bilateral_filtering_ ? num_bilateral_iters : 0); iter++) {

		if (iter < 2) {
			sigma_p = 1.f;
			sigma_n = 1e10;
			//			sigma_n = 0.3f;
		} else {
			sigma_p = 1.f;
			sigma_n = 0.2f;
		}
		cudaMemset (bilateral_N, 0, VSize * sizeof (float3));
		BilateralMeshFiltering <<< (TSize / block_size) + 1, block_size >>>
		(V, N, T, bilateral_V, bilateral_N, bilateral_weights, VSize, TSize,
		 sigma_p, sigma_n);
		cudaDeviceSynchronize ();
		BilateralMeshFilteringNormalize <<< (VSize / block_size) + 1, block_size >>>
		(V, N, T, bilateral_V, bilateral_N, bilateral_weights, VSize, TSize);
		cudaDeviceSynchronize ();
	}
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Bilateral Mesh Filtering] : "
	          << "mesh filtered in  "
	          << time2 - time1 << " ms." << std::endl;

	std::vector<Vec3f> & p = mesh.P ();
	std::vector<Vec3f> & n = mesh.N ();
	std::vector<Vec3<unsigned int> > & t = mesh.T ();
	p.resize (VSize);
	n.resize (VSize);
	t.resize (TSize);

	time1 = GET_TIME ();
	cudaMemcpy (&p[0], V, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (&n[0], N, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (&t[0], T, TSize * sizeof (uint3), cudaMemcpyDeviceToHost);
	time2 = GET_TIME ();
	ShowGPUMemoryUsage ();
	std::cout << "[Marching Cube] : "
	          << "mesh transfert "
	          << time2 - time1 << " ms." << std::endl;
}

void GMorpho::ErodeByMipmap (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	// Build a Mipmap on the dilation contour
	//	dilation_contour_grid_.Init (bbox_, res_, data_res_, cell_size_);
	dilation_contour_grid_.CopyVoxelsFrom (dilation_grid_);
	dilation_contour_grid_.TransformToContour ();
	dilation_contour_grid_.BuildTexMipmaps (se_size);

	// Run an erosion at the base resolution
	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);
	time1 = GET_TIME ();
	//	ErodeBySphereMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	ErodeBySphereTexMipmapBaseILP2 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));


	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "base erosion in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * closing_contour_base_tight = NULL;
	unsigned int closing_contour_base_tight_size = 0;
	//	closing_grid_.ComputeContourBaseTight (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);
	closing_grid_.ComputeContourBase (closing_contour_base_tight,
	                                  closing_contour_base_tight_size);
	//	closing_grid_.ComputeContourBaseConservative (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);

	// Allocate a 2x2x2 full grid of uints
	unsigned int grid_size = res_[0] * res_[1] * res_[2];
	bool is_allocated = false;

	if (grid_2x2x2_uint_ == NULL)
		is_allocated = false;
	else
		is_allocated = true;

	if (!is_allocated)
		cudaMalloc (&grid_2x2x2_uint_, 8 * grid_size * sizeof (unsigned int));

	//	unsigned int * closing_contour_base_host = NULL;
	//	closing_contour_base_host = new unsigned int[closing_contour_base_size];
	//	cudaMemcpy (closing_contour_base_host, closing_contour_base,
	//							closing_contour_base_size*sizeof (unsigned int),
	//							cudaMemcpyDeviceToHost);
	//	SaveCubeList ("closing_contour_base.ply",
	//								closing_contour_base_host, closing_contour_base_size,
	//								bbox_, res_, cell_size_);
	//	free (closing_contour_base_host);
	//	closing_grid_.SetGridValue (0);

	// Run an erosion at full resolution only on the
	// contour cells computed at base resolution
	unsigned int * morpho_centroids_grid = grid_2x2x2_uint_;
	cudaMemset (morpho_centroids_grid, 0xff, 8 * grid_size * sizeof (unsigned int));

	block_size = 256;
	int num_blocks = 100;
	int host_global_warp_counter = num_blocks * EROSION_ILP_BLOCK_SIZE;

	if (global_warp_counter_ == NULL)
		cudaMalloc (&global_warp_counter_, sizeof (int));
	cudaMemcpy (global_warp_counter_, &host_global_warp_counter,
	            sizeof (int), cudaMemcpyHostToDevice);
	CheckCUDAError ();

	time1 = GET_TIME ();
	//	ErodeBySphereMipmapFull<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_dim.x*block_dim.y*block_dim.z/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexOccupMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapPersistentWarp<<<num_blocks, EROSION_BLOCK_SIZE>>>
	//		(closing_contour_base_tight,
	//		 //			 32*(closing_contour_base_tight_size/32),
	//		 closing_contour_base_tight_size,
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 global_warp_counter,
	//		 floor (2.f*se_size/cell_size_));

	ErodeBySphereTexMipmapPersistentWarpILP <<< num_blocks, EROSION_ILP_BLOCK_SIZE>>>
	(closing_contour_base_tight,
	 //			 32*(closing_contour_base_tight_size/32),
	 closing_contour_base_tight_size,
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 global_warp_counter_,
	 floor (2.f * se_size / cell_size_));

	//	ErodeBySphereTexMipmapPersistentWarp<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 NULL,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexDualMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapCollab<<<(COLLAB_SIZE*8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		sizeof (unsigned char)  // voxel
	//			+ 8*sizeof (NodeTex) + // 8 current node for each warp
	//			sizeof (unsigned char) // a mask coding the next nodes to visit
	//			>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "erosion in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a crossing contour on the base grid
	unsigned int * closing_contour_base = NULL;
	unsigned int closing_contour_base_size = 0;
	closing_grid_.ComputeContourMCM (closing_contour_base,
	                                 closing_contour_base_size);

	// Compute Marching Cube Data
	unsigned int * mcm_contour_grid_indices = grid_2x2x2_uint_;
	unsigned int num_mcm_cells = 8 * closing_contour_base_size;

	if (!is_allocated) {
		cudaMalloc (&mcm_contour_indices_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_contour_values_, MAX_NUM_MCM_CELLS * sizeof (unsigned char));
		cudaMalloc (&mcm_contour_non_empty_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_values_, MAX_NUM_MCM_CELLS * sizeof (unsigned char));
		cudaMalloc (&mcm_compact_contour_indices_, MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_neigh_indices_, 8 * MAX_NUM_MCM_CELLS * sizeof (unsigned int));
		cudaMalloc (&mcm_compact_contour_neigh_morpho_centroids_, 8 * MAX_NUM_MCM_CELLS * sizeof (unsigned int));
	}

	block_size = 256;
	time1 = GET_TIME ();
	ComputeMCMData <<< (closing_contour_base_size / block_size) + 1,
	               block_size >>>
	               (closing_contour_base, closing_contour_base_size,
	                closing_grid_.grid_gpu (),
	                mcm_contour_indices_,
	                mcm_contour_values_,
	                mcm_contour_non_empty_);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Marching Cube Data] : "
	          << "marching cube data computed in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	unsigned int last_value, last_value_scan;
	cudaMemcpy (&last_value,
	            mcm_contour_non_empty_ + num_mcm_cells - 1,
	            sizeof (unsigned int), cudaMemcpyDeviceToHost);

	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (mcm_contour_non_empty_),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty_
	                                + num_mcm_cells),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty_));
	cudaDeviceSynchronize ();
	cudaMemcpy (&last_value_scan,
	            mcm_contour_non_empty_ + num_mcm_cells - 1,
	            sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int num_mcm_contour_non_empty = last_value + last_value_scan;
	CompactMCMData <<< (num_mcm_cells / block_size) + 1, block_size >>>
	(mcm_contour_values_, mcm_compact_contour_values_,
	 mcm_contour_indices_, mcm_compact_contour_indices_,
	 mcm_compact_contour_neigh_morpho_centroids_,
	 morpho_centroids_grid,
	 mcm_contour_non_empty_, num_mcm_cells,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << num_mcm_contour_non_empty << " non empty cells compacted in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	cudaMemset (mcm_contour_grid_indices, 0xff, 8 * grid_size * sizeof (unsigned int));
	SplatIndices <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices_,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty);
	cudaDeviceSynchronize ();

	BuildNeighbors <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices_,
	 mcm_compact_contour_neigh_indices_,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << " neighbors computed in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&closing_contour_base);
	ShowGPUMemoryUsage ();

	unsigned int * mcm_compact_contour_indices_host = NULL;
	mcm_compact_contour_indices_host = new unsigned int [num_mcm_contour_non_empty];
	cudaMemcpy (mcm_compact_contour_indices_host, mcm_compact_contour_indices_,
	            num_mcm_contour_non_empty * sizeof (unsigned int),
	            cudaMemcpyDeviceToHost);
	//	SaveCubeList ("closing_mcm_cubes.ply",
	//								mcm_compact_contour_indices_host,
	//								num_mcm_contour_non_empty,
	//								bbox_, ((unsigned int) 2) * res_, 0.5f*cell_size_);
	Vec3f bboxMCM = bbox_.min();

	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), 0.5f * cell_size_,
	                                    0.5f * cell_size_, 0.5f * cell_size_,
	                                    2 * res_[0], 2 * res_[1], 2 * res_[2]);
	//	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), cell_size_,
	//																			cell_size_, cell_size_,
	//																			res_[0], res_[1], res_[2]);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (mcm_compact_contour_values_,
	                     mcm_compact_contour_indices_,
	                     mcm_compact_contour_neigh_indices_,
	                     mcm_compact_contour_neigh_morpho_centroids_,
	                     num_mcm_contour_non_empty, 0.5f, 0.5 * NAN_EVAL);
	//	mesher.createMesh3D (closing_grid_.grid_gpu ().voxels, 128, 1e20);
	time2 = GET_TIME ();
	std::cout << "[Marching Cube] "
	          << "mesh computed in "
	          << time2 - time1 << " ms." << std::endl;

	float * V_host;
	float * N_host;
	unsigned int * T_host;
	float3 * V;
	float3 * N;
	uint3 * T;
	unsigned int VSize;
	unsigned int TSize;
	Mesh mesh;

	time1 = GET_TIME ();
	mesher.getDeviceMesh (&V, &N, &VSize, &T, &TSize);
	v_morpho_mesh_ = V;
	n_morpho_mesh_ = N;
	v_morpho_mesh_size_ = VSize;
	t_morpho_mesh_size_ = TSize;

	float3 * bilateral_V = NULL;
	float3 * bilateral_N = NULL;
	float * bilateral_weights = NULL;
	cudaMalloc (&bilateral_V, VSize * sizeof (float3));
	cudaMalloc (&bilateral_N, VSize * sizeof (float3));
	cudaMalloc (&bilateral_weights, VSize * sizeof (float));
	V_host = new float[3 * VSize];
	N_host = new float[3 * VSize];
	T_host = new unsigned int[3 * TSize];

	cudaMemcpy (V_host, V, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (N_host, N, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (T_host, T, TSize * sizeof (uint3), cudaMemcpyDeviceToHost);

	mesh.clear ();
	mesh.P ().resize (VSize);
	memcpy (&mesh.P ()[0], V_host, VSize * sizeof (Vec3f));
	mesh.N ().resize (VSize);
	memcpy (&mesh.N ()[0], N_host, VSize * sizeof (Vec3f));
	mesh.T ().resize (TSize);
	memcpy (&mesh.T ()[0], T_host, TSize * 3 * sizeof (unsigned int));
	mesh.recomputeNormals ();
	time2 = GET_TIME ();
	std::cout << "[Marching Cube] "
	          << "mesh transfert "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&V);
	FreeGPUResource (&N);
	FreeGPUResource (&T);
	FreeGPUResource (&bilateral_V);
	FreeGPUResource (&bilateral_N);
	FreeGPUResource (&bilateral_weights);

	ShowGPUMemoryUsage ();
	morpho_mesh_ = mesh;

}

void GMorpho::ErodeByMipmapFull (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	// Build a Mipmap on the dilation contour
	dilation_contour_grid_.Init (bbox_, res_, data_res_, cell_size_);
	dilation_contour_grid_.CopyVoxelsFrom (dilation_grid_);
	dilation_contour_grid_.TransformToContour ();
	dilation_contour_grid_.BuildTexMipmaps (se_size);

	// Run an erosion at the base resolution
	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);
	time1 = GET_TIME ();
	//	ErodeBySphereMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapBase<<<grid_dim, block_dim>>>
	//		(input_grid_.grid_gpu (),
	//		 dilation_grid_.grid_gpu (),
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 floor (2.f*se_size/cell_size_));

	ErodeBySphereTexMipmapBaseILP2 <<< grid_dim, block_dim>>>
	(input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));


	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "base erosion in "
	          << time2 - time1 << " ms." << std::endl;


	// Compute a tight crossing contour on the base grid
	unsigned int * closing_contour_base_tight = NULL;
	unsigned int closing_contour_base_tight_size = 0;
	//	closing_grid_.ComputeContourBaseTight (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);
	closing_grid_.ComputeContourBase (closing_contour_base_tight,
	                                  closing_contour_base_tight_size);
	//	closing_grid_.ComputeContourBaseConservative (closing_contour_base_tight,
	//																		closing_contour_base_tight_size);

	// Allocate a 2x2x2 full grid of uints
	unsigned int grid_size = res_[0] * res_[1] * res_[2];
	unsigned int * grid_2x2x2_uint = NULL;
	cudaMalloc (&grid_2x2x2_uint, 8 * grid_size * sizeof (unsigned int));

	//	unsigned int * closing_contour_base_host = NULL;
	//	closing_contour_base_host = new unsigned int[closing_contour_base_size];
	//	cudaMemcpy (closing_contour_base_host, closing_contour_base,
	//							closing_contour_base_size*sizeof (unsigned int),
	//							cudaMemcpyDeviceToHost);
	//	SaveCubeList ("closing_contour_base.ply",
	//								closing_contour_base_host, closing_contour_base_size,
	//								bbox_, res_, cell_size_);
	//	free (closing_contour_base_host);
	//	closing_grid_.SetGridValue (0);

	// Run an erosion at full resolution only on the
	// contour cells computed at base resolution
	unsigned int * morpho_centroids_grid = grid_2x2x2_uint;
	cudaMemset (morpho_centroids_grid, 0xff, 8 * grid_size * sizeof (unsigned int));

	block_size = 256;
	int num_blocks = 100;
	int * global_warp_counter = NULL;
	int host_global_warp_counter = num_blocks * EROSION_BLOCK_SIZE;
	cudaMalloc (&global_warp_counter, sizeof (int));
	cudaMemcpy (global_warp_counter, &host_global_warp_counter,
	            sizeof (int), cudaMemcpyHostToDevice);
	CheckCUDAError ();

	time1 = GET_TIME ();
	//	ErodeBySphereMipmapFull<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_dim.x*block_dim.y*block_dim.z/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexOccupMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapPersistentWarp<<<num_blocks, EROSION_BLOCK_SIZE>>>
	//		(closing_contour_base_tight,
	//		 //			 32*(closing_contour_base_tight_size/32),
	//		 closing_contour_base_tight_size,
	//		 dilation_contour_grid_.grid_gpu (),
	//		 closing_grid_.grid_gpu (),
	//		 global_warp_counter,
	//		 floor (2.f*se_size/cell_size_));

	ErodeBySphereTexMipmapPersistentWarpILP <<< num_blocks, EROSION_ILP_BLOCK_SIZE>>>
	(closing_contour_base_tight,
	 //			 32*(closing_contour_base_tight_size/32),
	 closing_contour_base_tight_size,
	 dilation_contour_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 global_warp_counter,
	 floor (2.f * se_size / cell_size_));

	//	ErodeBySphereTexMipmapPersistentWarp<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 NULL,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexDualMipmap<<<(8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		(block_size/8)*sizeof (unsigned char)>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	//	ErodeBySphereTexMipmapCollab<<<(COLLAB_SIZE*8*closing_contour_base_tight_size/block_size)+1,
	//		block_size,
	//		sizeof (unsigned char)  // voxel
	//			+ 8*sizeof (NodeTex) + // 8 current node for each warp
	//			sizeof (unsigned char) // a mask coding the next nodes to visit
	//			>>>
	//			(closing_contour_base_tight, closing_contour_base_tight_size,
	//			 dilation_contour_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 morpho_centroids_grid,
	//			 floor (2.f*se_size/cell_size_));

	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "erosion in "
	          << time2 - time1 << " ms." << std::endl;

	// Compute a crossing contour on the base grid
	unsigned int * closing_contour_base = NULL;
	unsigned int closing_contour_base_size = 0;
	closing_grid_.ComputeContourMCM (closing_contour_base,
	                                 closing_contour_base_size);

	// Compute Marching Cube Data
	unsigned int * mcm_contour_indices = NULL;
	unsigned char * mcm_contour_values = NULL;
	unsigned int * mcm_contour_non_empty = NULL;
	unsigned char * mcm_compact_contour_values = NULL;
	unsigned int * mcm_compact_contour_indices = NULL;
	unsigned int * mcm_contour_grid_indices = grid_2x2x2_uint;
	//	unsigned int * mcm_contour_neigh_indices = NULL;
	unsigned int * mcm_compact_contour_neigh_morpho_centroids = NULL;
	unsigned int * mcm_compact_contour_neigh_indices = NULL;
	unsigned int num_mcm_cells = 8 * closing_contour_base_size;
	cudaMalloc (&mcm_contour_indices, num_mcm_cells * sizeof (unsigned int));
	cudaMalloc (&mcm_contour_values, num_mcm_cells * sizeof (unsigned char));
	cudaMalloc (&mcm_contour_non_empty, num_mcm_cells * sizeof (unsigned int));
	cudaMalloc (&mcm_compact_contour_values, num_mcm_cells * sizeof (unsigned char));
	cudaMalloc (&mcm_compact_contour_indices, num_mcm_cells * sizeof (unsigned int));
	cudaMalloc (&mcm_compact_contour_neigh_indices, 8 * num_mcm_cells * sizeof (unsigned int));
	cudaMalloc (&mcm_compact_contour_neigh_morpho_centroids, 8 * num_mcm_cells * sizeof (unsigned int));

	block_size = 256;
	time1 = GET_TIME ();
	ComputeMCMData <<< (closing_contour_base_size / block_size) + 1,
	               block_size >>>
	               (closing_contour_base, closing_contour_base_size,
	                closing_grid_.grid_gpu (),
	                mcm_contour_indices,
	                mcm_contour_values,
	                mcm_contour_non_empty);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Marching Cube Data] : "
	          << "marching cube data computed in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	unsigned int last_value, last_value_scan;
	cudaMemcpy (&last_value,
	            mcm_contour_non_empty + num_mcm_cells - 1,
	            sizeof (unsigned int), cudaMemcpyDeviceToHost);

	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (mcm_contour_non_empty),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty
	                                + num_mcm_cells),
	                        thrust::device_ptr<unsigned int> (mcm_contour_non_empty));
	cudaDeviceSynchronize ();
	cudaMemcpy (&last_value_scan,
	            mcm_contour_non_empty + num_mcm_cells - 1,
	            sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int num_mcm_contour_non_empty = last_value + last_value_scan;
	CompactMCMData <<< (num_mcm_cells / block_size) + 1, block_size >>>
	(mcm_contour_values, mcm_compact_contour_values,
	 mcm_contour_indices, mcm_compact_contour_indices,
	 mcm_compact_contour_neigh_morpho_centroids,
	 morpho_centroids_grid,
	 mcm_contour_non_empty, num_mcm_cells,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << num_mcm_contour_non_empty << " non empty cells compacted in "
	          << time2 - time1 << " ms." << std::endl;

	time1 = GET_TIME ();
	cudaMemset (mcm_contour_grid_indices, 0xff, 8 * grid_size * sizeof (unsigned int));
	SplatIndices <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty);
	cudaDeviceSynchronize ();

	BuildNeighbors <<< (num_mcm_contour_non_empty / block_size) + 1, block_size >>>
	(mcm_compact_contour_indices,
	 mcm_compact_contour_neigh_indices,
	 mcm_contour_grid_indices,
	 num_mcm_contour_non_empty,
	 closing_grid_.grid_gpu ().res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Marching Cube Data] : "
	          << " neighbors computed in "
	          << time2 - time1 << " ms." << std::endl;

	unsigned int * mcm_compact_contour_indices_host = NULL;
	mcm_compact_contour_indices_host = new unsigned int [num_mcm_contour_non_empty];
	cudaMemcpy (mcm_compact_contour_indices_host, mcm_compact_contour_indices,
	            num_mcm_contour_non_empty * sizeof (unsigned int),
	            cudaMemcpyDeviceToHost);
	//	SaveCubeList ("closing_mcm_cubes.ply",
	//								mcm_compact_contour_indices_host,
	//								num_mcm_contour_non_empty,
	//								bbox_, ((unsigned int) 2) * res_, 0.5f*cell_size_);
	Vec3f bboxMCM = bbox_.min();

	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), 0.5f * cell_size_,
	                                    0.5f * cell_size_, 0.5f * cell_size_,
	                                    2 * res_[0], 2 * res_[1], 2 * res_[2]);
	//	MarchingCubesMesher::Grid grid_mcm (bbox_.min(), cell_size_,
	//																			cell_size_, cell_size_,
	//																			res_[0], res_[1], res_[2]);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (mcm_compact_contour_values,
	                     mcm_compact_contour_indices,
	                     mcm_compact_contour_neigh_indices,
	                     mcm_compact_contour_neigh_morpho_centroids,
	                     num_mcm_contour_non_empty, 0.5f, 0.5 * NAN_EVAL);
	//	mesher.createMesh3D (closing_grid_.grid_gpu ().voxels, 128, 1e20);
	time2 = GET_TIME ();
	std::cout << "[Marching Cube] "
	          << "mesh computed in "
	          << time2 - time1 << " ms." << std::endl;

	float * V_host;
	float * N_host;
	unsigned int * T_host;
	float3 * V;
	float3 * N;
	uint3 * T;
	unsigned int VSize;
	unsigned int TSize;

	mesher.getDeviceMesh (&V, &N, &VSize, &T, &TSize);
	v_morpho_mesh_ = V;
	n_morpho_mesh_ = N;
	v_morpho_mesh_size_ = VSize;
	t_morpho_mesh_size_ = TSize;


	float3 * bilateral_V = NULL;
	float3 * bilateral_N = NULL;
	float * bilateral_weights = NULL;
	cudaMalloc (&bilateral_V, VSize * sizeof (float3));
	cudaMalloc (&bilateral_N, VSize * sizeof (float3));
	cudaMalloc (&bilateral_weights, VSize * sizeof (float));
	V_host = new float[3 * VSize];
	N_host = new float[3 * VSize];
	T_host = new unsigned int[3 * TSize];

	cudaMemcpy (V_host, V, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (N_host, N, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (T_host, T, TSize * sizeof (uint3), cudaMemcpyDeviceToHost);

	Mesh mesh;
	mesh.clear ();
	mesh.P ().resize (VSize);
	memcpy (&mesh.P ()[0], V_host, VSize * sizeof (Vec3f));
	mesh.N ().resize (VSize);
	memcpy (&mesh.N ()[0], N_host, VSize * sizeof (Vec3f));
	mesh.T ().resize (TSize);
	memcpy (&mesh.T ()[0], T_host, TSize * 3 * sizeof (unsigned int));
	mesh.recomputeNormals ();

	morpho_mesh_ = mesh;

	for (unsigned int i = 0; i < VSize; i++) {
		N_host[3 * i] = mesh.N ()[i][0];
		N_host[3 * i + 1] = mesh.N ()[i][1];
		N_host[3 * i + 2] = mesh.N ()[i][2];
	}
	cudaMemcpy (N, N_host, VSize * sizeof (float3), cudaMemcpyHostToDevice);

	block_size = 512;
	float sigma_p = 1.f;
	float sigma_n = 0.3f;
	time1 = GET_TIME ();
	for (int iter = 0; iter < 100; iter++) {
		cudaMemset (bilateral_N, 0, VSize * sizeof (float3));
		BilateralMeshFiltering <<< (TSize / block_size) + 1, block_size >>>
		(V, N, T, bilateral_V, bilateral_N, bilateral_weights, VSize, TSize,
		 sigma_p, sigma_n);
		cudaDeviceSynchronize ();
		BilateralMeshFilteringNormalize <<< (VSize / block_size) + 1, block_size >>>
		(V, N, T, bilateral_V, bilateral_N, bilateral_weights, VSize, TSize);
		cudaDeviceSynchronize ();
	}
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Bilateral Mesh Filtering] : "
	          << "mesh filtered in  "
	          << time2 - time1 << " ms." << std::endl;

	cudaMemcpy (V_host, V, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (N_host, N, VSize * sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (T_host, T, TSize * sizeof (uint3), cudaMemcpyDeviceToHost);

	mesh.clear ();
	mesh.P ().resize (VSize);
	memcpy (&mesh.P ()[0], V_host, VSize * sizeof (Vec3f));
	mesh.N ().resize (VSize);
	memcpy (&mesh.N ()[0], N_host, VSize * sizeof (Vec3f));
	mesh.T ().resize (TSize);
	memcpy (&mesh.T ()[0], T_host, TSize * 3 * sizeof (unsigned int));

	FreeGPUResource (&closing_contour_base);
	ShowGPUMemoryUsage ();
}

void GMorpho::Load (const float * P, const float * N, int num_of_vertices,
                    const unsigned int * T, int num_of_faces, int base_res,
                    float se_size) {
	se_size_ = se_size;
	//	float se_size = 0.0875;
	// First Voxelize the input mesh
	float margin = se_size_ + 2.f / (2 * ((float)base_res));
	//	float margin = se_size;
	//	float margin = 0.01f + 1.f;
	voxelizer_.Load (P, N, num_of_vertices, T, num_of_faces);
//	voxelizer_.VoxelizeByMultiSlicing (base_res, margin);
	voxelizer_.VoxelizeConservative (base_res, margin);

	// The voxelization gives the input grid along with its
	// bounding box, resolution, and cell size.
	// All subsequent morphological operations will use
	// the first voxelization space discretisation
	input_grid_ = voxelizer_.grid ();
	bbox_ = input_grid_.bbox ();
	res_ = input_grid_.res ();
	data_res_ = input_grid_.data_res ();
	cell_size_ = input_grid_.cell_size ();

	//	// Debug
	//	input_grid_.TestByMeshing ("input_2x2x2.off");

	// Allocate and Initialize Dilation Grid
	dilation_grid_.Init (bbox_, res_, data_res_, cell_size_);
	dilation_grid_.CopyVoxelsFrom (input_grid_);

	// Allocate and Initinalize Closing Grid
	closing_grid_.Init (bbox_, res_, data_res_, cell_size_);
	closing_grid_.CopyVoxelsFrom (dilation_grid_);

	// Allocate and Initinalize Dilation Contour Grid
	dilation_contour_grid_.Init (bbox_, res_, data_res_, cell_size_);

	// Allocate Erosion Data
	ErodeAllocation ();

	// Allocate Mesh Extraction Data
	ExtractClosingMeshAllocation ();
}

void GMorpho::Update (Mesh & mesh, const ScaleField & scale_field,
                      const FrameField & frame_field) {
	double time1, time2;
	std::cout << std::endl;
	ShowGPUMemoryUsage ();
	// Initinalize Dilation Grid
	dilation_grid_.CopyVoxelsFrom (input_grid_);

	time1 = GET_TIME ();
	if (floor (2.f * scale_field.global_scale () / cell_size_) > 1) {
		if (use_asymmetric_closing_)
			if (use_frame_field_)
				DilateByRotCubeMipmap (scale_field, frame_field);
			else
				DilateByCubeMipmap (scale_field);
		else
			DilateBySphereMipmap (scale_field);
	}

	//	DilateBySphereMipmap (scale_field.global_scale ());
	time2 = GET_TIME ();
	std::cout << "[Morpho Update] : Dilation of size " << scale_field.global_scale ()
	          << " | " << floor (2.f * scale_field.global_scale () / cell_size_)
	          << " in " << time2 - time1 << " ms." << std::endl;

	// Initinalize Closing Grid
	closing_grid_.CopyVoxelsFrom (dilation_grid_);

	if (floor (2.f * scale_field.global_scale () / cell_size_) > 1)
		ErodeBySphereMipmap (scale_field);

	//	ErodeByBallSplatting (scale_field.global_scale ());

	time1 = GET_TIME ();
	if (use_asymmetric_closing_)
		ExtractClosingMesh (mesh, 200);
	else
		ExtractClosingMesh (mesh, 20);
	time2 = GET_TIME ();
	std::cout << "[Morpho Update] : mesh extraction in "
	          << time2 - time1 << " ms." << std::endl;
	ShowGPUMemoryUsage ();
}

void GMorpho::Update (Mesh & mesh, float se_size) {
	double time1, time2;
	se_size_ = se_size;
	std::cout << std::endl;
	ShowGPUMemoryUsage ();

	// Initinalize Dilation Grid
	dilation_grid_.CopyVoxelsFrom (input_grid_);

	time1 = GET_TIME ();
	DilateByCubeMipmap (se_size_);
	time2 = GET_TIME ();
	std::cout << "[Morpho Update] : Dilation in "
	          << time2 - time1 << " ms." << std::endl;

	// Initinalize Closing Grid
	closing_grid_.CopyVoxelsFrom (dilation_grid_);

	ErodeBySphereMipmap (se_size_);

	time1 = GET_TIME ();
	ExtractClosingMesh (mesh);
	time2 = GET_TIME ();
	std::cout << "[Morpho Update] : mesh extraction in "
	          << time2 - time1 << " ms." << std::endl;
	ShowGPUMemoryUsage ();
}

void FlattenCoordinates (const std::vector<Vec3c> & coordinates,
                         const Vec3ui & res,
                         unsigned int *& flat,
                         unsigned int & flat_size) {

	flat_size = coordinates.size ();
	flat = new unsigned int[flat_size];
	for (int i = 0; i < flat_size; i++) {
		Vec3c coord = coordinates[i];
		flat[i] = res[1] * res[0] * coord[2] + res[0] * coord[1] + coord[0];
	}
}

void ComputeCSphericalContour (std::vector<Vec3c> & contour,
                               float radius) {
	int i_radius = floor (radius);
	Vec3ui res (2 * i_radius + 6, 2 * i_radius + 6, 2 * i_radius + 6);
	Vec3f bbox_min (-i_radius - 3, -i_radius - 3, -i_radius - 3);
	// Center at 0.5f 0.5f 0.5f to match the previous step of dilation
	Vec3f center (0.5f, 0.5f, 0.5f);
	Vec3c c_center (floor (center[0] - bbox_min[0]),
	                floor (center[1] - bbox_min[1]),
	                floor (center[2] - bbox_min[2]));
	contour.clear ();
	std::vector<Vec3c> contour_test;
	contour_test.clear ();

	std::cout << "[Compute Complementary Spherical Contour] : " << "radius "
	          << radius << " i_radius " << i_radius << " center " << center << " c_center " <<
	          (int)c_center[0] << ", " << (int)c_center[1]
	          << ", " << (int)c_center[2] << std::endl;

	float min_dist_to_center = FLT_MAX;
	// Fill up an inside outside segmentation of the sphere
	unsigned char * voxels = new unsigned char[res[0]*res[1]*res[2]];
	for (int k = 0; k < res[2]; k++)
		for (int j = 0; j < res[1]; j++)
			for (int i = 0; i < res[0]; i++) {
				unsigned int key = res[1] * res[0] * k + res[0] * j + i;
				Vec3f coords (i, j, k);
				coords = bbox_min + coords + Vec3f (0.5f, 0.5f, 0.5f);
				float dist_to_center = dist (center, coords);

				if (dist_to_center <= min_dist_to_center) {
					min_dist_to_center = dist_to_center;
					c_center = Vec3c (i, j, k);
				}

				if (dist_to_center < radius) {
					voxels[key] = 0xff;
				} else {
					voxels[key] = 0x00;
				}
			}
	std::cout << "[Compute Complementary Spherical Contour] : "
	          << "optim c_center : "
	          << (int)c_center[0] << ", " << (int)c_center[1]
	          << ", " << (int)c_center[2]
	          << " with min dist : " << min_dist_to_center << std::endl;

	// Contour the segmentation in Z Y X order and 6-connectivity
	for (int k = 1; k < (res[2] - 1); k++)
		for (int j = 1; j < (res[1] - 1); j++)
			for (int i = 1; i < (res[0] - 1); i++) {
				bool is_contour = false;
				unsigned int key = res[1] * res[0] * k + res[0] * j + i;
				unsigned char center_val = voxels[key];
				if (center_val == 0x00) {
					//					for (int n = -1; n <=1; n++)
					//						for (int m = -1; m <=1; m++)
					//							for (int l = -1; l <=1; l++) {
					//								if (n != 0 && m != 0 && l != 0) {
					//									int id_x, id_y, id_z;
					//									id_x = i + l; id_y = j + m; id_z = k + n;
					//									key = res[1]*res[0]*id_z + res[0]*id_y + id_x;
					//									is_contour = is_contour || (voxels[key] != center_val);
					//								}
					//							}
					for (int n = -1; n <= 1; n++)
						if (n != 0) {
							int id_x, id_y, id_z;
							id_x = i; id_y = j; id_z = k + n;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
					for (int m = -1; m <= 1; m++)
						if (m != 0) {
							int id_x, id_y, id_z;
							id_x = i; id_y = j + m; id_z = k;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
					for (int l = -1; l <= 1; l++)
						if (l != 0) {
							int id_x, id_y, id_z;
							id_x = i + l; id_y = j; id_z = k;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
				}

				if (is_contour) {
					Vec3c cell (i, j, k);
					cell = cell - c_center;
					contour.push_back (cell);
					contour_test.push_back (Vec3c (i, j, k));
				}
			}

	//	unsigned int * se_contour_host = NULL;
	//	unsigned int se_contour_size_host = 0;
	//	BoundingBox bbox;
	//	Vec3ui res_test (256, 256, 256);
	//	bbox.min_ = Vec3f (0.f, 0.f, 0.f);
	//	FlattenCoordinates (contour_test, res_test,
	//											se_contour_host, se_contour_size_host);
	//	GMorpho::SaveCubeList ("se_contour.ply",
	//												 se_contour_host, se_contour_size_host,
	//												 bbox, res_test, 1.f);
	//	free (se_contour_host);

	std::cout << "[Compute Complementary Spherical Contour] : " << contour.size ()
	          << " in computed contour" << std::endl;
}

void ComputeSphericalContour (std::vector<Vec3c> & contour,
                              float radius) {
	int i_radius = floor (radius);
	Vec3ui res (2 * i_radius + 6, 2 * i_radius + 6, 2 * i_radius + 6);
	Vec3f bbox_min (-i_radius - 3, -i_radius - 3, -i_radius - 3);
	// Center at 0.5f 0.5f 0.5f to match the previous step of dilation
	Vec3f center (0.5f, 0.5f, 0.5f);
	Vec3c c_center (floor (center[0] - bbox_min[0]),
	                floor (center[1] - bbox_min[1]),
	                floor (center[2] - bbox_min[2]));
	contour.clear ();
	std::vector<Vec3c> contour_test;
	contour_test.clear ();

	std::cout << "[Compute Spherical Contour] : " << "radius "
	          << radius << " i_radius " << i_radius << " center " << center << " c_center " <<
	          (int)c_center[0] << ", " << (int)c_center[1]
	          << ", " << (int)c_center[2] << std::endl;

	float min_dist_to_center = FLT_MAX;
	// Fill up an inside outside segmentation of the sphere
	unsigned char * voxels = new unsigned char[res[0]*res[1]*res[2]];
	for (int k = 0; k < res[2]; k++)
		for (int j = 0; j < res[1]; j++)
			for (int i = 0; i < res[0]; i++) {
				unsigned int key = res[1] * res[0] * k + res[0] * j + i;
				Vec3f coords (i, j, k);
				coords = bbox_min + coords + Vec3f (0.5f, 0.5f, 0.5f);
				float dist_to_center = dist (center, coords);

				if (dist_to_center <= min_dist_to_center) {
					min_dist_to_center = dist_to_center;
					c_center = Vec3c (i, j, k);
				}

				if (dist_to_center < radius) {
					voxels[key] = 0xff;
				} else {
					voxels[key] = 0x00;
				}
			}
	std::cout << "[Compute Spherical Contour] : " << "optim c_center : "
	          << (int)c_center[0] << ", " << (int)c_center[1]
	          << ", " << (int)c_center[2]
	          << " with min dist : " << min_dist_to_center << std::endl;

	// Contour the segmentation in Z Y X order and 6-connectivity
	for (int k = 1; k < (res[2] - 1); k++)
		for (int j = 1; j < (res[1] - 1); j++)
			for (int i = 1; i < (res[0] - 1); i++) {
				bool is_contour = false;
				unsigned int key = res[1] * res[0] * k + res[0] * j + i;
				unsigned char center_val = voxels[key];
				if (center_val != 0x00) {
					//					for (int n = -1; n <=1; n++)
					//						for (int m = -1; m <=1; m++)
					//							for (int l = -1; l <=1; l++) {
					//								if (n != 0 && m != 0 && l != 0) {
					//									int id_x, id_y, id_z;
					//									id_x = i + l; id_y = j + m; id_z = k + n;
					//									key = res[1]*res[0]*id_z + res[0]*id_y + id_x;
					//									is_contour = is_contour || (voxels[key] != center_val);
					//								}
					//							}
					for (int n = -1; n <= 1; n++)
						if (n != 0) {
							int id_x, id_y, id_z;
							id_x = i; id_y = j; id_z = k + n;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
					for (int m = -1; m <= 1; m++)
						if (m != 0) {
							int id_x, id_y, id_z;
							id_x = i; id_y = j + m; id_z = k;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
					for (int l = -1; l <= 1; l++)
						if (l != 0) {
							int id_x, id_y, id_z;
							id_x = i + l; id_y = j; id_z = k;
							key = res[1] * res[0] * id_z + res[0] * id_y + id_x;
							is_contour = is_contour || (voxels[key] != center_val);
						}
				}

				if (is_contour) {
					Vec3c cell (i, j, k);
					cell = cell - c_center;
					contour.push_back (cell);
					contour_test.push_back (Vec3c (i, j, k));
				}
			}

	//	unsigned int * se_contour_host = NULL;
	//	unsigned int se_contour_size_host = 0;
	//	BoundingBox bbox;
	//	Vec3ui res_test (256, 256, 256);
	//	bbox.min = Vec3f (0.f, 0.f, 0.f);
	//	FlattenCoordinates (contour_test, res_test,
	//											se_contour_host, se_contour_size_host);
	//	GMorpho::SaveCubeList ("se_contour.ply",
	//												 se_contour_host, se_contour_size_host,
	//												 bbox, res_test, 1.f);
	//	free (se_contour_host);

	std::cout << "[Compute Spherical Contour] : " << contour.size ()
	          << " in computed contour" << std::endl;
}

void BuildPackedSE (const std::vector<Vec3c> & SEFineContour,
                    std::vector<char> & SECoarseContour, int radius) {
	std::vector<unsigned char> values (8);
	std::vector<Vec3c> coarseCell (8);
	std::vector<Vec3c> fineCell (8);
	std::vector<Vec3c> offCell (8);

	offCell[0] = Vec3c (0, 0, 0); offCell[4] = Vec3c (0, 0, 1);
	offCell[1] = Vec3c (1, 0, 0); offCell[5] = Vec3c (1, 0, 1);
	offCell[2] = Vec3c (0, 1, 0); offCell[6] = Vec3c (0, 1, 1);
	offCell[3] = Vec3c (1, 1, 0); offCell[7] = Vec3c (1, 1, 1);

	for (int k = -radius; k <= radius; k++)
		for (int j = -radius; j <= radius; j++)
			for (int i = -radius; i <= radius; i++) {
				bool found = false;
				Vec3c coarse (2 * i, 2 * j, 2 * k);

				for (int m = 0; m < 8; m++)
					coarseCell[m] = coarse + offCell[m];

				for (unsigned int m = 0; m < 8; m++)
					values[m] = 0x00;

				for (unsigned int l = 0; l < SEFineContour.size (); l++) {
					Vec3c fine = SEFineContour[l];

					for (int n = 0; n < 8; n++)
						fineCell[n] = fine + offCell[n];

					//					if (i == -5 && j == 0 && k == 0) {/*{{{*/
					//						if (fine[0] == -10 && fine[1] == 0 && fine[2] == 0) {
					//							printf ("found fine\n");
					//						}
					//					}/*}}}*/
					for (unsigned int m = 0; m < 8; m++) {
						for (unsigned int n = 0; n < 8; n++) {
							if ((fineCell[n][0] == coarseCell[m][0]) &&
							        (fineCell[n][1] == coarseCell[m][1]) &&
							        (fineCell[n][2] == coarseCell[m][2])) {
								values[n] |= (0x01 << m);
							}
						}
					}
				}

				for (unsigned int n = 0; n < 8; n++)
					if (values[n] != 0x00)
						found = true;

				//				if (i == 5 && j == 0 && k == 0) {/*{{{*/
				//					printf ("%i %i %i : x%.2x x%.2x x%.2x x%.2x x%.2x x%.2x x%.2x x%.2x\n",
				//									i, j, k, values[0], values[1], values[2], values[3],
				//									values[4], values[5], values[6], values[7]);
				//				}/*}}}*/
				if (found) {
					SECoarseContour.push_back (i);
					SECoarseContour.push_back (j);
					SECoarseContour.push_back (k);
					SECoarseContour.push_back (0xff);
					for (int n = 0; n < 8; n++)
						SECoarseContour.push_back (values[n]);
				}
			}
	std::cout << "[Build Packed SE] : " << SECoarseContour.size () / 12
	          << " coarse cells" << std::endl;
}

void GMorpho::ComputeSphericalSEPackedContour (char *& se_packed_contour,
        unsigned int & se_packed_contour_size,
        bool load, bool save, float se_size) {
	std::vector<Vec3c> se_contour_vec_host;
	std::vector<char> se_packed_contour_host;
	int se_size_int = floor (2.f * se_size / cell_size_);
	std::string int_str;
	std::stringstream out;
	out << se_size_int;
	int_str = out.str();
	std::string se_file_name = "SphericalSE"
	                           + int_str + ".se";

	if (load) {
		std::basic_ifstream<char> file(se_file_name.c_str (),  std::ios::binary);
		se_packed_contour_host = std::vector<char>(std::istreambuf_iterator<char>(file),
		                         std::istreambuf_iterator<char>());
		std::cout << "[Spherical SE Packed Contour] : "
		          << se_file_name << " loaded" << std::endl;
	} else {
		ComputeSphericalContour (se_contour_vec_host,
		                         floor (2.f * se_size / cell_size_));
		BuildPackedSE (se_contour_vec_host, se_packed_contour_host,
		               floor (2.f * se_size / cell_size_) / 2);

		if (save) {
			std::ofstream file (se_file_name.c_str (), std::ios::out | std::ofstream::binary);
			std::copy (se_packed_contour_host.begin (),
			           se_packed_contour_host.end (),
			           std::ostreambuf_iterator<char> (file));
			std::cout << "[Spherical SE Packed Contour] : "
			          << se_file_name << " saved" << std::endl;
		}
	}

	se_packed_contour_size = se_packed_contour_host.size () / 12;
	cudaMalloc (&se_packed_contour, se_packed_contour_host.size ()*sizeof (char));
	cudaMemcpy (se_packed_contour,
	            &se_packed_contour_host[0],
	            se_packed_contour_host.size ()*sizeof (char),
	            cudaMemcpyHostToDevice);

}

void GMorpho::ComputeSphericalSEContour (char *& se_contour,
        unsigned int & se_contour_size,
        bool load, bool save, float se_size) {
	std::vector<Vec3c> se_contour_vec_host;
	std::vector<char> se_contour_host;
	int se_size_int = floor (2.f * se_size / cell_size_);
	std::string int_str;
	std::stringstream out;
	out << se_size_int;
	int_str = out.str();
	std::string se_file_name = "SphericalSESimple"
	                           + int_str + ".se";

	if (load) {
		std::basic_ifstream<char> file(se_file_name.c_str (),  std::ios::binary);
		se_contour_host = std::vector<char>(std::istreambuf_iterator<char>(file),
		                                    std::istreambuf_iterator<char>());
		std::cout << "[Spherical SE Simple Contour] : "
		          << se_file_name << " loaded" << std::endl;
	} else {
		ComputeSphericalContour (se_contour_vec_host,
		                         floor (2.f * se_size / cell_size_));
		se_contour_host.resize (3 * se_contour_vec_host.size ());
		for (int i = 0; i < se_contour_vec_host.size (); i++) {
			Vec3c contour_pos = se_contour_vec_host[i];
			se_contour_host[3 * i] = contour_pos[0];
			se_contour_host[3 * i + 1] = contour_pos[1];
			se_contour_host[3 * i + 2] = contour_pos[2];
		}
		if (save) {
			std::ofstream file (se_file_name.c_str (), std::ios::out | std::ofstream::binary);
			std::copy (se_contour_host.begin (),
			           se_contour_host.end (),
			           std::ostreambuf_iterator<char> (file));
			std::cout << "[Spherical SE Simple Contour] : "
			          << se_file_name << " saved" << std::endl;
		}
	}

	se_contour_size = se_contour_host.size () / 3;
	cudaMalloc (&se_contour, se_contour_host.size ()*sizeof (char));
	cudaMemcpy (se_contour,
	            &se_contour_host[0],
	            se_contour_host.size ()*sizeof (char),
	            cudaMemcpyHostToDevice);

}

void GMorpho::ComputeSphericalCSEPackedContour (char *& se_packed_contour,
        unsigned int & se_packed_contour_size,
        bool load, bool save, float se_size) {
	std::vector<Vec3c> se_contour_vec_host;
	std::vector<char> se_packed_contour_host;
	int se_size_int = floor (2.f * se_size / cell_size_);
	std::string int_str;
	std::stringstream out;
	out << se_size_int;
	int_str = out.str();
	std::string se_file_name = "SphericalCSE"
	                           + int_str + ".se";

	if (load) {
		std::basic_ifstream<char> file(se_file_name.c_str (),  std::ios::binary);
		se_packed_contour_host = std::vector<char>(std::istreambuf_iterator<char>(file),
		                         std::istreambuf_iterator<char>());
		std::cout << "[Spherical CSE Packed Contour] : "
		          << se_file_name << " loaded" << std::endl;
	} else {
		ComputeCSphericalContour (se_contour_vec_host,
		                          floor (2.f * se_size / cell_size_));
		BuildPackedSE (se_contour_vec_host, se_packed_contour_host,
		               floor (2.f * se_size / cell_size_) / 2);

		if (save) {
			std::ofstream file (se_file_name.c_str (), std::ios::out | std::ofstream::binary);
			std::copy (se_packed_contour_host.begin (),
			           se_packed_contour_host.end (),
			           std::ostreambuf_iterator<char> (file));
			std::cout << "[Spherical CSE Packed Contour] : "
			          << se_file_name << " saved" << std::endl;
		}
	}

	se_packed_contour_size = se_packed_contour_host.size () / 12;
	cudaMalloc (&se_packed_contour, se_packed_contour_host.size ()*sizeof (char));
	cudaMemcpy (se_packed_contour,
	            &se_packed_contour_host[0],
	            se_packed_contour_host.size ()*sizeof (char),
	            cudaMemcpyHostToDevice);

}

__global__ void ErodeByPackedFrontierSplatting (unsigned int * contour,
        unsigned char * contour_values,
        unsigned int size_contour,
        char * se_contour,
        unsigned int se_contour_size,
        unsigned char * input_voxel_grid,
        int se_size,
        GridGPU erosion_grid) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size_contour)
		return;

	uint3 res = erosion_grid.res;
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = contour[idx];
	resxy = res.x * res.y;
	id_z = key / resxy;
	remainder = key % resxy;
	id_y = remainder / res.x;
	id_x = remainder % res.x;

	unsigned char * voxel_grid = erosion_grid.voxels;
	unsigned int occup = contour_values[idx];
	unsigned int uint_key = 0;
	unsigned int uint_val = 0xffffffff;
	char4 cell_pos_se_ref;
	int3 cell_pos_se;
	uint2 occup_mask;
	uint2 dilation_mask;

	occup_mask.x = ((occup & 0x01) != 0x00) ? 0x000000ff : 0;
	occup_mask.x |= ((occup & 0x02) != 0x00) ? 0x0000ff00 : 0;
	occup_mask.x |= ((occup & 0x04) != 0x00) ? 0x00ff0000 : 0;
	occup_mask.x |= ((occup & 0x08) != 0x00) ? 0xff000000 : 0;
	occup_mask.y = ((occup & 0x10) != 0x00) ? 0x000000ff : 0;
	occup_mask.y |= ((occup & 0x20) != 0x00) ? 0x0000ff00 : 0;
	occup_mask.y |= ((occup & 0x40) != 0x00) ? 0x00ff0000 : 0;
	occup_mask.y |= ((occup & 0x80) != 0x00) ? 0xff000000 : 0;

	for (int i = 0; i < se_contour_size; i++) {
		cell_pos_se_ref = *(((char4*) se_contour) + 3 * i);
		dilation_mask.x = *(((uint*) se_contour) + 3 * i + 1);
		dilation_mask.y = *(((uint*) se_contour) + 3 * i + 2);
		cell_pos_se = make_int3 (id_x + (int)cell_pos_se_ref.x,
		                         id_y + (int)cell_pos_se_ref.y,
		                         id_z + (int)cell_pos_se_ref.z);

		unsigned int curr_uchar_key = res.x * res.y * cell_pos_se.z +
		                              res.x * cell_pos_se.y + cell_pos_se.x;
		unsigned int curr_uint_key = curr_uchar_key >> 2;
		unsigned int curr_uint_off = curr_uchar_key % 4;
		unsigned int curr_uint_shift = 8 * curr_uint_off;
		//		unsigned char curr_uchar_val;

		if (0 <= cell_pos_se.x && cell_pos_se.x < res.x
		        && 0 <= cell_pos_se.y && cell_pos_se.y < res.y
		        && 0 <= cell_pos_se.z && cell_pos_se.z < res.z
		   ) {

			// Mask with the actual occupation mask
			dilation_mask.x &= occup_mask.x;
			dilation_mask.y &= occup_mask.y;

			// Do a local dilation
			unsigned int perm_dilation_mask;
			dilation_mask.x |= dilation_mask.y;
			perm_dilation_mask = __byte_perm (dilation_mask.x, dilation_mask.x,
			                                  0x00000032);
			dilation_mask.x |= perm_dilation_mask;
			perm_dilation_mask = __byte_perm (dilation_mask.x, dilation_mask.x,
			                                  0x00000001);
			dilation_mask.x |= perm_dilation_mask;
			dilation_mask.x &= 0x000000ff;
			// The first dilation mask contain the reversed erosion pattern.
			// Now we shift it and inverse it to update the UINT value
			dilation_mask.x = (dilation_mask.x << curr_uint_shift);
			dilation_mask.x = ~dilation_mask.x;
			//			curr_uchar_val = dilation_mask.x;
			//			myAtomicOr (voxel_grid + curr_uchar_key, curr_uchar_val);

			if (uint_key == 0) {
				uint_key = curr_uint_key;
				uint_val &= dilation_mask.x;
				//				uint_val |= dilation_mask.x;
			} else {
				if (curr_uint_key != uint_key) {
					// Write uint
					atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
					//					atomicOr (((unsigned int *)voxel_grid) + uint_key, uint_val);
					// Update uint_key
					uint_key = curr_uint_key;
					uint_val = 0xffffffff;
					//					uint_val = 0;
					uint_val	&= dilation_mask.x;
					//					uint_val	|= dilation_mask.x;
				} else {
					uint_val	&= dilation_mask.x;
					//					uint_val	|= dilation_mask.x;
				}
			}
		}
	}
	if (uint_key != 0) {
		// Write uint
		atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
		//		atomicOr (((unsigned int *)voxel_grid) + uint_key, uint_val);
	}
}

__global__ void MorphoTaggingByErosion (unsigned int * contour,
                                        unsigned char * contour_values,
                                        unsigned int size_contour,
                                        char * se_contour,
                                        unsigned int se_contour_size,
                                        unsigned char * input_voxel_grid,
                                        int se_size,
                                        GridGPU erosion_grid,
                                        unsigned int * morpho_centroids_grid) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size_contour)
		return;

	uint3 res = erosion_grid.res;
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = contour[idx];
	resxy = res.x * res.y;
	id_z = key / resxy;
	remainder = key % resxy;
	id_y = remainder / res.x;
	id_x = remainder % res.x;

	unsigned char * voxel_grid = erosion_grid.voxels;
	//	unsigned int occup = input_voxel_grid[key];
	unsigned int occup = contour_values[idx];
	char4 cell_pos_se_ref;
	int3 cell_pos_se;
	uint2 occup_mask;
	uint2 dilation_mask;

	occup_mask.x = ((occup & 0x01) != 0x00) ? 0x000000ff : 0;
	occup_mask.x |= ((occup & 0x02) != 0x00) ? 0x0000ff00 : 0;
	occup_mask.x |= ((occup & 0x04) != 0x00) ? 0x00ff0000 : 0;
	occup_mask.x |= ((occup & 0x08) != 0x00) ? 0xff000000 : 0;
	occup_mask.y = ((occup & 0x10) != 0x00) ? 0x000000ff : 0;
	occup_mask.y |= ((occup & 0x20) != 0x00) ? 0x0000ff00 : 0;
	occup_mask.y |= ((occup & 0x40) != 0x00) ? 0x00ff0000 : 0;
	occup_mask.y |= ((occup & 0x80) != 0x00) ? 0xff000000 : 0;

	//	bool se_touch = false;
	//	unsigned int se_touch_idx[10];
	//	int cursor = -1;

	for (int i = 0; i < se_contour_size; i++) {
		cell_pos_se_ref = *(((char4*) se_contour) + 3 * i);
		dilation_mask.x = *(((uint*) se_contour) + 3 * i + 1);
		dilation_mask.y = *(((uint*) se_contour) + 3 * i + 2);
		cell_pos_se = make_int3 (id_x + (int)cell_pos_se_ref.x,
		                         id_y + (int)cell_pos_se_ref.y,
		                         id_z + (int)cell_pos_se_ref.z);

		unsigned int curr_uchar_key = res.x * res.y * cell_pos_se.z +
		                              res.x * cell_pos_se.y + cell_pos_se.x;
		unsigned char curr_uchar_val;

		if (0 <= cell_pos_se.x && cell_pos_se.x < res.x
		        && 0 <= cell_pos_se.y && cell_pos_se.y < res.y
		        && 0 <= cell_pos_se.z && cell_pos_se.z < res.z
		   ) {

			// Mask with the actual occupation mask
			dilation_mask.x &= occup_mask.x;
			dilation_mask.y &= occup_mask.y;

			// Do a local dilation
			unsigned int perm_dilation_mask;
			dilation_mask.x |= dilation_mask.y;
			perm_dilation_mask = __byte_perm (dilation_mask.x, dilation_mask.x,
			                                  0x00000032);
			dilation_mask.x |= perm_dilation_mask;
			perm_dilation_mask = __byte_perm (dilation_mask.x, dilation_mask.x,
			                                  0x00000001);
			dilation_mask.x |= perm_dilation_mask;
			dilation_mask.x &= 0x000000ff;
			curr_uchar_val = dilation_mask.x;
			if ((voxel_grid[curr_uchar_key] & curr_uchar_val) != 0) {
				//				se_touch = true;
				//				if (cursor < 9) {
				//					cursor++;
				//					//					se_touch_idx[cursor] = curr_uchar_key;
				//					morpho_centroids_grid[curr_uchar_key] = key;
				//				}
				morpho_centroids_grid[curr_uchar_key] = key;
			}

			//			if ((voxel_grid[curr_uchar_key] & curr_uchar_val) != 0)
			//				morpho_centroids[curr_uchar_key] = key;
			//			if (voxel_grid[curr_uchar_key] != 0)
			//				morpho_centroids[curr_uchar_key] = key;
			//			myAtomicOr (voxel_grid + curr_uchar_key, curr_uchar_val);
		}
	}

	//	if (se_touch) {
	//		for (int i = 0; i < cursor; i++) {
	//			morpho_centroids_grid[se_touch_idx[i]] = key;
	//		}
	//	}
}

__global__ void MorphoTaggingByBVH (unsigned int * cells,
                                    unsigned int cells_size,
                                    unsigned int * input_contour,
                                    unsigned int input_contour_size,
                                    BVHGPU input_bvh,
                                    GridGPU input_grid,
                                    GridGPU dilation_grid,
                                    GridGPU erosion_grid,
                                    float radius,
                                    unsigned int * morpho_centroids) {
	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	//	if (thread_idx >= 8*cells_size)
	//		return;
	if (thread_idx >= cells_size)
		return;

	//	unsigned int cell_idx = cells[thread_idx/8];
	unsigned int cell_idx = cells[thread_idx];
	//	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;
	//	unsigned int res = 2*input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	//	id_x += (0xaa & (1 << (threadIdx.x%8))) ? 1 : 0;
	//	id_y += (0xcc & (1 << (threadIdx.x%8))) ? 1 : 0;
	//	id_z += (0xf0 & (1 << (threadIdx.x%8))) ? 1 : 0;
	id_x += 1; id_y += 1; id_z += 1;
	//	id_x += 2;

	//	unsigned char * erosion_voxels = erosion_grid.voxels;

	//	float3 query_center = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	float3 query_center = make_float3 (id_x, id_y, id_z);
	//	unsigned char erosion_occup = 0xff;
	//	int max_depth = input_grid.max_mipmap_depth;
	float3 morpho_centroid;
	float sq_dist_min = 1e10;

	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	SphereNodePtr stack[64];
	SphereNodePtr * stack_ptr = stack;
	*stack_ptr++ = NULL;

	// Traverse nodes starting from the root
	//	SphereNodePtr node = input_bvh.internal_nodes_;

	for (int i = 0; i < input_bvh.num_objects_; i++) {
		float3 curr_node_center = input_bvh.leaf_nodes_[i].center_;
		float sq_dist = distanceS (curr_node_center, query_center);
		if (sq_dist < sq_dist_min) {
			morpho_centroid = curr_node_center;
			sq_dist_min = sq_dist;
		}
	}

	//	for (int i = 0; i < input_contour_size; i++) {
	//		float3 curr_node_center;
	//		Compute3DIdx (input_contour[i], res, resxy,
	//									curr_node_center);
	////		curr_node_center.x *= 2.f;
	////		curr_node_center.y *= 2.f;
	////		curr_node_center.z *= 2.f;
	//		float sq_dist = distanceS (curr_node_center, query_center);
	//		if (sq_dist < sq_dist_min) {
	//			morpho_centroid = curr_node_center;
	//			sq_dist_min = sq_dist;
	//		}
	//	}

	int3 int_morpho_centroid = make_int3 (morpho_centroid.x, morpho_centroid.y,
	                                      morpho_centroid.z);
	int_morpho_centroid.x /= 2;
	int_morpho_centroid.y /= 2;
	int_morpho_centroid.z /= 2;
	//	morpho_centroids[thread_idx] = (resxy/4)*int_morpho_centroid.z +
	//		(res/2)*int_morpho_centroid.y + int_morpho_centroid.x;
	morpho_centroids[thread_idx] = resxy * int_morpho_centroid.z +
	                               res * int_morpho_centroid.y + int_morpho_centroid.x;

	//	do {
	//		// Check each child node for overlap
	//		SphereNodePtr left_child_ptr = node->left_child_;
	//		SphereNodePtr right_child_ptr = node->right_child_;
	//
	//		SphereLeaf left_child = *((SphereLeafPtr) left_child_ptr);
	//		SphereLeaf right_child = *((SphereLeafPtr) right_child_ptr);
	//
	//		float left_sq_dist = distanceS (left_child.center_, query_center);
	//		float right_sq_dist = distanceS (right_child.center_, query_center);
	//
	//		bool left_overlap = left_sq_dist < left_child.sq_radius_;
	//		bool right_overlap = right_sq_dist < right_child.sq_radius_;
	//
	//		// Query overalps a leaf node => report collision.
	//		bool left_is_leaf = left_child.depth_ == max_depth;
	//		bool right_is_leaf = right_child.depth_ == max_depth;
	//		bool report_left_collision = left_overlap && left_is_leaf;
	//		bool report_right_collision = right_overlap && right_is_leaf;
	//
	//		if (report_left_collision || report_right_collision) {
	//			erosion_occup = 0x00;
	//			if (report_left_collision && (left_sq_dist < sq_dist_min)) {
	//				morpho_centroid = left_child.center_;
	//				sq_dist_min = left_sq_dist;
	//			}
	//			if (report_right_collision && (right_sq_dist < sq_dist_min)) {
	//				morpho_centroid = right_child.center_;
	//				sq_dist_min = right_sq_dist;
	//			}
	//			//			node = *--stack_ptr; // pop
	//			//			node = NULL;
	//		}
	//		// Query overlaps an internal node => traverse.
	//		bool traverse_left = left_overlap && !left_is_leaf;
	//		bool traverse_right = right_overlap && !right_is_leaf;
	//
	//		if (!traverse_left && !traverse_right)
	//			node = *--stack_ptr; // pop
	//		else {
	//			if (traverse_left && traverse_right) {
	//				bool l_or_r = left_sq_dist < right_sq_dist;
	//				node = l_or_r ? left_child_ptr : right_child_ptr;
	//				*stack_ptr++ = l_or_r ? right_child_ptr : left_child_ptr;
	//			} else {
	//				node = traverse_left ? left_child_ptr : right_child_ptr;
	//			}
	//		}
	//	} while (node != NULL);
	//
	//	if (erosion_occup == 0x00) {
	//		morpho_centroid.x /= 2.f;
	//		morpho_centroid.y /= 2.f;
	//		morpho_centroid.z /= 2.f;
	//		int3 int_morpho_centroid = make_int3 (morpho_centroid.x, morpho_centroid.y,
	//																					morpho_centroid.z);
	////		morpho_centroids[cell_idx] = resxy*int_morpho_centroid.z +
	////			res*int_morpho_centroid.y + int_morpho_centroid.x;
	//		morpho_centroids[thread_idx] = resxy*int_morpho_centroid.z +
	//			res*int_morpho_centroid.y + int_morpho_centroid.x;
	//	}
}

void GMorpho::ErodeByPackedSphericalSplatting (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	char * se_packed_contour = NULL;
	unsigned int se_packed_contour_size = 0;
	char * se_packed_contour_delta = NULL;
	unsigned int se_packed_contour_delta_size = 0;
	bool load = false, save = true;
	ComputeSphericalSEPackedContour (se_packed_contour,
	                                 se_packed_contour_size,
	                                 load, save, se_size);
	ComputeSphericalSEPackedContour (se_packed_contour_delta,
	                                 se_packed_contour_delta_size,
	                                 load, save,
	                                 se_size + (cell_size_ / 2.f));

	// Compute a contour list of the dilation
	unsigned int * dilation_contour = NULL;
	unsigned char * dilation_contour_values = NULL;
	unsigned int dilation_contour_size = 0;
	dilation_grid_.ComputeContourPacked (dilation_contour,
	                                     dilation_contour_values,
	                                     dilation_contour_size);

	block_size = 128;
	time1 = GET_TIME ();
	ErodeByPackedFrontierSplatting <<< (dilation_contour_size / block_size) + 1, block_size >>>
	(dilation_contour,
	 dilation_contour_values,
	 dilation_contour_size,
	 se_packed_contour, se_packed_contour_size,
	 dilation_grid_.grid_gpu ().voxels,
	 floor (2.f * se_size / cell_size_) / 2,
	 closing_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Backward Erosion] : "
	          << "erosion of radius " << floor (2.f * se_size / cell_size_)
	          << " on " << dilation_contour_size <<  " contour cells computed in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour);
	FreeGPUResource (&dilation_contour_values);
}

__global__ void ErodeByFrontierSplatting (unsigned int * contour,
        unsigned char * contour_values,
        unsigned int size_contour,
        char * se_contour,
        unsigned int se_contour_size,
        unsigned char * input_voxel_grid,
        int se_size,
        GridGPU erosion_grid) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size_contour)
		return;

	uint3 res = erosion_grid.res;
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = contour[idx];
	resxy = 4 * res.x * res.y;
	id_z = key / resxy;
	remainder = key % resxy;
	id_y = remainder / (2 * res.x);
	id_x = remainder % (2 * res.x);

	unsigned char * voxel_grid = erosion_grid.voxels;
	unsigned int uint_key = 0;
	unsigned int uint_val = 0xffffffff;
	char3 cell_pos_se_ref;
	int3 cell_pos_se;

	for (int i = 0; i < se_contour_size; i++) {
		cell_pos_se_ref = *(((char3*) se_contour) + i);
		cell_pos_se = make_int3 (id_x + (int)cell_pos_se_ref.x,
		                         id_y + (int)cell_pos_se_ref.y,
		                         id_z + (int)cell_pos_se_ref.z);

		unsigned int curr_uchar_key = res.x * res.y * (cell_pos_se.z / 2) +
		                              res.x * (cell_pos_se.y / 2) + (cell_pos_se.x / 2);
		unsigned int curr_uint_key = curr_uchar_key >> 2;
		unsigned int curr_uint_off = curr_uchar_key % 4;
		unsigned int curr_uint_shift = 8 * curr_uint_off;

		unsigned char cidx = cell_pos_se.x % 2;
		unsigned char cidy = cell_pos_se.y % 2;
		unsigned char cidz = cell_pos_se.z % 2;
		unsigned char curr_uchar_val = 1 << (4 * cidz + 2 * cidy + cidx);

		//		curr_uchar_val = ~curr_uchar_val;
		unsigned int curr_uchar_val_exp = curr_uchar_val;
		curr_uchar_val_exp = (curr_uchar_val_exp << curr_uint_shift);
		curr_uchar_val_exp = ~curr_uchar_val_exp;

		if (0 <= cell_pos_se.x && cell_pos_se.x < 2 * res.x
		        && 0 <= cell_pos_se.y && cell_pos_se.y < 2 * res.y
		        && 0 <= cell_pos_se.z && cell_pos_se.z < 2 * res.z
		   ) {

			if (uint_key == 0) {
				uint_key = curr_uint_key;
				uint_val &= curr_uchar_val_exp;
			} else {
				if (curr_uint_key != uint_key) {
					// Write uint
					atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
					//					atomicOr (((unsigned int *)voxel_grid) + uint_key, uint_val);
					// Update uint_key
					uint_key = curr_uint_key;
					uint_val = 0xffffffff;
					//					uint_val = 0;
					uint_val &= curr_uchar_val_exp;
				} else {
					uint_val &= curr_uchar_val_exp;
				}
			}
			//			myAtomicAnd (voxel_grid + curr_uchar_key, curr_uchar_val);
		}
	}
	if (uint_key != 0) {
		// Write uint
		atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
	}
}

void GMorpho::ErodeBySphericalSplatting (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	char * se_packed_contour = NULL;
	unsigned int se_packed_contour_size = 0;
	char * se_packed_contour_delta = NULL;
	unsigned int se_packed_contour_delta_size = 0;
	bool load = false, save = true;
	ComputeSphericalSEContour (se_packed_contour,
	                           se_packed_contour_size,
	                           load, save, se_size);
	ComputeSphericalSEContour (se_packed_contour_delta,
	                           se_packed_contour_delta_size,
	                           load, save,
	                           se_size + (cell_size_ / 2.f));

	// Compute a contour list of the dilation
	unsigned int * dilation_contour = NULL;
	unsigned char * dilation_contour_values = NULL;
	unsigned int dilation_contour_size = 0;
	dilation_grid_.ComputeContour (dilation_contour,
	                               dilation_contour_size);

	block_size = 128;
	time1 = GET_TIME ();
	ErodeByFrontierSplatting <<< (dilation_contour_size / block_size) + 1, block_size >>>
	(dilation_contour,
	 dilation_contour_values,
	 dilation_contour_size,
	 se_packed_contour, se_packed_contour_size,
	 dilation_grid_.grid_gpu ().voxels,
	 floor (2.f * se_size / cell_size_) / 2,
	 closing_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Backward Erosion] : "
	          << "erosion of radius " << floor (2.f * se_size / cell_size_)
	          << " on " << dilation_contour_size <<  " contour cells computed in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour);
	FreeGPUResource (&dilation_contour_values);
}

__global__ void ErodeByInteriorSplatting (unsigned int * contour,
        unsigned char * contour_values,
        unsigned int size_contour,
        char * se_contour,
        unsigned int se_contour_size,
        unsigned char * input_voxel_grid,
        int se_size,
        GridGPU erosion_grid) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= size_contour)
		return;

	uint3 res = erosion_grid.res;
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = contour[idx];
	resxy = 4 * res.x * res.y;
	id_z = key / resxy;
	remainder = key % resxy;
	id_y = remainder / (2 * res.x);
	id_x = remainder % (2 * res.x);

	unsigned char * voxel_grid = erosion_grid.voxels;
	unsigned int uint_key = 0;
	unsigned int uint_val = 0xffffffff;
	//	char3 cell_pos_se_ref;
	int3 cell_pos_se;
	se_size = 2 * se_size;
	for (int k = -se_size; k < se_size; k++)
		for (int j = -se_size; j < se_size; j++)
			for (int i = -se_size; i < se_size; i++) {
				cell_pos_se = make_int3 (id_x + i,
				                         id_y + j,
				                         id_z + k);

				if ((i * i + j * j + k * k) < se_size * se_size) {

					unsigned int curr_uchar_key = res.x * res.y * (cell_pos_se.z / 2) +
					                              res.x * (cell_pos_se.y / 2) + (cell_pos_se.x / 2);
					unsigned int curr_uint_key = curr_uchar_key >> 2;
					unsigned int curr_uint_off = curr_uchar_key % 4;
					unsigned int curr_uint_shift = 8 * curr_uint_off;

					unsigned char cidx = cell_pos_se.x % 2;
					unsigned char cidy = cell_pos_se.y % 2;
					unsigned char cidz = cell_pos_se.z % 2;
					unsigned char curr_uchar_val = 1 << (4 * cidz + 2 * cidy + cidx);

					//		curr_uchar_val = ~curr_uchar_val;
					unsigned int curr_uchar_val_exp = curr_uchar_val;
					curr_uchar_val_exp = (curr_uchar_val_exp << curr_uint_shift);
					curr_uchar_val_exp = ~curr_uchar_val_exp;

					if (0 <= cell_pos_se.x && cell_pos_se.x < 2 * res.x
					        && 0 <= cell_pos_se.y && cell_pos_se.y < 2 * res.y
					        && 0 <= cell_pos_se.z && cell_pos_se.z < 2 * res.z
					   ) {
						if (uint_key == 0) {
							uint_key = curr_uint_key;
							uint_val &= curr_uchar_val_exp;
						} else {
							if (curr_uint_key != uint_key) {
								// Write uint
								atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
								//					atomicOr (((unsigned int *)voxel_grid) + uint_key, uint_val);
								// Update uint_key
								uint_key = curr_uint_key;
								uint_val = 0xffffffff;
								//					uint_val = 0;
								uint_val &= curr_uchar_val_exp;
							} else {
								uint_val &= curr_uchar_val_exp;
							}
						}
						//			myAtomicAnd (voxel_grid + curr_uchar_key, curr_uchar_val);
					}
				}
			}
	if (uint_key != 0) {
		// Write uint
		atomicAnd (((unsigned int *)voxel_grid) + uint_key, uint_val);
	}
}

void GMorpho::ErodeByBallSplatting (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	char * se_packed_contour = NULL;
	unsigned int se_packed_contour_size = 0;
	char * se_packed_contour_delta = NULL;
	unsigned int se_packed_contour_delta_size = 0;
	bool load = false, save = true;
	ComputeSphericalSEContour (se_packed_contour,
	                           se_packed_contour_size,
	                           load, save, se_size);
	ComputeSphericalSEContour (se_packed_contour_delta,
	                           se_packed_contour_delta_size,
	                           load, save,
	                           se_size + (cell_size_ / 2.f));

	// Compute a contour list of the dilation
	unsigned int * dilation_contour = NULL;
	unsigned char * dilation_contour_values = NULL;
	unsigned int dilation_contour_size = 0;
	dilation_grid_.ComputeContour (dilation_contour,
	                               dilation_contour_size);

	block_size = 128;
	time1 = GET_TIME ();
	ErodeByInteriorSplatting <<< (dilation_contour_size / block_size) + 1, block_size >>>
	(dilation_contour,
	 dilation_contour_values,
	 dilation_contour_size,
	 se_packed_contour, se_packed_contour_size,
	 dilation_grid_.grid_gpu ().voxels,
	 floor (2.f * se_size / cell_size_) / 2,
	 closing_grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[Backward Erosion] : "
	          << "erosion of radius " << floor (2.f * se_size / cell_size_)
	          << " on " << dilation_contour_size <<  " contour cells computed in "
	          << time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&dilation_contour);
	FreeGPUResource (&dilation_contour_values);
}

__global__ void ErodeBySphereDynamicStack (BVHGPU input_bvh,
        GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU erosion_grid,
        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int res = input_grid.res.x;
	uint3 data_res = input_grid.data_res;

	if (id_x >= 2 * data_res.x || id_y >= 2 * data_res.y || id_z >= 2 * data_res.z)
		return;

	unsigned int i = res * res * (id_z / 2) + res * (id_y / 2) + (id_x / 2);
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	float3 query_center = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	unsigned char erosion_occup = 0xff;
	int max_depth = input_grid.max_mipmap_depth;

	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	SphereNodePtr stack[64];
	SphereNodePtr * stack_ptr = stack;
	*stack_ptr++ = NULL;

	// Traverse nodes starting from the root
	SphereNodePtr node = input_bvh.internal_nodes_;

	do {
		// Check each child node for overlap
		SphereNodePtr left_child_ptr = node->left_child_;
		SphereNodePtr right_child_ptr = node->right_child_;

		SphereLeaf left_child = *((SphereLeafPtr) left_child_ptr);
		SphereLeaf right_child = *((SphereLeafPtr) right_child_ptr);

		float left_sq_dist = distanceS (left_child.center_, query_center);
		float right_sq_dist = distanceS (right_child.center_, query_center);

		bool left_overlap = left_sq_dist < left_child.sq_radius_;
		bool right_overlap = right_sq_dist < right_child.sq_radius_;

		// Query overalps a leaf node => report collision.
		//		bool left_is_leaf = left_child.left_child_ == NULL
		//			&& left_child.right_child_ == NULL;
		//		bool right_is_leaf = right_child.left_child_ == NULL
		//			&& right_child.right_child_ == NULL;
		bool left_is_leaf = left_child.depth_ == max_depth;
		bool right_is_leaf = right_child.depth_ == max_depth;
		bool report_left_collision = left_overlap && left_is_leaf;
		bool report_right_collision = right_overlap && right_is_leaf;

		if (report_left_collision || report_right_collision) {
			erosion_occup = 0x00;
			break;
		}

		// Query overlaps an internal node => traverse.
		bool traverse_left = left_overlap && !left_is_leaf;
		bool traverse_right = right_overlap && !right_is_leaf;

		if (!traverse_left && !traverse_right)
			node = *--stack_ptr; // pop
		else {
			if (traverse_left && traverse_right) {
				bool l_or_r = left_sq_dist < right_sq_dist;
				node = l_or_r ? left_child_ptr : right_child_ptr;
				*stack_ptr++ = l_or_r ? right_child_ptr : left_child_ptr;
			} else {
				node = traverse_left ? left_child_ptr : right_child_ptr;
			}
		}
	} while (node != NULL);

	//	if (erosion_occup == 0x00)
	//		erosion_voxels[i] = 0x00;

	//	// Shared Memory Access by atomics
	//	unsigned char tidx = threadIdx.x%2;
	//	unsigned char tidy = threadIdx.y%2;
	//	unsigned char tidz = threadIdx.z%2;
	//	unsigned char erosion_fine = (1 << (4*tidz + 2*tidy + tidx));
	//
	//	// Shared Memory Initialization
	//	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
	//		shared_erosion_occup[blockDim.y*blockDim.x*(threadIdx.z/2) +
	//			blockDim.x*(threadIdx.y/2) + (threadIdx.x/2)] = 0x00;
	//	}
	//	__syncthreads ();
	//
	//	if (erosion_occup == 0x00)
	//		myAtomicOr (&shared_erosion_occup[blockDim.y*blockDim.x*(threadIdx.z/2) +
	//								 blockDim.x*(threadIdx.y/2) + (threadIdx.x/2)], erosion_fine);
	//	__syncthreads ();
	//
	//	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
	//		erosion_voxels[i] = shared_erosion_occup[blockDim.y*blockDim.x*(threadIdx.z/2) +
	//			blockDim.x*(threadIdx.y/2) + (threadIdx.x/2)];
	//	}

	// Shared Memory Access by atomics
	unsigned char tidx = threadIdx.x % 2;
	unsigned char tidy = threadIdx.y % 2;
	unsigned char tidz = threadIdx.z % 2;
	unsigned char erosion_fine = ~(1 << (4 * tidz + 2 * tidy + tidx));

	// Shared Memory Initialization
	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
		shared_erosion_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                     blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)] = 0xff;
	}
	__syncthreads ();

	if (erosion_occup == 0x00)
		myAtomicAnd (&shared_erosion_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                                   blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)], erosion_fine);
	__syncthreads ();

	if ((tidx == 0) && (tidy == 0) && (tidz == 0)) {
		erosion_voxels[i] = shared_erosion_occup[blockDim.y * blockDim.x * (threadIdx.z / 2) +
		                    blockDim.x * (threadIdx.y / 2) + (threadIdx.x / 2)];
	}
}

__global__ void ErodeBySphereDynamicStackBase (BVHGPU input_bvh,
        GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU erosion_grid,
        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int id_x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int id_z = blockIdx.z * blockDim.z + threadIdx.z;
	unsigned int res = input_grid.res.x;
	uint3 data_res = input_grid.data_res;

	if (id_x >= data_res.x || id_y >= data_res.y || id_z >= data_res.z)
		return;

	unsigned int i = res * res * id_z + res * id_y + id_x;
	unsigned char * erosion_voxels = erosion_grid.voxels;
	unsigned char * input_voxels = input_grid.voxels;
	unsigned char * dilation_voxels = dilation_grid.voxels;

	if (dilation_voxels[i] == 0x00 || input_voxels[i] == 0xff)
		return;

	//	float3 query_center = make_float3 (2*id_x + 1, 2*id_y + 1, 2*id_z + 1);
	float3 query_center = make_float3 (2 * id_x + 1 + 0.5f,
	                                   2 * id_y + 1 + 0.5f,
	                                   2 * id_z + 1 + 0.5f);
	unsigned char erosion_occup = 0xff;
	int max_depth = input_grid.max_mipmap_depth;

	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	SphereNodePtr stack[64];
	SphereNodePtr * stack_ptr = stack;
	*stack_ptr++ = NULL;

	// Traverse nodes starting from the root
	SphereNodePtr node = input_bvh.internal_nodes_;

	do {
		// Check each child node for overlap
		SphereNodePtr left_child_ptr = node->left_child_;
		SphereNodePtr right_child_ptr = node->right_child_;

		SphereLeaf left_child = *((SphereLeafPtr) left_child_ptr);
		SphereLeaf right_child = *((SphereLeafPtr) right_child_ptr);

		float left_sq_dist = distanceS (left_child.center_, query_center);
		float right_sq_dist = distanceS (right_child.center_, query_center);

		bool left_overlap = left_sq_dist < left_child.sq_radius_;
		bool right_overlap = right_sq_dist < right_child.sq_radius_;

		// Query overalps a leaf node => report collision.
		bool left_is_leaf = left_child.depth_ == max_depth;
		bool right_is_leaf = right_child.depth_ == max_depth;
		bool report_left_collision = left_overlap && left_is_leaf;
		bool report_right_collision = right_overlap && right_is_leaf;

		if (report_left_collision || report_right_collision) {
			erosion_occup = 0x00;
			break;
		}

		// Query overlaps an internal node => traverse.
		bool traverse_left = left_overlap && !left_is_leaf;
		bool traverse_right = right_overlap && !right_is_leaf;

		if (!traverse_left && !traverse_right)
			node = *--stack_ptr; // pop
		else {
			if (traverse_left && traverse_right) {
				bool l_or_r = left_sq_dist < right_sq_dist;
				node = l_or_r ? left_child_ptr : right_child_ptr;
				*stack_ptr++ = l_or_r ? right_child_ptr : left_child_ptr;
			} else {
				node = traverse_left ? left_child_ptr : right_child_ptr;
			}
		}
	} while (node != NULL);


	if (erosion_occup == 0x00)
		erosion_voxels[i] = 0x00;
}

__global__ void ErodeBySphereDynamicStack (unsigned int * cells,
        unsigned int cells_size,
        BVHGPU input_bvh,
        GridGPU input_grid,
        GridGPU dilation_grid,
        GridGPU erosion_grid,
        float radius) {
	extern __shared__ unsigned char shared_erosion_occup[];

	unsigned int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (thread_idx >= 8 * cells_size)
		return;

	unsigned int cell_idx = cells[thread_idx / 8];
	unsigned int i = cell_idx;
	unsigned int res = input_grid.res.x;

	// Compute 3D indices from linear indices
	unsigned int remainder, resxy, id_x, id_y, id_z;
	resxy = res * res;
	id_z = cell_idx / resxy;
	remainder = cell_idx % resxy;
	id_y = remainder / res;
	id_x = remainder % res;

	id_x *= 2; id_y *= 2; id_z *= 2;
	id_x += (0xaa & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_y += (0xcc & (1 << (threadIdx.x % 8))) ? 1 : 0;
	id_z += (0xf0 & (1 << (threadIdx.x % 8))) ? 1 : 0;

	unsigned char * erosion_voxels = erosion_grid.voxels;

	float3 query_center = make_float3 (id_x + 0.5f, id_y + 0.5f, id_z + 0.5f);
	unsigned char erosion_occup = 0xff;
	int max_depth = input_grid.max_mipmap_depth;

	// Allocate traversal stack from thread-local memory,
	// and push NULL to indicate that there are no postponed nodes.
	SphereNodePtr stack[64];
	SphereNodePtr * stack_ptr = stack;
	*stack_ptr++ = NULL;

	// Traverse nodes starting from the root
	SphereNodePtr node = input_bvh.internal_nodes_;

	do {
		// Check each child node for overlap
		SphereNodePtr left_child_ptr = node->left_child_;
		SphereNodePtr right_child_ptr = node->right_child_;

		SphereLeaf left_child = *((SphereLeafPtr) left_child_ptr);
		SphereLeaf right_child = *((SphereLeafPtr) right_child_ptr);

		float left_sq_dist = distanceS (left_child.center_, query_center);
		float right_sq_dist = distanceS (right_child.center_, query_center);

		bool left_overlap = left_sq_dist < left_child.sq_radius_;
		bool right_overlap = right_sq_dist < right_child.sq_radius_;

		// Query overalps a leaf node => report collision.
		bool left_is_leaf = left_child.depth_ == max_depth;
		bool right_is_leaf = right_child.depth_ == max_depth;
		bool report_left_collision = left_overlap && left_is_leaf;
		bool report_right_collision = right_overlap && right_is_leaf;

		if (report_left_collision || report_right_collision) {
			erosion_occup = 0x00;
			node = NULL;
		} else {
			// Query overlaps an internal node => traverse.
			bool traverse_left = left_overlap && !left_is_leaf;
			bool traverse_right = right_overlap && !right_is_leaf;

			if (!traverse_left && !traverse_right)
				node = *--stack_ptr; // pop
			else {
				if (traverse_left && traverse_right) {
					bool l_or_r = left_sq_dist < right_sq_dist;
					node = l_or_r ? left_child_ptr : right_child_ptr;
					*stack_ptr++ = l_or_r ? right_child_ptr : left_child_ptr;
				} else {
					node = traverse_left ? left_child_ptr : right_child_ptr;
				}
			}
		}
	} while (node != NULL);


	//	erosion_voxels[i] = 0xff;	/*{{{*/
	//	__syncthreads ();
	//
	//	if (erosion_occup == 0x00)
	//		erosion_voxels[i] = 0x00;
	//
	// Shared Memory Access by atomics
	//	unsigned char tid = threadIdx.x%8;
	//	unsigned char erosion_fine = (1 << tid);
	//
	// Shared Memory Initialization
	//	if (tid == 0) {
	//		shared_erosion_occup[threadIdx.x/8] = 0x00;
	//	}
	//	__syncthreads ();
	//
	//	myAtomicOr (&shared_erosion_occup[threadIdx.x/8], erosion_fine);
	//	__syncthreads ();
	//
	//	if (tid == 0) {
	//		erosion_voxels[i] = shared_erosion_occup[threadIdx.x/8];
	//	}/*}}}*/

	// Shared Memory Access by atomics
	unsigned char tid = threadIdx.x % 8;
	unsigned char erosion_fine = ~(1 << tid);

	// Shared Memory Initialization
	if (tid == 0) {
		shared_erosion_occup[threadIdx.x / 8] = 0xff;
	}
	__syncthreads ();

	if (erosion_occup == 0x00)
		myAtomicAnd (&shared_erosion_occup[threadIdx.x / 8], erosion_fine);
	__syncthreads ();

	if (tid == 0) {
		erosion_voxels[i] = shared_erosion_occup[threadIdx.x / 8];
	}
}

void GMorpho::ErodeByBVH (float se_size) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int block_size;

	// Compute a tight 26-connex contour list of the dilation
	unsigned int * dilation_contour = NULL;
	unsigned int dilation_contour_size = 0;
	dilation_grid_.ComputeContour (dilation_contour, dilation_contour_size);

	// Construct BVH from the contour
	dilation_contour_bvh_.BuildFromSEList (dilation_contour,
	                                       dilation_contour_size,
	                                       res_,
	                                       floor (2.f * se_size / cell_size_));


	// Run an erosion at the base resolution
	block_dim = dim3 (8, 4, 4);
	grid_dim = dim3 ((data_res_[0] / block_dim.x) + 1,
	                 (data_res_[1] / block_dim.y) + 1,
	                 (data_res_[2] / block_dim.z) + 1);
	time1 = GET_TIME ();
	ErodeBySphereDynamicStackBase <<< grid_dim, block_dim>>>
	(dilation_contour_bvh_.bvh_gpu (),
	 input_grid_.grid_gpu (),
	 dilation_grid_.grid_gpu (),
	 closing_grid_.grid_gpu (),
	 floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "base erosion in "
	          << time2 - time1 << " ms." << std::endl;

	// Compute a conservative contour
	unsigned int * closing_contour_base = NULL;
	unsigned int closing_contour_base_size = 0;
	closing_grid_.ComputeContourBaseTight (closing_contour_base,
	                                       closing_contour_base_size);

	//	unsigned int * closing_contour_base_host = NULL;/*{{{*/
	//	closing_contour_base_host = new unsigned int[closing_contour_base_size];
	//	cudaMemcpy (closing_contour_base_host, closing_contour_base,
	//							closing_contour_base_size*sizeof (unsigned int),
	//							cudaMemcpyDeviceToHost);
	//	SaveCubeList ("closing_contour_base.ply",
	//								closing_contour_base_host, closing_contour_base_size,
	//								bbox_, res_, cell_size_);
	//	free (closing_contour_base_host);
	//	closing_grid_.SetGridValue (0);/*}}}*/

	// Run an erosion at full resolution only on the
	// contour cells computed at base resolution

	block_size = 128;
	time1 = GET_TIME ();
	ErodeBySphereDynamicStack <<< (8 * closing_contour_base_size / block_size) + 1,
	                          block_size,
	                          (block_size / 8)*sizeof (unsigned char) >>>
	                          (closing_contour_base,
	                           closing_contour_base_size,
	                           dilation_contour_bvh_.bvh_gpu (),
	                           input_grid_.grid_gpu (),
	                           dilation_grid_.grid_gpu (),
	                           closing_grid_.grid_gpu (),
	                           floor (2.f * se_size / cell_size_));
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Erode] : " << "erosion by list of cells in "
	          << time2 - time1 << " ms." << std::endl;

	//	time1 = GET_TIME ();
	//	block_dim = dim3 (8, 4, 4);
	//	grid_dim = dim3 ((2*data_res_[0]/block_dim.x)+1,
	//									 (2*data_res_[1]/block_dim.y)+1,
	//									 (2*data_res_[2]/block_dim.z)+1);
	//	ErodeBySphereDynamicStack<<<grid_dim, block_dim,
	//		(block_dim.x*block_dim.y*block_dim.z/8)*sizeof (unsigned char)>>>
	//			(dilation_contour_bvh_.bvh_gpu (),
	//			 input_grid_.grid_gpu (),
	//			 dilation_grid_.grid_gpu (),
	//			 closing_grid_.grid_gpu (),
	//			 floor (2.f*se_size/cell_size_));
	//	cudaDeviceSynchronize ();
	//	CheckCUDAError ();
	//	time2 = GET_TIME ();
	//	std::cout << "[Erode] : " << "erosion in "
	//		<< time2 - time1 << " ms." << std::endl;

	FreeGPUResource (&closing_contour_base);
	ShowGPUMemoryUsage ();
}
