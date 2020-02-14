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

#include <cstdio>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "cuda_math.h"
#include "BVH.h"

template<typename T>
void BVH::FreeGPUResource (T ** res) {
	if (*res != NULL) {
		cudaFree (*res);
		*res = NULL;
	} 
}

void BVH::print (const std::string & msg) {
	std::cout << "[BVH] : " << msg << std::endl;
}

void BVH::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if(err != cudaSuccess) {
		BVH::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw BVH::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

BVH::BVH () {
	bvh_gpu_.sorted_morton_ = NULL;
}

BVH::~BVH () {
}

void BVH::ShowGPUMemoryUsage () {
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);
	used = total - avail;
	std::cout << "[BVH] : Device memory used: " << (float)used/1e6 
		<< " of " << (float)total/1e6 << std::endl;
}

__global__ void ComputeMortonCodes (unsigned int * indices, 
																		unsigned int indices_size, 
																		uint3 indices_res, 
																		unsigned int * morton_codes) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx >= indices_size)
		return;

	// Compute 3D indices from linear indices
	unsigned int key, remainder, resxy, id_x, id_y, id_z;
	key = indices[idx];
	resxy = 4*indices_res.x*indices_res.y;
	id_z = key/resxy;
	remainder = key % resxy;
	id_y = remainder/(2*indices_res.x);
	id_x = remainder % (2*indices_res.x);

//	if (idx == 30000) {
////		printf ("center : %i %i %i\n", remainder % (2*indices_res.x), 
////						remainder/(2*indices_res.x), key/resxy);
//		printf ("center : %i %i %i\n", id_x, 
//						id_y, id_z);
//		printf ("indice : %i\n", key);
//	}
	// Compute 3D morton codes from 3D indices
	unsigned int morton = EncodeMorton3 (id_x, id_y, id_z);
	morton_codes[idx] = morton;
}

//__global__ void BuildAABBLeafNodes (BVHGPU bvh, 
//																		uint3 res, 
//																		float radius) {
//	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
//	if (idx >= bvh.num_objects_)
//		return;
//
//	// Compute 3D indices from linear indices
//	unsigned int object_id, remainder, resxy, id_x, id_y, id_z;
//	object_id = bvh.sorted_object_ids_[idx];
//	resxy = res.x*res.y;
//	id_z = object_id/resxy;
//	remainder = object_id % resxy;
//	id_y = remainder/res.x;
//	id_x = remainder % res.x;
//
//
//	bvh.leaf_nodes_[idx].object_id_ = object_id;
//	bvh.leaf_nodes_[idx].left_child_ = NULL;
//	bvh.leaf_nodes_[idx].right_child_ = NULL;
//
//	// Multiply by two the 3D indices to because each index
//	// referers to a cell containing 8 voxels encoded on 1 uchar
//	bvh.leaf_nodes_[idx].min_ = make_float3 (2*id_x + 1 - radius, 
//																					 2*id_y + 1 - radius, 
//																					 2*id_z + 1 - radius);
//	bvh.leaf_nodes_[idx].max_ = make_float3 (2*id_x + 1 + radius, 
//																					 2*id_y + 1 + radius, 
//																					 2*id_z + 1 + radius);
//
//	if (idx == 0)
//		printf ("leaf size : %f\n", distanceR (bvh.leaf_nodes_[idx].min_, 
//																					 bvh.leaf_nodes_[idx].max_));
//
//	bool left_child_ptr_null = bvh.leaf_nodes_[idx].left_child_ == NULL;
//	bool right_child_ptr_null = bvh.leaf_nodes_[idx].right_child_ == NULL;
//
//	if (!left_child_ptr_null || !right_child_ptr_null)
//		printf ("leaf %i hasn't all child to NULL %d\n", idx, !left_child_ptr_null || !right_child_ptr_null);
//
//
////	if (bvh.leaf_nodes_[idx].left_child_ != NULL ||
////			bvh.leaf_nodes_[idx].right_child_ != NULL);
////		printf ("leaf %i hasn't all child to NULL\n", idx);
//
//}

__device__ inline int Min (int i, int j) {
	return (i < j) ? i : j;
}

__device__ inline int Max (int i, int j) {
	return (i > j) ? i : j;
}

__device__ inline int Delta (int i, int j, 
														 unsigned int * keys, 
														 int max_num_key) {
	return (j < 0 || j >= max_num_key) ? -1 : __clz (keys[i] ^ keys[j]);
}

//__global__ void BuildAABBInternalNodes (BVHGPU bvh) {
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
//	int num_objects = bvh.num_objects_;
//
//	if (i >= (num_objects - 1))
//		return;
//
//	AABBNode * leaf_nodes = bvh.leaf_nodes_;
//	AABBNode * internal_nodes = bvh.internal_nodes_;
//	unsigned int * sorted_morton_codes = bvh.sorted_morton_;
//
////	int i_test = 0;/*{{{*/
///*}}}*/
//
//	// Determine the direction of the range (+1 or -1)	
//	int d = Delta (i, i + 1, sorted_morton_codes, num_objects) >
//		Delta (i, i - 1, sorted_morton_codes, num_objects) ? 1 : -1;
//
//	// Compute upper bound for the length range
//	int delta_min = Delta (i, i - d, sorted_morton_codes, num_objects);
//	int l_max = 2;
//	while (Delta (i, i + d*l_max, 
//								sorted_morton_codes, num_objects) > delta_min)
//		l_max = l_max*2;
//
//	// Find the other end using binary search
//	int l = 0;
//	for (int t = l_max/2; 0 < t; t/=2)
//		if (Delta (i, i + (l + t)*d, 
//							 sorted_morton_codes, num_objects) > delta_min)
//			l = l + t;
//
//	int j = i + l*d;
//
////	if (i == i_test) {/*{{{*/
////		printf ("delta (i, i + 1) : %i\n", 
////						Delta (i, i + 1, sorted_morton_codes, num_objects));
////		printf ("delta (i, i - 1) : %i\n", 
////						Delta (i, i - 1, sorted_morton_codes, num_objects));
////		printf ("d : %i\n", d);
////		printf ("delta min : %i\n", delta_min);
////		printf ("l_max : %i\n", l_max);
////		printf ("l : %i\n", l);
////	}/*}}}*/
//
//	// Find the split position using binary search
//	int delta_node = Delta (i, j, sorted_morton_codes, num_objects);
//
//	int s = 0, t = 1;
//	for (int k = 2; t > 0; k *= 2) {
//		t = ceilf ((float)l/(float)k);
//		if (Delta (i, i + (s + t)*d, 
//							 sorted_morton_codes, num_objects) > delta_node)
//			s = s + t;
//	}
//
//	int gamma = i + s*d + (d > 0 ? 0 : -1);
//
//	// Output child pointers
//	AABBNode * left_child, * right_child, * parent;
//
//	parent = internal_nodes + i;
//	if (Min (i, j) == gamma)
//		left_child = leaf_nodes + gamma;
//	else
//		left_child = internal_nodes + gamma;
//
//	if (Max (i, j) == (gamma + 1))
//		right_child = leaf_nodes + gamma + 1;
//	else
//		right_child = internal_nodes + gamma + 1;
//
//
//	internal_nodes[i].left_child_ = left_child;
//	internal_nodes[i].right_child_ = right_child;
//	left_child->parent_ = parent;
//	right_child->parent_ = parent;
//
////	if (i == i_test) {/*{{{*/
////		printf ("node index %i\n", i);
////		printf ("node selft x%.16x\n", internal_nodes + i);
////		printf ("node min %f %f %f\n", internal_nodes[i].min_.x, internal_nodes[i].min_.y, internal_nodes[i].min_.z);
////		printf ("node max %f %f %f\n", internal_nodes[i].max_.x, internal_nodes[i].max_.y, internal_nodes[i].max_.z);
////		printf ("node parent x%.16x\n", internal_nodes[i].parent_);
////		printf ("node left child x%.16x\n", internal_nodes[i].left_child_);
////		printf ("node right child x%.16x\n", internal_nodes[i].right_child_);
////		printf ("node range %i %i\n", i, j);
////		printf ("node gamma %i\n", gamma);
////	}/*}}}*/
//}

__device__ void ComputeAABB (const float3 & min0, const float3 & max0, 
														 const float3 & min1, const float3 & max1, 
														 float3 & min, float3 & max) {
	min.x = fminf (min0.x, min1.x);
	min.y = fminf (min0.y, min1.y);
	min.z = fminf (min0.z, min1.z);
	max.x = fmaxf (max0.x, max1.x);
	max.y = fmaxf (max0.y, max1.y);
	max.z = fmaxf (max0.z, max1.z);
}

//__global__ void ComputeBottomUpAABBs (BVHGPU bvh) {
//	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
//	int num_objects = bvh.num_objects_;
//
//	if (i >= num_objects)
//		return;
//
//	AABBNode * node = bvh.leaf_nodes_[i].parent_;
//	AABBNode * leaf_node = bvh.leaf_nodes_ + i;
//
////	if (bvh.leaf_nodes_[i].left_child_ != NULL ||
////			bvh.leaf_nodes_[i].right_child_ != NULL);
////		printf ("leaf %i hasn't all child to NULL\n", i);
//	
////		if (bvh.leaf_nodes_[i].left_child_ == NULL &&
////			bvh.leaf_nodes_[i].right_child_ == NULL);
////		printf ("leaf %i has all child to NULL\n", i);
//
////	bool left_child_ptr_null = bvh.leaf_nodes_[i].left_child_ == NULL;
////	bool right_child_ptr_null = bvh.leaf_nodes_[i].right_child_ == NULL;
////
////	if (!left_child_ptr_null || !right_child_ptr_null)
////		printf ("leaf %i hasn't all child to NULL %d\n", i, !left_child_ptr_null || !right_child_ptr_null);
//
//	if (i == 67406) {
//		//			if (node->parent_ == NULL) 
//		//				printf ("root\n");
//		printf ("leaf node min %f %f %f\n", leaf_node->min_.x, leaf_node->min_.y, leaf_node->min_.z);
//		printf ("leaf node max %f %f %f\n", leaf_node->max_.x, leaf_node->max_.y, leaf_node->max_.z);
//		printf ("leaf node left child %i\n", leaf_node->left_child_);
//		printf ("leaf node right child %i\n", leaf_node->right_child_);
//		printf ("leaf node parent x%.16x\n", leaf_node->parent_);
////		printf ("leaf is leaf %d\n", left_child_ptr_null && right_child_ptr_null);
//	}
//
//	while (node != NULL) {
//		if (atomicInc (&(node->atomic_counter_), 1) == 0)
//			return;
//		float3 min, max;
//		ComputeAABB (node->left_child_->min_, node->left_child_->max_, 
//								 node->right_child_->min_, node->right_child_->max_, 
//								 min, max);
//		node->min_ = min;
//		node->max_ = max;
//
//		if (node->parent_ == NULL) {
//			printf ("root\n");
//			printf ("root left child x%.16x\n", node->left_child_);
//			printf ("root right child x%.16x\n", node->right_child_);
//			printf ("node left child min %f %f %f\n", node->left_child_->min_.x, 
//							 node->left_child_->min_.y,  node->left_child_->min_.z);
//			printf ("node left child max %f %f %f\n", node->left_child_->max_.x, 
//							 node->left_child_->max_.y,  node->left_child_->max_.z);
//			printf ("node right child min %f %f %f\n", node->right_child_->min_.x, 
//							 node->right_child_->min_.y,  node->right_child_->min_.z);
//			printf ("node right child max %f %f %f\n", node->right_child_->max_.x, 
//							 node->right_child_->max_.y,  node->right_child_->max_.z);
//		}
//
//		//		if (i == 0) {/*{{{*/
//		//			//			if (node->parent_ == NULL) 
//		//			//				printf ("root\n");
//		//			printf ("node min\n", node->min_);
//		//			printf ("node max\n", node->max_);
//		//			printf ("node left child min\n", node->left_child_->min_);
//		//			printf ("node left child max\n", node->left_child_->max_);
//		//			printf ("node right child min\n", node->right_child_->min_);
//		//			printf ("node right child max\n", node->right_child_->max_);
//		//		}/*}}}*/
//		node = node->parent_;
//	}
//}

//void BVH::BuildFromSEListAABB (unsigned int * se_list, /*{{{*/
//															 int se_list_size, 
//															 const Vec3Dui & res, 
//															 float se_size) {
//	double time1, time2;
//
//	// Allocate GPU Data
//	cudaMalloc (&bvh_gpu_.sorted_morton_, se_list_size*sizeof (unsigned int));
//	cudaMalloc (&bvh_gpu_.sorted_object_ids_, se_list_size*sizeof (unsigned int));
//	cudaMemcpy (bvh_gpu_.sorted_object_ids_, se_list, 
//							se_list_size*sizeof (unsigned int), cudaMemcpyDeviceToDevice);
//
//	cudaMalloc (&bvh_gpu_.leaf_nodes_, se_list_size*sizeof (AABBNode));
//	cudaMemset (bvh_gpu_.leaf_nodes_, 0, se_list_size*sizeof (AABBNode));
//	cudaMalloc (&bvh_gpu_.internal_nodes_, (se_list_size - 1)*sizeof (AABBNode));
//	cudaMemset (bvh_gpu_.internal_nodes_, 0, (se_list_size - 1)*sizeof (AABBNode));
//
//	// Compute Morton Codes for each Structuring Element of the list
//	unsigned int block_size = 512;
//	uint3 res3 = make_uint3 (res[0], res[1], res[2]);
//
//	time1 = GET_TIME ();
//	ComputeMortonCodes<<<(se_list_size/block_size)+1,block_size>>> 
//		(se_list, se_list_size, res3, bvh_gpu_.sorted_morton_);
//	cudaDeviceSynchronize ();
//	time2 = GET_TIME ();
//	CheckCUDAError ();
//	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
//		<< " morton codes computed in " 
//		<< time2 - time1 << " ms." << std::endl;/*}}}*/
//
//	// Sort Morton Codes
//	thrust::device_ptr<unsigned int> devptr_keys (bvh_gpu_.sorted_morton_);
//	thrust::device_ptr<unsigned int> devptr_values (bvh_gpu_.sorted_object_ids_);
//	time1 = GET_TIME ();
//	thrust::sort_by_key (devptr_keys, 
//											 devptr_keys + se_list_size, 
//											 devptr_values);
//	cudaDeviceSynchronize ();
//	time2 = GET_TIME ();
//	CheckCUDAError ();
//	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
//		<< " morton codes sorted in " 
//		<< time2 - time1 << " ms." << std::endl;/*}}}*/
//
//	// Build the Leaf Nodes
//	bvh_gpu_.num_objects_ = se_list_size;
//	time1 = GET_TIME ();
//	BuildAABBLeafNodes<<<(se_list_size/block_size)+1,block_size>>> 
//		(bvh_gpu_, res3, se_size);
//	cudaDeviceSynchronize ();
//	time2 = GET_TIME ();
//	CheckCUDAError ();
//	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
//		<< " leaf nodes computed in " 
//		<< time2 - time1 << " ms." << std::endl;/*}}}*/
//
//	// Build the Internal Nodes
//	time1 = GET_TIME ();
//	BuildAABBInternalNodes<<<((se_list_size - 1)/block_size)+1,block_size>>> 
//		(bvh_gpu_);
//	cudaDeviceSynchronize ();
//	time2 = GET_TIME ();
//	CheckCUDAError ();
//	std::cout << "[BVH] : " << "SE List " << se_list_size - 1/*{{{*/
//		<< " internal nodes computed in " 
//		<< time2 - time1 << " ms." << std::endl;/*}}}*/
//
//	// Build AABBS of all the nodes starting from the leaf
//	time1 = GET_TIME ();
//	ComputeBottomUpAABBs<<<(se_list_size/block_size)+1,block_size>>> 
//		(bvh_gpu_);
//	cudaDeviceSynchronize ();
//	time2 = GET_TIME ();
//	CheckCUDAError ();
//	std::cout << "[BVH] : " << "All AABBs of the tree computed in " /*{{{*/
//		<< time2 - time1 << " ms." << std::endl;/*}}}*/
//}/*}}}*/

__global__ void BuildSphereLeafNodes (BVHGPU bvh, 
																			uint3 res, 
																			int max_depth, 
																			float radius) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx >= bvh.num_objects_)
		return;

	// Compute 3D indices from linear indices
	unsigned int object_id, remainder, resxy, id_x, id_y, id_z;
	object_id = bvh.sorted_object_ids_[idx];
	resxy = 4*res.x*res.y;
	id_z = object_id/resxy;
	remainder = object_id % resxy;
	id_y = remainder/(2*res.x);
	id_x = remainder % (2*res.x);


	bvh.leaf_nodes_[idx].depth_ = max_depth;
	bvh.leaf_nodes_[idx].left_child_ = NULL;
	bvh.leaf_nodes_[idx].right_child_ = NULL;

	// Multiply by two the 3D indices to because each index
	// referers to a cell containing 8 voxels encoded on 1 uchar
//	bvh.leaf_nodes_[idx].center_ = make_float3 (2*id_x + 1, 
//																							2*id_y + 1, 
//																							2*id_z + 1);
	
//	bvh.leaf_nodes_[idx].center_ = make_float3 (id_x, 
//																							id_y, 
//																							id_z);
	bvh.leaf_nodes_[idx].center_ = make_float3 (id_x + 0.5f, 
																							id_y + 0.5f, 
																							id_z + 0.5f);
	bvh.leaf_nodes_[idx].radius_ = radius;
	bvh.leaf_nodes_[idx].sq_radius_ = radius*radius;

//	if (idx == 30000) 
//		printf ("radius : %f\n", radius);
//		printf ("center : %i %i %i\n", id_x, id_y, id_z);

//	bool left_child_ptr_null = bvh.leaf_nodes_[idx].left_child_ == NULL;
//	bool right_child_ptr_null = bvh.leaf_nodes_[idx].right_child_ == NULL;

	//	if (!left_child_ptr_null || !right_child_ptr_null)
	//		printf ("leaf %i hasn't all child to NULL %d\n", idx, !left_child_ptr_null || !right_child_ptr_null);


	//	if (bvh.leaf_nodes_[idx].left_child_ != NULL ||
	//			bvh.leaf_nodes_[idx].right_child_ != NULL);
	//		printf ("leaf %i hasn't all child to NULL\n", idx);

}

__global__ void BuildSphereInternalNodes (BVHGPU bvh) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int num_objects = bvh.num_objects_;

	if (i >= (num_objects - 1))
		return;

	SphereNode * leaf_nodes = bvh.leaf_nodes_;
	SphereNode * internal_nodes = bvh.internal_nodes_;
	unsigned int * sorted_morton_codes = bvh.sorted_morton_;

	//	int i_test = 0;/*{{{*/
	/*}}}*/

	// Determine the direction of the range (+1 or -1)	
	int d = Delta (i, i + 1, sorted_morton_codes, num_objects) >
		Delta (i, i - 1, sorted_morton_codes, num_objects) ? 1 : -1;

	// Compute upper bound for the length range
	int delta_min = Delta (i, i - d, sorted_morton_codes, num_objects);
	int l_max = 2;
	while (Delta (i, i + d*l_max, 
								sorted_morton_codes, num_objects) > delta_min)
		l_max = l_max*2;

	// Find the other end using binary search
	int l = 0;
	for (int t = l_max/2; 0 < t; t/=2)
		if (Delta (i, i + (l + t)*d, 
							 sorted_morton_codes, num_objects) > delta_min)
			l = l + t;

	int j = i + l*d;

	//	if (i == i_test) {/*{{{*/
	//		printf ("delta (i, i + 1) : %i\n", 
	//						Delta (i, i + 1, sorted_morton_codes, num_objects));
	//		printf ("delta (i, i - 1) : %i\n", 
	//						Delta (i, i - 1, sorted_morton_codes, num_objects));
	//		printf ("d : %i\n", d);
	//		printf ("delta min : %i\n", delta_min);
	//		printf ("l_max : %i\n", l_max);
	//		printf ("l : %i\n", l);
	//	}/*}}}*/

	// Find the split position using binary search
	int delta_node = Delta (i, j, sorted_morton_codes, num_objects);

	int s = 0, t = 1;
	for (int k = 2; t > 0; k *= 2) {
		t = ceilf ((float)l/(float)k);
		if (Delta (i, i + (s + t)*d, 
							 sorted_morton_codes, num_objects) > delta_node)
			s = s + t;
	}

	int gamma = i + s*d + (d > 0 ? 0 : -1);

	// Output child pointers
	SphereNode * left_child, * right_child, * parent;

	parent = internal_nodes + i;
	if (Min (i, j) == gamma)
		left_child = leaf_nodes + gamma;
	else
		left_child = internal_nodes + gamma;

	if (Max (i, j) == (gamma + 1))
		right_child = leaf_nodes + gamma + 1;
	else
		right_child = internal_nodes + gamma + 1;

	internal_nodes[i].left_child_ = left_child;
	internal_nodes[i].right_child_ = right_child;
	internal_nodes[i].depth_ = 0;
	left_child->parent_ = parent;
	right_child->parent_ = parent;
	//	if (i == i_test) {/*{{{*/
	//		printf ("node index %i\n", i);
	//		printf ("node selft x%.16x\n", internal_nodes + i);
	//		printf ("node min %f %f %f\n", internal_nodes[i].min_.x, internal_nodes[i].min_.y, internal_nodes[i].min_.z);
	//		printf ("node max %f %f %f\n", internal_nodes[i].max_.x, internal_nodes[i].max_.y, internal_nodes[i].max_.z);
	//		printf ("node parent x%.16x\n", internal_nodes[i].parent_);
	//		printf ("node left child x%.16x\n", internal_nodes[i].left_child_);
	//		printf ("node right child x%.16x\n", internal_nodes[i].right_child_);
	//		printf ("node range %i %i\n", i, j);
	//		printf ("node gamma %i\n", gamma);
	//	}/*}}}*/
}

__device__ void ComputeSphere (const float3 & center0, float radius0, 
															 const float3 & center1, float radius1, 
															 float3 & center, float & radius) {
	float3 center_max = radius1 > radius0 ? center1 : center0;
	float3 center_min = radius1 > radius0 ? center0 : center1;
	float radius_max = radius1 > radius0 ? radius1 : radius0;
	float radius_min = radius1 > radius0 ? radius0 : radius1;

	float3 ref_dir = center_min - center_max;
	float ref_dist = sqrt (dotProduct (ref_dir, ref_dir));
	ref_dir = (1.f/ref_dist)*ref_dir;

	float3 min_point = center_min + radius_min*ref_dir;
	float3 max_point = center_max - radius_max*ref_dir;

	center = 0.5f*(max_point + min_point);
	radius = 0.5f*(radius_max + radius_min + ref_dist);

//	if (isnan (center0.x) || isnan (center0.y) || isnan (center0.z)
//			|| isnan (center0.x) || isnan (center0.y) || isnan (center0.z)
//			|| isnan (center1.x) || isnan (center1.y) || isnan (center1.z)
//			|| isnan (center1.x) || isnan (center1.y) || isnan (center1.z)
//			|| isnan (radius) || isnan (center.x) || isnan (center.y) || isnan (center.z)) {
//		printf ("node nan : %f %f %f %f\n", radius, ref_dist);
//	}
}

__global__ void ComputeBottomUpSpheres (BVHGPU bvh) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int num_objects = bvh.num_objects_;

	if (i >= num_objects)
		return;

	SphereNode * node = bvh.leaf_nodes_[i].parent_;

	//	SphereNode * leaf_node = bvh.leaf_nodes_ + i;/*{{{*/
	//	if (bvh.leaf_nodes_[i].left_child_ != NULL ||
	//			bvh.leaf_nodes_[i].right_child_ != NULL);
	//		printf ("leaf %i hasn't all child to NULL\n", i);

	//		if (bvh.leaf_nodes_[i].left_child_ == NULL &&
	//			bvh.leaf_nodes_[i].right_child_ == NULL);
	//		printf ("leaf %i has all child to NULL\n", i);

	//	bool left_child_ptr_null = bvh.leaf_nodes_[i].left_child_ == NULL;
	//	bool right_child_ptr_null = bvh.leaf_nodes_[i].right_child_ == NULL;
	//
	//	if (!left_child_ptr_null || !right_child_ptr_null)
	//		printf ("leaf %i hasn't all child to NULL %d\n", i, !left_child_ptr_null || !right_child_ptr_null);

	//	if (i == 67406) {
	//		//			if (node->parent_ == NULL) 
	//		//				printf ("root\n");
	//		printf ("leaf node center %f %f %f\n", leaf_node->center_.x, leaf_node->center_.y, leaf_node->center_.z);
	//		printf ("leaf node radius %f\n", leaf_node->radius_);
	//		printf ("leaf node left child %i\n", leaf_node->left_child_);
	//		printf ("leaf node right child %i\n", leaf_node->right_child_);
	//		printf ("leaf node parent x%.16x\n", leaf_node->parent_);
	//		//		printf ("leaf is leaf %d\n", left_child_ptr_null && right_child_ptr_null);
	//	}
	//	int trav = 0;
	//	int max_trav = 1;
/*}}}*/

	while (node != NULL 
				 //				 && trav < max_trav
				) {
		if (atomicInc (&(node->atomic_counter_), 1) == 0)
			return;
		float3 center;
		float radius;
		ComputeSphere (node->left_child_->center_, node->left_child_->radius_, 
									 node->right_child_->center_, node->right_child_->radius_, 
									 center, radius);
		node->center_ = center;
		node->radius_ = radius;
		node->sq_radius_ = radius*radius;

		//		trav++;/*{{{*/
		//		if (node->parent_ == NULL) {
		//			printf ("root\n");
		//			printf ("root left child x%.16x\n", node->left_child_);
		//			printf ("root right child x%.16x\n", node->right_child_);
		//			printf ("node left child center %f %f %f\n", node->left_child_->center_.x, 
		//							node->left_child_->center_.y,  node->left_child_->center_.z);
		//			printf ("node left child radius %f\n", node->left_child_->radius_);
		//			printf ("node right child center %f %f %f\n", node->right_child_->center_.x, 
		//							node->right_child_->center_.y,  node->right_child_->center_.z);
		//			printf ("node right child radius %f\n", node->right_child_->radius_);
		//		}

		//		if (i == 0) {
		//			//			if (node->parent_ == NULL) 
		//			//				printf ("root\n");
		//			printf ("node min\n", node->min_);
		//			printf ("node max\n", node->max_);
		//			printf ("node left child min\n", node->left_child_->min_);
		//			printf ("node left child max\n", node->left_child_->max_);
		//			printf ("node right child min\n", node->right_child_->min_);
		//			printf ("node right child max\n", node->right_child_->max_);
		//		}/*}}}*/

		node = node->parent_;
	}
}

void BVH::BuildFromSEList (unsigned int * se_list, 
													 int se_list_size, 
													 const MorphoGraphics::Vec3ui & res, 
													 float se_size) {
	double time1, time2;

	// Allocate GPU Data
	cudaMalloc (&bvh_gpu_.sorted_morton_, se_list_size*sizeof (unsigned int));
	cudaMalloc (&bvh_gpu_.sorted_object_ids_, se_list_size*sizeof (unsigned int));
	cudaMemcpy (bvh_gpu_.sorted_object_ids_, se_list, 
							se_list_size*sizeof (unsigned int), cudaMemcpyDeviceToDevice);

	cudaMalloc (&bvh_gpu_.leaf_nodes_, se_list_size*sizeof (SphereNode));
	cudaMemset (bvh_gpu_.leaf_nodes_, 0, se_list_size*sizeof (SphereNode));
	cudaMalloc (&bvh_gpu_.internal_nodes_, (se_list_size - 1)*sizeof (SphereNode));
	cudaMemset (bvh_gpu_.internal_nodes_, 0, (se_list_size - 1)*sizeof (SphereNode));

	// Compute Morton Codes for each Structuring Element of the list
	unsigned int block_size = 512;
	uint3 res3 = make_uint3 (res[0], res[1], res[2]);
	int max_depth = log2f (res[0]) + 1;

	time1 = GET_TIME ();
	ComputeMortonCodes<<<(se_list_size/block_size)+1,block_size>>> 
		(se_list, se_list_size, res3, bvh_gpu_.sorted_morton_);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
		<< " morton codes computed in " 
		<< time2 - time1 << " ms." << std::endl;/*}}}*/

	// Sort Morton Codes
	thrust::device_ptr<unsigned int> devptr_keys (bvh_gpu_.sorted_morton_);
	thrust::device_ptr<unsigned int> devptr_values (bvh_gpu_.sorted_object_ids_);
	time1 = GET_TIME ();
	thrust::sort_by_key (devptr_keys, 
											 devptr_keys + se_list_size, 
											 devptr_values);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
		<< " morton codes sorted in " 
		<< time2 - time1 << " ms." << std::endl;/*}}}*/

	// Build the Leaf Nodes
	bvh_gpu_.num_objects_ = se_list_size;
	time1 = GET_TIME ();
	BuildSphereLeafNodes<<<(se_list_size/block_size)+1,block_size>>> 
		(bvh_gpu_, res3, max_depth, se_size);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[BVH] : " << "SE List " << se_list_size /*{{{*/
		<< " leaf nodes computed in " 
		<< time2 - time1 << " ms." << std::endl;/*}}}*/

	// Build the Internal Nodes
	time1 = GET_TIME ();
	BuildSphereInternalNodes<<<((se_list_size - 1)/block_size)+1,block_size>>> 
		(bvh_gpu_);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[BVH] : " << "SE List " << se_list_size - 1/*{{{*/
		<< " internal nodes computed in " 
		<< time2 - time1 << " ms." << std::endl;/*}}}*/

	// Build AABBS of all the nodes starting from the leaf
	time1 = GET_TIME ();
	ComputeBottomUpSpheres<<<(se_list_size/block_size)+1,block_size>>> 
		(bvh_gpu_);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	CheckCUDAError ();
	std::cout << "[BVH] : " << "All AABBs of the tree computed in " /*{{{*/
		<< time2 - time1 << " ms." << std::endl;/*}}}*/
}
