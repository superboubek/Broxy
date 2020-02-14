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

#ifndef  BROXY_BVH_H
#define  BROXY_BVH_H

#include <Common/Vec3.h>
#include <Common/BoundingVolume.h>

struct AABBNode {
	AABBNode * parent_;
	AABBNode * left_child_;
	AABBNode * right_child_;
	float3 min_;
	float3 max_;
	unsigned int object_id_;
	unsigned int atomic_counter_;
};

struct SphereNode {
	SphereNode * left_child_;
	SphereNode * right_child_;
	float3 center_;
	float sq_radius_;
	int depth_;
	SphereNode * parent_;
	float radius_;
	unsigned int atomic_counter_;
};

struct SphereLeaf {
	SphereNode * left_child_;
	SphereNode * right_child_;
	float3 center_;
	float sq_radius_;
	int depth_;
};

struct BVHGPU {
	float3 center_;
	float sq_radius_;
	float radius_;
	SphereNode * internal_nodes_;
	SphereNode * leaf_nodes_;
	int num_objects_;
	unsigned int * sorted_morton_;
	unsigned int * sorted_object_ids_;
};

typedef AABBNode* AABBNodePtr;
typedef SphereNode* SphereNodePtr;
typedef SphereLeaf* SphereLeafPtr;

class BVH {
public:
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("BVH Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	typedef MorphoGraphics::AxisAlignedBoundingBox BoundingBox;

	BVH ();
	virtual ~BVH ();

	inline const BoundingBox & bbox () const { return bbox_; }
	inline BVHGPU bvh_gpu () const { return bvh_gpu_; }

	void BuildFromSEList (unsigned int * se_list,
	                      int se_list_size,
	                      const MorphoGraphics::Vec3ui & res,
	                      float se_size);

protected:
	void print (const std::string & msg);
	void ShowGPUMemoryUsage ();

private:
	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);

	BoundingBox bbox_; // CPU Data
	BVHGPU bvh_gpu_; // GPU Data
};


#endif // BROXY_BVH_H
