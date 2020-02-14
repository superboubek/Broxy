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

#ifndef  BROXY_GRID_H
#define  BROXY_GRID_H

#include <string>
#include <stdint.h>

#include <Common/Vec3.h>
#include <Common/BoundingVolume.h>

#include "cuda_math.h"

#include "ScaleField.h"

#define MAX_MIPMAP_DEPTH 10

// A packed layout is made of a grid of unsigned char or bytes
// that represent 8 packed voxels arranged in a 8 pieces cube.

struct GridGPU {
	float3 bbox_min;
	float3 bbox_max;
	uint3 res;
	uint3 data_res;
	float cell_size;
	unsigned char * voxels;
	unsigned char * mipmap[MAX_MIPMAP_DEPTH];
	unsigned char * morton_mipmap[MAX_MIPMAP_DEPTH];
	unsigned char * ex_mipmap[MAX_MIPMAP_DEPTH];
	unsigned char * ex_morton_mipmap[MAX_MIPMAP_DEPTH];
	cudaTextureObject_t tex_mipmap;
	cudaMipmappedArray_t mipmapped_array;
	int max_mipmap_level;
	cudaTextureObject_t tex_dual_mipmap;
	float3 * morton_coords;
	int max_mipmap_depth;
};

#define C0 0x01
#define C1 0x02
#define C2 0x04
#define C3 0x08
#define C4 0x10
#define C5 0x20
#define C6 0x40
#define C7 0x80

#define LC0 0x01000000
#define LC1 0x02000000
#define LC2 0x04000000
#define LC3 0x08000000
#define LC4 0x10000000
#define LC5 0x20000000
#define LC6 0x40000000
#define LC7 0x80000000

namespace MorphoGraphics {

class Grid {
public:
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("Grid Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	typedef MorphoGraphics::AxisAlignedBoundingBox BoundingBox;

	// Type defining the grid connectivity
	typedef enum {GRID_3D_6C, GRID_3D_26C} GridConnectivity;

	Grid ();
	virtual ~Grid ();
	inline const BoundingBox & bbox () const { return bbox_; }
	inline const MorphoGraphics::Vec3ui & res () const { return res_; }
	inline const MorphoGraphics::Vec3ui & data_res () const { return data_res_; }
	inline float cell_size () const { return cell_size_; }
	inline GridGPU grid_gpu () const { return grid_gpu_; }
	inline void set_grid_gpu (const GridGPU & grid_gpu) { grid_gpu_ = grid_gpu; }

	void Init (const BoundingBox & bbox, const MorphoGraphics::Vec3ui & res,
	           const MorphoGraphics::Vec3ui & data_res, float cell_size);
	void SetGridValue (unsigned char value);
	void CopyVoxelsFrom (const Grid & grid);

	void BuildMipmaps ();
	void BuildTexMipmaps (float se_size);
	void BuildTexMipmaps (const MorphoGraphics::ScaleField & scale_field);
	void ClearTexMipmaps ();

	void BuildTexDualMipmaps (float se_size);
	void BuildMortonCoords ();

	void TransformToContour ();
	void ComputeContour (unsigned int * & contour, unsigned int & contour_size);
	void ComputeContourPacked (unsigned int * & contour,
	                           unsigned char * & contour_values,
	                           unsigned int & contour_size);
	void ComputeContourPacked (unsigned int * & contour,
	                           unsigned char * & contour_values,
	                           unsigned int & contour_size,
	                           GridConnectivity contour_type);
	void ComputeContourBase (unsigned int * & contour, unsigned int & contour_size);
	void ComputeContourBaseTight (unsigned int * & contour, unsigned int & contour_size);
	void ComputeContourBaseConservative (unsigned int * & contour, unsigned int & contour_size);
	void ComputeContourMCM (unsigned int * & contour, unsigned int & contour_size);
	/*void TestByMeshing (const std::string & filename);
	void TestByMeshingBilateral (const std::string & filename);
	void TestByMeshingUCHAR (const std::string & filename);*/
	void MipmapMeshing2x2x2 (int max_mipmap_depth);
	void Meshing2x2x2 (unsigned char * voxels, unsigned int base_res, float base_cell_size,
	                   const std::string & filename);
protected:
	void print (const std::string & msg);
	void ShowGPUMemoryUsage ();

private:
	// --------------------------------------------------------------
	//  CPU Data
	// --------------------------------------------------------------
	BoundingBox bbox_;
	MorphoGraphics::Vec3ui res_;
	MorphoGraphics::Vec3ui data_res_;
	float cell_size_;

	// --------------------------------------------------------------
	//  GPU Data
	// --------------------------------------------------------------
	GridGPU grid_gpu_;

	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);
};
} // MorphoGraphics

#endif // BROXY_GRID_H