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


#ifndef  GMORPHO_INC
#define  GMORPHO_INC

#define GMORPHO_DEBUG

#include <Common/Vec3.h>
#include <Common/BoundingVolume.h>
#include <Common/Mesh.h>

#include "Grid.h"
#include "BVH.h"
#include "Voxelizer.h"
#include "FrameField.h"

#define MAX_NUM_MCM_CELLS 1600000
#define MAX_T_SIZE 2000000
#define MAX_V_SIZE 1000000

//#define MAX_NUM_MCM_CELLS (10*1600000)
//#define MAX_T_SIZE 13000000
//#define MAX_V_SIZE 7000000

namespace MorphoGraphics {
class GMorpho {
public:
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("GMorpho Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	typedef MorphoGraphics::AxisAlignedBoundingBox BoundingBox;

	GMorpho ();
	virtual ~GMorpho ();

	inline const BoundingBox & bbox () const { return bbox_; }
	inline const Vec3ui & res () const { return res_; }
	inline float cell_size () const { return cell_size_; }
	inline float se_size () const { return se_size_; }
	inline void set_se_size (float se_size)  { se_size_ = se_size; }
	inline void set_use_asymmetric_closing (bool use_asymmetric_closing) { use_asymmetric_closing_ = use_asymmetric_closing; }
	inline bool use_asymmetric_closing () { return use_asymmetric_closing_; }
	inline void set_use_frame_field (bool use_frame_field) { use_frame_field_ = use_frame_field; }
	inline bool use_frame_field () { return use_frame_field_; }
	inline void set_bilateral_filtering (bool bilateral_filtering) { bilateral_filtering_ = bilateral_filtering;}
	inline Mesh get_morpho_mesh () { return morpho_mesh_; }
	unsigned int * grid_2x2x2_uint () { return grid_2x2x2_uint_; }

	void Load (const float * P, const float * N, int num_of_vertices,
	           const unsigned int * T, int num_of_faces, int base_res,
	           float se_size);

	void Update (Mesh & mesh, const ScaleField & scale_field,
	             const FrameField & frame_field);
	void DilateBySphereMipmap (const ScaleField & scale_field);
	void DilateByRotCubeMipmap (const ScaleField & scale_field, const FrameField & frame_field);
	void DilateByCubeMipmap (const ScaleField & scale_field);
	void ErodeBySphereMipmap (const ScaleField & scale_field);
	void ErodeAllocation ();
	void ExtractClosingMeshAllocation ();

	void Update (Mesh & mesh, float se_size);
	void DilateByCubeMipmap (float se_size);
	void ErodeBySphereMipmap (float se_size);
	void ExtractClosingMesh ();
	void ExtractClosingMesh (Mesh & mesh, int num_bilateral_iters = 20);

	void DilateByMipmap (float se_size);
	void DilateBySphereMipmap (float se_size);
	void ErodeByMipmap (float se_size);
	void ErodeByMipmapFull (float se_size);
	void ErodeByBVH (float se_size);
	void ErodeByPackedSphericalSplatting (float se_size);
	void ErodeBySphericalSplatting (float se_size);
	void ErodeByBallSplatting (float se_size);

protected:
	void ComputeSphericalSEContour (char *& se_contour,
	                                unsigned int & se_contour_size,
	                                bool load, bool save,
	                                float se_size);
	void ComputeSphericalSEPackedContour (char *& se_packed_contour,
	                                      unsigned int & se_packed_contour_size,
	                                      bool load, bool save,
	                                      float se_size);
	void ComputeSphericalCSEPackedContour (char *& se_packed_contour,
	                                       unsigned int & se_packed_contour_size,
	                                       bool load, bool save,
	                                       float se_size);
	void print (const std::string & msg);
	void ShowGPUMemoryUsage ();

private:
	Voxelizer voxelizer_;

	// --------------------------------------------------------------
	//  CPU Data
	// --------------------------------------------------------------
	BoundingBox bbox_;
	Vec3ui res_;
	Vec3ui data_res_;
	float cell_size_;
	float se_size_;
	bool use_asymmetric_closing_;
	bool use_frame_field_;
	Mesh morpho_mesh_;

	// --------------------------------------------------------------
	//  GPU Data
	// --------------------------------------------------------------
	Grid input_grid_;
	Grid dilation_grid_;
	Grid dilation_contour_grid_;
	Grid closing_grid_;

	BVH dilation_contour_bvh_;

	float3 * v_morpho_mesh_;
	float3 * n_morpho_mesh_;
	uint3 * t_morpho_mesh_;
	unsigned int v_morpho_mesh_size_;
	unsigned int t_morpho_mesh_size_;

	int * global_warp_counter_;
	unsigned int * grid_2x2x2_uint_;
	unsigned int * mcm_contour_indices_;
	unsigned char * mcm_contour_values_;
	unsigned int * mcm_contour_non_empty_;
	unsigned char * mcm_compact_contour_values_;
	unsigned int * mcm_compact_contour_indices_;
	unsigned int * mcm_contour_neigh_indices_;
	unsigned int * mcm_compact_contour_neigh_morpho_centroids_;
	unsigned int * mcm_compact_contour_neigh_indices_;
	float3 * mcm_vertices;
	float3 * mcm_normals;
	float3 * mcm_bilateral_normals;
	uint3 * mcm_triangles;
	bool bilateral_filtering_;

	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);
};
}
#endif   /* ----- #ifndef GMORPHO_INC  ----- */
