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

#ifndef  VOXELIZER_INC
#define  VOXELIZER_INC

#include <string>

#include <Common/Vec3.h>
#include <Common/BoundingVolume.h>
#include <Common/Mesh.h>

#include "Grid.h"

namespace MorphoGraphics {

struct MeshGPU {
	float3 * vertices;
	float3 * normals;
	int num_of_vertices;
	uint3 * faces;
	int num_of_faces;
};

class Voxelizer {
public:
	class Exception {
	public:
		inline Exception (const std::string & msg) : msg_ ("Voxelizer Error: " + msg) {}
		inline const std::string & msg () const { return msg_; }
	protected:
		std::string msg_;
	};

	Voxelizer ();
	virtual ~Voxelizer ();
	inline const AxisAlignedBoundingBox & bbox () const { return bbox_; }
	inline const Vec3ui & res () const { return res_; }
	inline const Vec3ui & data_res () const { return data_res_; }
	inline float cell_size () const { return cell_size_; }
	inline Grid grid () const { return grid_; }


	void Load (const float * P, const float * N, int num_of_vertices,
	           const unsigned int * T, int num_of_faces);
	void ComputeGridAttributes (int base_res, float margin);

	void VoxelizeByMultiSlicing (int base_res, float margin);
	void VoxelizeConservative (int base_res, float margin);
	void LoadShaders ();
	void BuildFBO (int res, int num_bit_vox);
	void ClearFBO ();
	void VoxelizeSurfaceConservative (GL::Program * surf_vox_conserv_program);
	void VoxelizeVolume (GL::Program * vol_vox_program);

	void ConvertBucketToPacketGrid (const std::vector<unsigned int> & buck_vox_grid,
	                                int res, int num_bit_vox,
	                                std::vector<unsigned char> & pack_vox_grid);
	void ParseGrid (const std::vector<unsigned int> & vox_grid,
	                const AxisAlignedBoundingBox & bbox,
	                int res, int num_bit_vox,
	                std::vector<Vec3f> & positions,
	                std::vector<Vec3f> & normals);

protected:
	void print (const std::string & msg);
	void TestByMeshing (const std::string & filename);
	void TestByMeshing (const std::string & filename,
	                    const GridGPU & grid_gpu);
private:
	// --------------------------------------------------------------
	//  CPU Data
	// --------------------------------------------------------------
	AxisAlignedBoundingBox bbox_;
	unsigned int base_res_;
	Vec3ui res_;
	Vec3ui data_res_;
	float cell_size_;

	// --------------------------------------------------------------
	//  OpenGL Data
	// --------------------------------------------------------------
	MorphoGraphics::Mesh mesh_;
	GLuint vox_fbo_;
	GLuint vox_fbo_tex_;
	GLuint vox_tex_;
	GL::Program * surf_vox_conserv_bit_program_;
	GL::Program * vol_vox_bit_program_;
	GL::Program * surf_vox_conserv_byte_program_;
	GL::Program * vol_vox_byte_program_;

	// --------------------------------------------------------------
	//  GPU Data
	// --------------------------------------------------------------
	MeshGPU mesh_gpu_;
	Grid grid_;

	void CheckCUDAError ();
	template<typename T>
	void FreeGPUResource (T ** res);
};
}

#endif   /* ----- #ifndef VOXELIZER_INC  ----- */
