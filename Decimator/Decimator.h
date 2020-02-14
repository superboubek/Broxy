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

#ifndef  Decimator_INC
#define  Decimator_INC

#define NDEBUG
#include <iostream>
#include <string>
#include <cassert>

#include "Common/Vec3.h"
#include "Common/Mesh.h"


class Decimator {
	public:
		Decimator ();
		Decimator (const MorphoGraphics::Mesh & mesh, int target_num_faces, 
							 float target_error, float max_edge_length, 
							 const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
							 const MorphoGraphics::Vec3<unsigned int> & res, 
							 float cell_size, char * scale_grid);
		Decimator (const MorphoGraphics::Mesh & mesh, int target_num_faces, 
							 float target_error, 
							 bool use_linear_constraints, bool use_features, 
							 float max_edge_length, 
							 const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
							 const MorphoGraphics::Vec3<unsigned int> & res, 
							 float cell_size, char * scale_grid, 
							 const std::vector<bool> & feature_taggs, 
							 int num_hc_iters = 3);
		virtual ~Decimator ();
		inline int num_faces () { return num_faces_; }
		inline int target_num_faces () { return target_num_faces_; }
		inline float error () { return error_; }
		inline float target_error () { return target_error_; }
		inline bool use_linear_constraints () { return use_linear_constraints_; }
		inline bool use_features () { return use_features_; }
		inline void set_num_faces (int num_faces) { num_faces_ = num_faces; }
		inline void set_target_num_faces (int target_num_faces) { target_num_faces_ = target_num_faces; }
		inline void set_target_error (float target_error) { target_error_ = target_error; }
		inline void set_error (float error) { error_ = error; }
		inline void set_use_linear_constraints (float use_linear_constraints) { use_linear_constraints_ = use_linear_constraints; }
		inline void set_use_features (float use_features) { use_features_ = use_features; }
		bool Optimize ();
		void GetMesh (MorphoGraphics::Mesh & output_mesh);

	private:
		int num_faces_;
		int target_num_faces_;
		float error_;
		float target_error_;
		bool use_linear_constraints_;
		bool use_features_;
		class VCGOptimSession;
		VCGOptimSession * vcg_optim_session_;
};

#endif   /* ----- #ifndef Decimator_INC  ----- */
