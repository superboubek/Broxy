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

#include<GL/glew.h>

#include <cfloat>

#include <Common/OpenGL.h>
#include <Common/BoundingVolume.h>
#include <GL/glext.h>

#include "cuda_math.h"
#include "MarchingCubesMesher.h"

#include "Voxelizer.h"

using namespace std;
using namespace MorphoGraphics;

template<typename T>
void Voxelizer::FreeGPUResource (T ** res) {
	if (*res != 0) {
		cudaFree (*res);
		*res = 0;
	}
}

void Voxelizer::CheckCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if (err != cudaSuccess) {
		Voxelizer::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw Voxelizer::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}

void Voxelizer::print (const std::string & msg) {
	std::cout << "[Voxelizer]: " << msg << std::endl;
}

Voxelizer::Voxelizer () {
	vox_fbo_ = 0;
	vox_fbo_tex_ = 0;
	mesh_gpu_.vertices = 0;
	mesh_gpu_.normals = 0;
	mesh_gpu_.faces = 0;
}

Voxelizer::~Voxelizer () {
	FreeGPUResource (&mesh_gpu_.vertices);
	FreeGPUResource (&mesh_gpu_.normals);
	FreeGPUResource (&mesh_gpu_.faces);
}

void Voxelizer::Load (const float * P, const float * N, int num_of_vertices,
                      const unsigned int * T, int num_of_faces) {
	mesh_.clear ();
	vector<Vec3f> & MP = mesh_.P ();
	vector<Vec3f> & MN = mesh_.N ();
	vector<Mesh::Triangle> & MT = mesh_.T();
	MP.resize (num_of_vertices);
	MN.resize (num_of_vertices);
	for (size_t i = 0; i < num_of_vertices; i++) {
		MP[i] = Vec3f (P[3 * i], P[3 * i + 1], P[3 * i + 2]);
		MN[i] = Vec3f (N[3 * i], N[3 * i + 1], N[3 * i + 2]);
	}
	MT.resize (num_of_faces);
	for (size_t i = 0; i < num_of_faces; i++)
		MT[i] = Mesh::Triangle (T[3 * i], T[3 * i + 1], T[3 * i + 2]);

	mesh_gpu_.num_of_vertices = num_of_vertices;
	mesh_gpu_.num_of_faces = num_of_faces;
	cudaMalloc ((void**)&mesh_gpu_.vertices, num_of_vertices * sizeof (float3));
	cudaMalloc ((void**)&mesh_gpu_.normals, num_of_vertices * sizeof (float3));
	cudaMalloc ((void**)&mesh_gpu_.faces, num_of_faces * sizeof (uint3));
	cudaMemcpy (mesh_gpu_.vertices, MP.data (), num_of_vertices * sizeof (float3), cudaMemcpyHostToDevice);
	cudaMemcpy (mesh_gpu_.normals, MN.data (), num_of_vertices * sizeof (float3), cudaMemcpyHostToDevice);
	cudaMemcpy (mesh_gpu_.faces, MT.data (), num_of_faces * sizeof (uint3), cudaMemcpyHostToDevice);
	CheckCUDAError ();

	std::cout << "[Voxelizer] : loaded " << num_of_vertices
	          << " point-normal vertices and " << num_of_faces << " faces loaded " << std::endl;
}

__device__ 
unsigned char atomicFlip (unsigned char * address) {
	unsigned int * base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214,  0x3240,  0x3410,  0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, flip, new_;

	old = *base_address;

	do {
		assumed = old;
		flip = ~((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
		new_ = __byte_perm(old,  flip, sel);

		if (new_ == old)
			break;

		old = atomicCAS(base_address, assumed, new_);

	} while (assumed != old);

	return old;
}

__device__ 
unsigned char atomicXor (unsigned char * address, unsigned char val) {
	unsigned int * base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214,  0x3240,  0x3410,  0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, flip, new_;

	old = *base_address;

	do {
		assumed = old;
		flip = val ^ ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
		//		flip = val | ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
		new_ = __byte_perm(old,  flip, sel);

		if (new_ == old)
			break;

		old = atomicCAS(base_address, assumed, new_);

	} while (assumed != old);

	return old;
}

__device__ 
unsigned char atomicOr (unsigned char * address, unsigned char val) {
	unsigned int * base_address = (unsigned int *)((size_t)address & ~3);
	unsigned int selectors[] = {0x3214,  0x3240,  0x3410,  0x4210};
	unsigned int sel = selectors[(size_t)address & 3];
	unsigned int old, assumed, flip, new_;

	old = *base_address;

	do {
		assumed = old;
		flip = val | ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
		//		flip = val | ((char)__byte_perm(old, 0, ((size_t)address & 3) | 0x4440));
		new_ = __byte_perm(old,  flip, sel);

		if (new_ == old)
			break;

		old = atomicCAS(base_address, assumed, new_);

	} while (assumed != old);

	return old;
}

__device__ 
void triBBoxYZ (const float3 & v0, const float3 & v1, const float3 & v2,
                           float3 & triBBoxYZMin, float3 & triBBoxYZMax) {
	triBBoxYZMin.y = min (min (v0.y, v1.y), v2.y);
	triBBoxYZMin.z = min (min (v0.z, v1.z), v2.z);
	triBBoxYZMax.y = max (max (v0.y, v1.y), v2.y);
	triBBoxYZMax.z = max (max (v0.z, v1.z), v2.z);
}

__global__ 
void __launch_bounds__ (512, 4) MultiSlicing (float * vertices, unsigned int numOfVertices,
        								 	  unsigned int * faces, unsigned int numOfFaces,
        							     	  GridGPU grid) {
	unsigned int triIdx = blockIdx.x * blockDim.x + threadIdx.x;
	uint3 res = grid.res;
	float cellSize = grid.cell_size;

	if (triIdx >= numOfFaces)
		return;

	unsigned char * voxelGrid = grid.voxels;
	float3 ux = make_float3 (1, 0, 0);

	// Fetch Triangle
	float3 v0, v1, v2;
	uint3 tri;
	tri = make_uint3 (faces[3 * triIdx], faces[3 * triIdx + 1], faces[3 * triIdx + 2]);
	v0 = make_float3 (vertices[3 * tri.x],
	                  vertices[3 * tri.x + 1],
	                  vertices[3 * tri.x + 2]);
	v1 = make_float3 (vertices[3 * tri.y],
	                  vertices[3 * tri.y + 1],
	                  vertices[3 * tri.y + 2]);
	v2 = make_float3 (vertices[3 * tri.z],
	                  vertices[3 * tri.z + 1],
	                  vertices[3 * tri.z + 2]);

	// Compute YZ BBox and cell ranges of the YZ BBox
	float3 triBBoxYZMin, triBBoxYZMax;
	uint3 triCellYZMin, triCellYZMax;

	triBBoxYZ (v0, v1, v2, triBBoxYZMin, triBBoxYZMax);
	triCellYZMin = make_uint3 (0,
	                           floor ((triBBoxYZMin.y - grid.bbox_min.y) / cellSize),
	                           floor ((triBBoxYZMin.z - grid.bbox_min.z) / cellSize));
	triCellYZMax = make_uint3 (0,
	                           floor ((triBBoxYZMax.y - grid.bbox_min.y) / cellSize),
	                           floor ((triBBoxYZMax.z - grid.bbox_min.z) / cellSize));

	// Compute the triangle normal direction
	float3 n = crossProduct (v1 - v0, v2 - v0);
	normalize (n);

	// Compute edges normals in YZ plane
	float3 e0, e1, e2, ne0yz, ne1yz, ne2yz;
	e0 = v1 - v0;
	e1 = v2 - v1;
	e2 = v0 - v2;

	ne0yz = make_float3 (0, -e0.z, e0.y);
	ne0yz = copysign (1.f, n.x) * ne0yz;

	ne1yz = make_float3 (0, -e1.z, e1.y);
	ne1yz = copysign (1.f, n.x) * ne1yz;

	ne2yz = make_float3 (0, -e2.z, e2.y);
	ne2yz = copysign (1.f, n.x) * ne2yz;

	// Compute Top-Left Fill Rules
	float fe0yz, fe1yz, fe2yz;
	fe0yz = (ne0yz.y > 0) || (ne0yz.y == 0 && ne0yz.z < 0) ? FLT_MIN : 0;
	fe1yz = (ne1yz.y > 0) || (ne1yz.y == 0 && ne1yz.z < 0) ? FLT_MIN : 0;
	fe2yz = (ne2yz.y > 0) || (ne2yz.y == 0 && ne2yz.z < 0) ? FLT_MIN : 0;

	float3 vMean0 = v0 + v1;
	vMean0 = 0.5f * vMean0;
	float3 vMean1 = v2 + v1;
	vMean1 = 0.5f * vMean1;
	float3 vMean2 = v0 + v2;
	vMean2 = 0.5f * vMean2;

	for (unsigned int j = triCellYZMin.y; j <= triCellYZMax.y; j++)
		for (unsigned int k = triCellYZMin.z; k <= triCellYZMax.z; k++) {
			int qi[4] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
			for (unsigned int jsub = 0; jsub < 2; jsub++)
				for (unsigned int ksub = 0; ksub < 2; ksub++) {
					// Compute Edges Functions
					float3 pyz, pyzv0, pyzv1, pyzv2;
					pyz = make_float3 (0,
					                   grid.bbox_min.y + j * cellSize
					                   + jsub * 0.5f * cellSize + 0.25f * cellSize
					                   ,
					                   grid.bbox_min.z + k * cellSize
					                   + ksub * 0.5f * cellSize + 0.25f * cellSize
					                  );
					float e0Func, e1Func, e2Func;

					pyzv0.y = pyz.y - vMean0.y;
					pyzv0.z = pyz.z - vMean0.z;
					pyzv1.y = pyz.y - vMean1.y;
					pyzv1.z = pyz.z - vMean1.z;
					pyzv2.y = pyz.y - vMean2.y;
					pyzv2.z = pyz.z - vMean2.z;

					e0Func = ne0yz.y * pyzv0.y + ne0yz.z * pyzv0.z;
					e1Func = ne1yz.y * pyzv1.y + ne1yz.z * pyzv1.z;
					e2Func = ne2yz.y * pyzv2.y + ne2yz.z * pyzv2.z;

					bool test = ((e0Func + fe0yz) > 0) && ((e1Func + fe1yz) > 0) && ((e2Func + fe2yz) > 0);

					if (test) {
						// Compute Projection of pyz on the triangle
						pyz.x = grid.bbox_min.x;
						float s = dotProduct (n, v0 - pyz) / n.x;
						float3 p = pyz + s * ux;

						// Compute inital range X axis index
						qi[jsub + 2 * ksub] = (p.x - grid.bbox_min.x) / (0.5f * cellSize);
					}
				}

			unsigned int minqi = min (min (qi[0], qi[1]), min (qi[2], qi[3]));
			unsigned int startqi = minqi / 2;

			//			if (j == 78 && k == -146 && (30 <= (int)startqi && (int)startqi <= 41)) {
			//				printf ("tri %i\n", triIdx);
			//			}

			for (int i = startqi; 0 <= i && i < res.x
			        //						 && i < startqi + 30
			        ; i++) {
				unsigned char val = 0;
				if (qi[0] <= 2 * i + 1)
					val |= 0x02;
				if (qi[1] <= 2 * i + 1)
					val |= 0x08;
				if (qi[2] <= 2 * i + 1)
					val |= 0x20;
				if (qi[3] <= 2 * i + 1)
					val |= 0x80;

				if (qi[0] <= 2 * i)
					val |= 0x01;
				if (qi[1] <= 2 * i)
					val |= 0x04;
				if (qi[2] <= 2 * i)
					val |= 0x10;
				if (qi[3] <= 2 * i)
					val |= 0x40;

				atomicXor (voxelGrid + (i + j * res.x + k * res.x * res.y), val);
			}
		}
}

__global__ void
__launch_bounds__ (512, 4) RobustSimple (float * vertices, unsigned int numOfVertices,
        							     unsigned int * faces, unsigned int numOfFaces,
        								 GridGPU grid) {
	unsigned int triIdx = blockIdx.x * blockDim.x + threadIdx.x;
	uint3 res = grid.res;
	float cellSize = grid.cell_size;

	if (triIdx >= numOfVertices)
		return;

	unsigned char * voxelGrid = grid.voxels;
	float3 ux = make_float3 (1, 0, 0);

	// Fetch Triangle
	float3 v0;

	v0 = make_float3 (vertices[3 * triIdx],
	                  vertices[3 * triIdx + 1],
	                  vertices[3 * triIdx + 2]);

	uint3 v0_cell;
	//	uint3 v0_cell, v1_cell, v2_cell;
	v0_cell = make_uint3 (floor ((v0.x - grid.bbox_min.x) / (0.5f * cellSize)),
	                      floor ((v0.y - grid.bbox_min.y) / (0.5f * cellSize)),
	                      floor ((v0.z - grid.bbox_min.z) / (0.5f * cellSize)));

	// Shared Memory Access by atomics
	unsigned char cidx, cidy, cidz, v_val;
	uint3 v_cell;

	v_cell = v0_cell;
	cidx = v_cell.x % 2;
	cidy = v_cell.y % 2;
	cidz = v_cell.z % 2;
	v_val = 1 << (4 * cidz + 2 * cidy + cidx);
	atomicOr (voxelGrid + ((v_cell.x / 2) +
	                       (v_cell.y / 2)*res.x +
	                       (v_cell.z / 2)*res.x * res.y), v_val);
}


void Voxelizer::VoxelizeByMultiSlicing (int base_res, float margin) {
	double time1, time2;

	// Compute Grid Attributes
	ComputeGridAttributes (base_res, margin);

	std::cout << "[Voxelizer] : " << "Cell Size : " << cell_size_ << std::endl;
	std::cout << "[Voxelizer] : "	<< "Grid Resolution : "
	          << res_[0] << "x" << res_[1] << "x" << res_[2] << std::endl;
	std::cout << "[Voxelizer] : "	<< "Bounding Box : "
	          << bbox_.min() << " | " << bbox_.max() << std::endl;
	std::cout << "[Voxelizer] : "	<< "Grid Data Resolution : "
	          << data_res_[0] << "x" << data_res_[1] << "x" << data_res_[2] << std::endl;

	// Initialize grid
	grid_.Init (bbox_, res_, data_res_, cell_size_);
	grid_.SetGridValue (0x00);

	// Launch Multi Slicing Kernel
	dim3 block_dim, grid_dim;
	block_dim = dim3 (128, 4, 4);
	grid_dim = dim3 ((res_[0] / block_dim.x) + 1, (res_[1] / block_dim.y) + 1,
	                 (res_[2] / block_dim.z) + 1);

	time1 = GET_TIME ();
	MultiSlicing <<< ((mesh_gpu_.num_of_faces / block_dim.x) + 1), block_dim.x >>>
	((float *) mesh_gpu_.vertices, mesh_gpu_.num_of_vertices,
	 (unsigned int *) mesh_gpu_.faces, mesh_gpu_.num_of_faces,
	 grid_.grid_gpu ());
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Voxelizer] : " << "rasterize by multislicing by bit computed in "
	          << time2 - time1 << " ms." << std::endl;
	CheckCUDAError ();

	time1 = GET_TIME ();
	//	RobustSimple<<<((mesh_gpu_.num_of_faces/block_dim.x)+1), block_dim.x>>>
	//		((float *) mesh_gpu_.vertices, mesh_gpu_.num_of_vertices,
	//		 (unsigned int *) mesh_gpu_.faces, mesh_gpu_.num_of_faces,
	//		 grid_.grid_gpu ());

	//	RobustSimple<<<((mesh_gpu_.num_of_vertices/block_dim.x)+1), block_dim.x>>>
	//		((float *) mesh_gpu_.vertices, mesh_gpu_.num_of_vertices,
	//		 (unsigned int *) mesh_gpu_.faces, mesh_gpu_.num_of_faces,
	//		 grid_.grid_gpu ());
	//
	//	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Voxelizer] : " << "robust simple by bit computed in "
	          << time2 - time1 << " ms." << std::endl;
	CheckCUDAError ();


	//	TestByMeshing ("input_2x2x2.off");
}

void Voxelizer::ComputeGridAttributes (int base_res, float margin) {
	const vector<Vec3f> & P = mesh_.P();
	bbox_.init (P[0]);
	for (unsigned int i  = 0; i < P.size (); i++)
		bbox_.extendTo (P[i]);

	base_res_ = base_res;
	float thickness = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                       max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                            fabs (bbox_.max()[2] - bbox_.min()[2]))
	                      ) / base_res_;
	bbox_.extend (2 * thickness + margin);
	cell_size_ = max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                  max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                       fabs (bbox_.max()[2] - bbox_.min()[2])
	                      )
	                 ) / base_res_;

	for (unsigned int i = 0; i < 3; i++) {
		res_[i] = base_res_;
	}

	for (unsigned int i = 0; i < 3; i++)
		data_res_[i] = (unsigned int) ceil ((bbox_.max()[i] - bbox_.min()[i]) / cell_size_);
}

void Voxelizer::LoadShaders () {
	// Create the shader programs.
	try {
		surf_vox_conserv_bit_program_ = GL::Program::genVGFProgram ("Conservative Voxelization",
		                                							"Resources/Shaders/surf_vox_conserv.vert",
		                                							"Resources/Shaders/surf_vox_conserv.geo",
		                                							"Resources/Shaders/surf_vox_conserv_bit.frag");

		vol_vox_bit_program_ = GL::Program::genVGFProgram ("Conservative Voxelization",
		                       							   "Resources/Shaders/vol_vox.vert",
		                       							   "Resources/Shaders/vol_vox.geo",
		                       							   "Resources/Shaders/vol_vox_bit.frag");
		surf_vox_conserv_byte_program_ = GL::Program::genVGFProgram ("Conservative Voxelization",
		                                 							 "Resources/Shaders/surf_vox_conserv.vert",
		                                 							 "Resources/Shaders/surf_vox_conserv.geo",
		                                 							 "Resources/Shaders/surf_vox_conserv_byte.frag");
		vol_vox_byte_program_ = GL::Program::genVGFProgram ("Conservative Voxelization",
		                        							"Resources/Shaders/vol_vox.vert",
		                        							"Resources/Shaders/vol_vox.geo",
		                        							"Resources/Shaders/vol_vox_byte.frag");
	} catch (GL::Exception & e) {
		std::cout << e.msg ().c_str () << std::endl;
		exit (EXIT_FAILURE);
	}
}

void Voxelizer::ClearFBO () {
	// 4) Delete the dummy fbo and its attached texture
	glDeleteTextures (1, &vox_fbo_tex_);
	MorphoGraphics::GL::printOpenGLError ("Deleting the dummy texture");
	glDeleteFramebuffers (1, &vox_fbo_);
	MorphoGraphics::GL::printOpenGLError ("Deleting the dummy fbo");

	// 4) Restore a "standard" OpenGL state
	glColorMask (GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	MorphoGraphics::GL::printOpenGLError ("Enabling Back-buffer writing");

	glEnable (GL_CULL_FACE);
	MorphoGraphics::GL::printOpenGLError ("Enabling Face Culling");

	glDisable (GL_TEXTURE_3D);
	MorphoGraphics::GL::printOpenGLError ("Disabling 3D Textures");

	glEnable (GL_DEPTH_TEST);
	MorphoGraphics::GL::printOpenGLError ("Enabling Depth Test");

	glBindFramebuffer (GL_FRAMEBUFFER, 0);
	MorphoGraphics::GL::printOpenGLError ("Binding to backbuffer");
}

void Voxelizer::BuildFBO (int res, int num_bit_vox) {
	int bucket_size = 32;
	int res_x = res, res_y = res, res_z = res / ((bucket_size / num_bit_vox));
	// Some general OpenGL setup :
	// Render Target (FBO + attached texture), viewport, depth test etc.
	glGenFramebuffers (1, &vox_fbo_);
	glBindFramebuffer (GL_FRAMEBUFFER, vox_fbo_);
	MorphoGraphics::GL::printOpenGLError ("Binding to a dummy fbo");

	glEnable (GL_TEXTURE_2D);
	glGenTextures (1, &vox_fbo_tex_);
	glBindTexture (GL_TEXTURE_2D, vox_fbo_tex_);
	MorphoGraphics::GL::printOpenGLError ("Binding dummy texture");

	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	MorphoGraphics::GL::printOpenGLError ("Setting dummy texture Tex Parameters");

	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA32F, res_x, res_y, 0,
	              GL_RGBA, GL_FLOAT, NULL);
	MorphoGraphics::GL::printOpenGLError ("Allocating dummy texture");

	glFramebufferTexture2D (GL_FRAMEBUFFER,
	                        GL_COLOR_ATTACHMENT0,
	                        GL_TEXTURE_2D,
	                        vox_fbo_tex_, 0);
	MorphoGraphics::GL::printOpenGLError ("Attaching dummy texture to dummy fbo");

	glViewport (0, 0, res_x, res_y);
	MorphoGraphics::GL::printOpenGLError ("Setting viewport");

	glDisable (GL_DEPTH_TEST);
	MorphoGraphics::GL::printOpenGLError ("Disabling Depth Test");

	glColorMask (GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	MorphoGraphics::GL::printOpenGLError ("Disabling framebuffer writing");

	// Necessary with the non-conservative rasterizer
	glDisable (GL_CULL_FACE);
	MorphoGraphics::GL::printOpenGLError ("Disabling Face Culling");

	// Create the texture we're going to render to
	glEnable (GL_TEXTURE_3D);
	glGenTextures (1, &vox_tex_);
	MorphoGraphics::GL::printOpenGLError ("Generating voxel texture");

	// "Bind" the newly created texture : all future texture
	// functions will modify this texture
	glBindTexture (GL_TEXTURE_3D, vox_tex_);
	MorphoGraphics::GL::printOpenGLError ("Binding voxel texture");

	// Give an empty image to OpenGL ( the last "0" )
	int tex_level = 0;
	int border = 0;
	glTexImage3D (GL_TEXTURE_3D, tex_level, GL_R32UI, res_x, res_y, res_z,
	              border, GL_RED_INTEGER, GL_UNSIGNED_INT, 0);
	MorphoGraphics::GL::printOpenGLError ("Allocating voxel texture");
	// The two following line are absolutely needed !!!!!
	// But I don't know exactly why :-( ....
	glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	MorphoGraphics::GL::printOpenGLError ("Setting magic Tex Parameter");

	// Clear the texture
	unsigned int vox_tex_clear_value = 0;
	glBindTexture (GL_TEXTURE_3D, vox_tex_);
	MorphoGraphics::GL::printOpenGLError ("Binding voxel texture");
	glClearTexImage (vox_tex_, tex_level, GL_RED_INTEGER, GL_UNSIGNED_INT,
	                 &vox_tex_clear_value);
	MorphoGraphics::GL::printOpenGLError ("Clearing voxel texture");

	// Not sure if this barrier is necessary
	glMemoryBarrier (GL_TEXTURE_UPDATE_BARRIER_BIT);
	MorphoGraphics::GL::printOpenGLError ("Querying Texture Update Barrier");

	// Bind the texture to an image unit
	glBindImageTexture (0, vox_tex_, tex_level,
	                    GL_TRUE,
	                    0,
	                    GL_READ_WRITE,
	                    GL_R32UI
	                   );
	MorphoGraphics::GL::printOpenGLError ("Binding voxel texture to image unit");
}

void Voxelizer::VoxelizeVolume (MorphoGraphics::GL::Program * vol_vox_program) {
	int res = 2 * base_res_;
	float rescale = 2.0 / std::max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                                std::max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                                          fabs (bbox_.max()[2] - bbox_.min()[2])));
	// Timing related query
	GLuint query;
	GLuint64 elapsed_time;
	glGenQueries (1, &query);

	// 1) Use the Volume Voxelization Shader
	vol_vox_program->use ();
	MorphoGraphics::GL::printOpenGLError ("Using Conservative Voxelization Program");

	// 2) Initialization of the shader parameters
	// Vertex and Fragment Shader Common Setup
	glUniform3f (vol_vox_program->getUniformLocation ("bbox_min"),
	             bbox_.min()[0], bbox_.min()[1], bbox_.min()[2]);
	MorphoGraphics::GL::printOpenGLError ("Setting [bbox_min] uniform ");
	glUniform1f (vol_vox_program->getUniformLocation ("rescale"), rescale);
	MorphoGraphics::GL::printOpenGLError ("Setting [rescale] uniform ");
	glUniform1ui (vol_vox_program->getUniformLocation ("res"), res);
	MorphoGraphics::GL::printOpenGLError ("Setting [res] uniform ");

	// Fragment Shader Setup : assign image unit
	glUniform1i (vol_vox_program->getUniformLocation ("vox_grid"), 0);
	MorphoGraphics::GL::printOpenGLError ("Setting [vox_grid] uniform");

	// 3) Single Pass Volume Voxelization
	// Make sure the conservative rasterizer is disabled
	glDisable (GL_CONSERVATIVE_RASTERIZATION_NV);
	MorphoGraphics::GL::printOpenGLError ("Disabling Conservative Rasterizer");

	glBeginQuery (GL_TIME_ELAPSED, query);
	MorphoGraphics::GL::printOpenGLError ("Begining Time Elapsed Query");

	try {
		mesh_.drawVBO ();
	} catch (GL::Exception & e) {
		std::cout << e.msg ().c_str () << std::endl;
		exit (EXIT_FAILURE);
	}

	glMemoryBarrier (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	MorphoGraphics::GL::printOpenGLError ("Querying a Shader Image Access Barrier");

	glEndQuery (GL_TIME_ELAPSED);
	MorphoGraphics::GL::printOpenGLError ("Ending Time Elapsed Query");

	// Retrieving the recorded elapsed time : wait until
	// the query result is available
	int done = 0;
	while (!done)
		glGetQueryObjectiv (query, GL_QUERY_RESULT_AVAILABLE, &done);

	// Get the query result for elapsed time
	glGetQueryObjectui64v (query, GL_QUERY_RESULT, &elapsed_time);
	std::cout << "[Voxelizer] : " << "conservative volume voxelization done in "
	          << elapsed_time / 1000000.0 << " ms." << std::endl;
}

void Voxelizer::VoxelizeSurfaceConservative (GL::Program * surf_vox_conserv_program) {
	int res = 2 * base_res_;
	float rescale = 2.0 / std::max (fabs (bbox_.max()[0] - bbox_.min()[0]),
	                                std::max (fabs (bbox_.max()[1] - bbox_.min()[1]),
	                                        fabs (bbox_.max()[2] - bbox_.min()[2])));
	// Timing related query
	GLuint query;
	GLuint64 elapsed_time;
	glGenQueries (1, &query);

	// 1) Use the Surface Conservative Voxelization Shader
	surf_vox_conserv_program->use ();
	MorphoGraphics::GL::printOpenGLError ("Using Conservative Voxelization Program");

	// 2) Initialization of the shader parameters
	// Vertex and Fragment Shader Common Setup
	glUniform3f (surf_vox_conserv_program->getUniformLocation ("bbox_min"),
	             bbox_.min()[0], bbox_.min()[1], bbox_.min()[2]);
	MorphoGraphics::GL::printOpenGLError ("Setting [bbox_min] uniform ");
	glUniform1f (surf_vox_conserv_program->getUniformLocation ("rescale"), rescale);
	MorphoGraphics::GL::printOpenGLError ("Setting [scale] uniform ");
	glUniform1ui (surf_vox_conserv_program->getUniformLocation ("res"), res);
	MorphoGraphics::GL::printOpenGLError ("Setting [res] uniform ");

	// Geometry Shader Setup
	glUniform1f (surf_vox_conserv_program->getUniformLocation ("sq_ar_thresh"), 50.f * 50.f);
	MorphoGraphics::GL::printOpenGLError ("Setting [sq_ar_thresh] uniform ");

	// Fragment Shader Setup : assign image unit
	glUniform1i (surf_vox_conserv_program->getUniformLocation ("vox_grid"), 0);
	MorphoGraphics::GL::printOpenGLError ("Setting [vox_grid] uniform");

	// 3) Single Pass Surface Conservative Voxelization
	// Enable the conservative rasterizer
	glEnable (GL_CONSERVATIVE_RASTERIZATION_NV);
	MorphoGraphics::GL::printOpenGLError ("Enabling Conservative Rasterizer");

	glBeginQuery (GL_TIME_ELAPSED, query);
	MorphoGraphics::GL::printOpenGLError ("Begining Time Elapsed Query");

	try {
		mesh_.drawVBO ();
	} catch (GL::Exception & e) {
		std::cout << e.msg ().c_str () << std::endl;
		exit (EXIT_FAILURE);
	}

	glMemoryBarrier (GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	MorphoGraphics::GL::printOpenGLError ("Querying a Shader Image Access Barrier");

	glEndQuery (GL_TIME_ELAPSED);
	MorphoGraphics::GL::printOpenGLError ("Ending Time Elapsed Query");

	// Disable the conservative rasterizer
	glDisable (GL_CONSERVATIVE_RASTERIZATION_NV);
	MorphoGraphics::GL::printOpenGLError ("Disabling Conservative Rasterizer");

	// Retrieving the recorded elapsed time : wait until
	// the query result is available
	int done = 0;
	while (!done)
		glGetQueryObjectiv (query, GL_QUERY_RESULT_AVAILABLE, &done);

	// Get the query result for elapsed time
	glGetQueryObjectui64v (query, GL_QUERY_RESULT, &elapsed_time);
	std::cout << "[Voxelizer] : " << "conservative surface voxelization done in "
	          << elapsed_time / 1000000.0 << " ms." << std::endl;
}

void Voxelizer::VoxelizeConservative (int base_res, float margin) {
	// Important note: an OpenGL context is needed
	// to call this function. We first flush the OpenGL
	// error pipeline.
	MorphoGraphics::GL::printOpenGLError ("Error before any OpenGL usage");

	// This voxelization algorithm use the OpenGL pipeline to compute a
	// conservative voxelization of an input mesh. It use the conservative
	// rasterization capability of Maxwell and beyond NVIDIA GPUs.
	//
	// In this method, a voxel grid is encoded as an XY array of Z-Buckets.
	// G[X, Y] = Z-Buckets = [z-bucket[0], ... , z-bucket[N]]
	// And a z-bucket is an 'unsigned int' encoding 32 (resp. 4) z-voxels.
	// As such a voxel is encoded by 1 (resp. 8) bit(s).
	//
	// Important note: this voxel grid layout is not the same as the one
	// by CUDA voxelizer (which is not conservative) and the morphological
	// operators. A subsequent step of conversion is needed to match
	// the specific packed layout of the CUDA voxelizer and
	// the morphological operators.
	// A packed layout is made of a grid of unsigned char or bytes
	// that represent 8 packed voxels arranged in a 8 pieces cube.

	// 1) Compute Grid Attributes
	ComputeGridAttributes (base_res, margin);

	// 2) Push the mesh into GPU Memory using a VBO
	mesh_.initVBO ();

	int res = 2 * base_res_; // the base_res_ is the resolution of the packed layout
	int num_bit_vox = 1; // number of bits in one bucket

	LoadShaders ();
	BuildFBO (res, num_bit_vox);
	if (num_bit_vox == 1) {
		VoxelizeVolume (vol_vox_bit_program_);
		VoxelizeSurfaceConservative (surf_vox_conserv_bit_program_);
	} else {
		VoxelizeVolume (vol_vox_byte_program_);
		VoxelizeSurfaceConservative (surf_vox_conserv_byte_program_);
	}
	ClearFBO ();

	// Retrieve the data
	int res_x = res, res_y = res, res_z = res / ((32 / num_bit_vox));
	std::vector<unsigned int> buck_vox_grid (res_x * res_y * res_z, 0);
	glBindTexture (GL_TEXTURE_3D, vox_tex_);
	MorphoGraphics::GL::printOpenGLError ("Binding voxel texture");
	int get_tex_level = 0;
	glGetTexImage (GL_TEXTURE_3D, get_tex_level,
	               GL_RED_INTEGER,
	               GL_UNSIGNED_INT,
	               &buck_vox_grid[0]);
	MorphoGraphics::GL::printOpenGLError ("Retrieving Data");

	std::vector<unsigned char> pack_vox_grid;
	// Set up a GridGPU structure
	ConvertBucketToPacketGrid (buck_vox_grid, res, num_bit_vox, pack_vox_grid);
	grid_.Init (bbox_, res_, data_res_, cell_size_);

	cudaMemcpy (grid_.grid_gpu ().voxels, &pack_vox_grid[0],
	            pack_vox_grid.size ()*sizeof (unsigned char),
	            cudaMemcpyHostToDevice);
	CheckCUDAError ();
	std::cout << "[Voxelizer] : " << "GridGPU structure set up done." << std::endl;

	//TestByMeshing ("gl_vox_grid_mesh.off", grid_.grid_gpu ());

	//	ParseGrid (buck_vox_grid, bbox_, res, num_bit_vox, positions, normals);
	//	SaveCubeSet ("vox_grid_cubes.ply", positions, bbox_, res);
	//	std::cout << "[Voxelizer] : number of non empty cells : "
	//		<< positions.size () << std::endl;
}

void Voxelizer::ConvertBucketToPacketGrid (const std::vector<unsigned int> & buck_vox_grid,
        								   int res, int num_bit_vox,
        								   std::vector<unsigned char> & pack_vox_grid) {
	int pack_res_x = res / 2, pack_res_y = res / 2, pack_res_z = res / 2;
	pack_vox_grid.resize (pack_res_x * pack_res_y * pack_res_z, 0);

	int res_x = res, res_y = res, res_z = res / ((32 / num_bit_vox));
	for (int i = 0; i < res_x; i++)
		for (int j = 0; j < res_y; j++)
			for (int k = 0; k < res_z; k++) {
				unsigned int value = buck_vox_grid[i + res_x * j + res_x * res_y * k];
				if (value != 0)	{
					if (num_bit_vox == 1)
						for (int s = 0; s < 32; s++) {
							if ((value & (0x00000001)) == 0x00000001) {
								Vec3i position (i, j, 32 * k + s);
								Vec3i pack_position (position[0] / 2,
								                     position[1] / 2,
								                     position[2] / 2);
								unsigned char num_pack_shift = position[0] % 2
								                               + 2 * (position[1] % 2)
								                               + 4 * (position[2] % 2);
								unsigned char pack_val = 1 << num_pack_shift;
								unsigned int pack_idx = pack_position[0]
								                        + pack_res_x * pack_position[1]
								                        + pack_res_x * pack_res_y * pack_position[2];
								pack_vox_grid[pack_idx] |= pack_val;
							}
							value /= 2; // 2 -> 2^(1 bit shifts)
						}

					if (num_bit_vox == 8)
						for (int s = 0; s < 4; s++) {
							if ((value & (0x00000001)) == 0x00000001) {
								Vec3i position (i, j, 4 * k + s);
								Vec3i pack_position (position[0] / 2,
								                     position[1] / 2,
								                     position[2] / 2);
								unsigned char num_pack_shift = position[0] % 2
								                               + 2 * (position[1] % 2)
								                               + 4 * (position[2] % 2);
								unsigned char pack_val = 1 << num_pack_shift;
								unsigned int pack_idx = pack_position[0]
								                        + pack_res_x * pack_position[1]
								                        + pack_res_x * pack_res_y * pack_position[2];
								pack_vox_grid[pack_idx] |= pack_val;
							}
							value /= 256; // 256 -> 2^(8 bit shifts)
						}
				}
			}
}

void Voxelizer::ParseGrid (const std::vector<unsigned int> & vox_grid,
                           const AxisAlignedBoundingBox & bbox,
                           int res, int num_bit_vox,
                           std::vector<Vec3f> & positions,
                           std::vector<Vec3f> & normals) {
	int res_x = res, res_y = res, res_z = res / ((32 / num_bit_vox));
	float max_bbox_length = std::max (fabs (bbox.max()[0] - bbox.min()[0]),
	                                  std::max (fabs (bbox.max()[1] - bbox.min()[1]),
	                                          fabs (bbox.max()[2] - bbox.min()[2])
	                                           )
	                                 );

	float voxel_size = max_bbox_length / ((float)res);
	for (int i = 0; i < res_x; i++)
		for (int j = 0; j < res_y; j++)
			for (int k = 0; k < res_z; k++) {
				unsigned int value = vox_grid[i + res_x * j + res_x * res_y * k];
				if (value != 0)	{
					if (num_bit_vox == 1)
						for (int s = 0; s < 32; s++) {
							if ((value & (0x00000001)) == 0x00000001) {
								Vec3f position (((float)i)*voxel_size,
								                ((float)j)*voxel_size,
								                ((float)(32 * k + s))*voxel_size
								               );
								position = position + bbox.min ()
								           + 0.5f * Vec3f (voxel_size, voxel_size, voxel_size);
								positions.push_back (position);
								normals.push_back (Vec3f (1, 0, 0));
							}
							value /= 2; // 2 -> 2^(1 bit shifts)
						}

					if (num_bit_vox == 8)
						for (int s = 0; s < 4; s++) {
							if ((value & (0x00000001)) == 0x00000001) {
								Vec3f pos (((float)i)*voxel_size,
								           ((float)j)*voxel_size,
								           ((float)(4 * k + s))*voxel_size
								          );
								pos += bbox.min ()+ 0.5f * Vec3f (voxel_size, voxel_size, voxel_size);
								positions.push_back (pos);
								normals.push_back (Vec3f (1, 0, 0));
							}
							value /= 256; // 256 -> 2^(8 bit shifts)
						}
				}
			}
}
__global__ 
void ConvertVoxelGridByBitVoxelizer (unsigned char * voxelGrid, uint3 res, float * voxelGrid2x2x2) {
	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z;

	if (idX >= (2 * res.x - 1) || idY >= (2 * res.y - 1) || idZ >= (2 * res.z - 1))
		return;

	unsigned char val;
	val = voxelGrid[(idX / 2) + res.x * (idY / 2) + res.x * res.y * (idZ / 2)] & (1 << (idX % 2 + (idY % 2) * 2 + (idZ % 2) * 4));
	//	val = voxelGrid[(idX/2) + res.x*(idY/2) + res.x*res.y*(idZ/2)];

	voxelGrid2x2x2[idX + 2 * res.x * idY + 4 * res.x * res.y * idZ] = val ? 0.f : 1.f;
}
/*
void Voxelizer::TestByMeshing (const std::string & filename) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int num_cells_2x2x2 = 2 * res_[0] * 2 * res_[1] * 2 * res_[2];

	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 (((2 * res_[0]) / block_dim.x) + 1, ((2 * res_[1]) / block_dim.y) + 1,
	                 ((2 * res_[2]) / block_dim.z) + 1);

	float * voxels_2x2x2 = NULL;
	cudaMalloc (&voxels_2x2x2, num_cells_2x2x2 * sizeof (float));
	CheckCUDAError ();

	time1 = GET_TIME ();
	ConvertVoxelGridByBitVoxelizer <<< grid_dim, block_dim>>>
	(grid_.grid_gpu ().voxels,
	 grid_.grid_gpu ().res,
	 voxels_2x2x2);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	std::cout << "[Convert Voxel Grid] "
	          << "convert by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	Vec3f bboxMCM = bbox_.min;

	MarchingCubesMesher::Grid grid_mcm (bbox_.min, 0.5f * cell_size_, 0.5f * cell_size_, 0.5f * cell_size_,
	                                    2 * res_[0], 2 * res_[1], 2 * res_[2]);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (voxels_2x2x2, 0.5f, 0.5 * NAN_EVAL);
	mesher.saveMesh (filename.c_str ());
	time2 = GET_TIME ();

	FreeGPUResource (&voxels_2x2x2);
}

void Voxelizer::TestByMeshing (const std::string & filename,
                               const GridGPU & grid_gpu) {
	double time1, time2;
	dim3 block_dim, grid_dim;
	unsigned int num_cells_2x2x2 = 2 * grid_gpu.res.x * 2 * grid_gpu.res.y * 2 * grid_gpu.res.z;

	block_dim = dim3 (4, 4, 4);
	grid_dim = dim3 (((2 * grid_gpu.res.x) / block_dim.x) + 1,
	                 ((2 * grid_gpu.res.y) / block_dim.y) + 1,
	                 ((2 * grid_gpu.res.z) / block_dim.z) + 1);

	float * voxels_2x2x2 = NULL;
	cudaMalloc (&voxels_2x2x2, num_cells_2x2x2 * sizeof (float));
	CheckCUDAError ();
	std::cout << "[Voxelizer] : " << "TestByMeshing : voxels_2x2x2 allocated"
	          << " grid_gpu.res : "
	          << grid_gpu.res.x << ", " << grid_gpu.res.y << ", " << grid_gpu.res.z
	          << std::endl;

	time1 = GET_TIME ();
	ConvertVoxelGridByBitVoxelizer <<< grid_dim, block_dim>>>
	(grid_gpu.voxels,
	 grid_gpu.res,
	 voxels_2x2x2);
	cudaDeviceSynchronize ();
	CheckCUDAError ();
	time2 = GET_TIME ();
	std::cout << "[Convert Voxel Grid] "
	          << "convert by bit computed in "
	          << time2 - time1 << " ms." << std::endl;

	Vec3f bbox_min_mcm = Vec3f (grid_gpu.bbox_min.x,
	                            grid_gpu.bbox_min.y,
	                            grid_gpu.bbox_min.z);

	MarchingCubesMesher::Grid grid_mcm (bbox_min_mcm,
	                                    0.5f * grid_gpu.cell_size,
	                                    0.5f * grid_gpu.cell_size,
	                                    0.5f * grid_gpu.cell_size,
	                                    2 * grid_gpu.res.x,
	                                    2 * grid_gpu.res.y,
	                                    2 * grid_gpu.res.z);
	MarchingCubesMesher mesher (&grid_mcm);

	time1 = GET_TIME ();
	mesher.createMesh3D (voxels_2x2x2, 0.5f, 0.5 * NAN_EVAL);
	mesher.saveMesh (filename.c_str ());
	time2 = GET_TIME ();

	FreeGPUResource (&voxels_2x2x2);
}
*/