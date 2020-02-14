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

#ifndef MARCHINGCUBES_H
#define MARCHINGCUBES_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <cstdio>
#include <cassert>

#include <Common/Vec3.h>

using namespace std;
using namespace MorphoGraphics;

#define MC0 1
#define MC1 2
#define MC2 4
#define MC3 8
#define MC4 16
#define MC5 32
#define MC6 64
#define MC7 128

class MarchingCubesMesher {
public:
	class Exception {
	public:
		inline Exception (const std::string & msg) : _msg ("MarchingCubesMesher Error: " + msg) {}
		inline const std::string & msg () const { return _msg; }
	protected:
		std::string _msg;
	};

	class Grid {
	private:
		Vec3f origin;
		float cellSizeX;
		float cellSizeY;
		float cellSizeZ;
		int nbCellsX;
		int nbCellsY;
		int nbCellsZ;

	public:
		//constructors
		Grid(Vec3f orig, float cSizeX, float cSizeY, float cSizeZ, int nbCX, int nbCY, int nbCZ)
			: origin(orig), cellSizeX(cSizeX), cellSizeY(cSizeY), cellSizeZ(cSizeZ), nbCellsX(nbCX), nbCellsY(nbCY), nbCellsZ(nbCZ) {}
		Grid() : cellSizeX(1), cellSizeY(1), cellSizeZ(1), nbCellsX(50), nbCellsY(50), nbCellsZ(50) {origin = Vec3f(0, 0, 0);}
		Grid(Vec3f orig, float cSize, int nbC)
			: origin(orig), cellSizeX(cSize), cellSizeY(cSize), cellSizeZ(cSize), nbCellsX(nbC), nbCellsY(nbC), nbCellsZ(nbC) {}

		//accessors
		Vec3f& getOrigin() {return origin;}

		float getCellSizeX() {return cellSizeX;}
		float getCellSizeY() {return cellSizeY;}
		float getCellSizeZ() {return cellSizeZ;}

		int getNbCellsX() {return nbCellsX;}
		int getNbCellsY() {return nbCellsY;}
		int getNbCellsZ() {return nbCellsZ;}
	};

	//Mesh

	class Mesh {
	public:
		float *V;
		float *N;
		unsigned int VSize;
		unsigned int *T;
		unsigned int TSize;

		float3 *devV;
		float3 *devN;
		uint3 *devT;

		void saveOFF (const string & filename);
		~Mesh() {delete V; delete T;}
	};

	MarchingCubesMesher(Grid* g) : grid(g) {mesh = new Mesh();}
	~MarchingCubesMesher() {delete mesh;}

	void saveMesh(string filename) {mesh->saveOFF(filename);}
	void createMesh(float * evalValuesGPU,
	                float isovalue = 0, float invalidIsovalue = 1e20f);
	void createMesh3D (unsigned char * evalValuesGPU,
	                   unsigned char isovalue = 0, float invalidIsovalue = 1e20f);
	void createMesh3D (float * evalValuesGPU,
	                   float isovalue = 0, float invalidIsovalue = 1e20f);
	void createMesh3DByBit (unsigned char * evalValuesGPU,
	                        unsigned char isovalue = 0, float invalidIsovalue = 1e20f);
	void createMesh3D (unsigned char * contour_values,
	                   unsigned int * contour_indices,
	                   unsigned int * contour_neigh_indices,
	                   unsigned int * contour_neigh_morpho_centroids,
	                   unsigned int contour_size,
	                   float isovalue = 0, float invalidIsovalue = 1e20f);
	void getMesh (float ** ptrV, unsigned int * ptrVSize,
	              unsigned int ** ptrT, unsigned int * ptrTSize) {
		*ptrV	= mesh->V;
		*ptrVSize	= mesh->VSize;
		*ptrT	= mesh->T;
		*ptrTSize	= mesh->TSize;
	}

	void getMesh (float ** ptrV, float ** ptrN, unsigned int * ptrVSize,
	              unsigned int ** ptrT, unsigned int * ptrTSize) {
		*ptrV	= mesh->V;
		*ptrN	= mesh->N;
		*ptrVSize	= mesh->VSize;
		*ptrT	= mesh->T;
		*ptrTSize	= mesh->TSize;
	}

	void getDeviceMesh (float3 ** ptrV, float3 ** ptrN, unsigned int * ptrVSize,
	                    uint3 ** ptrT, unsigned int * ptrTSize) {
		*ptrV	= mesh->devV;
		*ptrN	= mesh->devN;
		*ptrVSize	= mesh->VSize;
		*ptrT	= mesh->devT;
		*ptrTSize	= mesh->TSize;
	}

	void setDeviceMesh (float3 * V, float3 * N, unsigned int VSize,
	                    uint3 * T, unsigned int TSize) {
		mesh->devV = V;
		mesh->devN = N;
		mesh->VSize = VSize;
		mesh->devT = T;
		mesh->TSize = TSize;
	}

	static void print (const std::string & msg);
	void showGPUMemoryUsage ();
private :
	Grid* grid;
	Mesh* mesh;
	void checkCUDAError ();
};






#endif
