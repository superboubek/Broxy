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

#include "MarchingCubesMesher.h"
#include "tables.h"

#include <cmath>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>


#include "timing.h"
#include "cuda_math.h"

using namespace std;

void MarchingCubesMesher::showGPUMemoryUsage () {
	size_t avail, total, used;
	cudaMemGetInfo(&avail, &total);
	used = total - avail;
	std::cout << "[MarchingCubesMesher] : Device memory used: " << (float)used/1e6 
		<< " of " << (float)total/1e6 << std::endl;
}
void MarchingCubesMesher::print (const std::string & msg) {
	std::cout << "[MarchingCubesMesher] : " << msg << std::endl;
}

void MarchingCubesMesher::checkCUDAError () {
	cudaError_t err = cudaGetLastError ();
	if(err != cudaSuccess) {
		MarchingCubesMesher::print ("CUDA Error : " + std::string (cudaGetErrorString (err)));
		throw MarchingCubesMesher::Exception ("CUDA Error: " + std::string (cudaGetErrorString (err)));
	}
}


//fonctions device

__device__ float eval(float x, float y, float z){
	float phi = 1.6f;
	float b = 0.5f;
	float value = 4.f*(phi * phi * x*x - y*y)*(phi*phi * y*y - z*z)*(phi*phi * z*z - x*x) - (b + 2.f*phi)*(x*x + y*y + z*z - b*b)*(x*x + y*y + z*z - b*b)*b*b;
	return (value);
}


//#ifndef NO_SEPARATE_COMPILATION
//inline __host__ __device__ float3 operator-( float3 a, float3 b){
//	return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
//}
//
//inline __host__ __device__ float3 operator+( float3 a, float3 b){
//	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
//}
//
//inline __host__ __device__ float3 operator*( float3 a, float3 b){
//	return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
//}
//
//inline __host__ __device__ float3 operator/( float3 a, float b){
//	return make_float3(a.x/b, a.y/b, a.z/b);
//}
//
//inline __host__ __device__ float3 operator*( float a, float3 b){
//	return make_float3(a*b.x, a*b.y, a*b.z);
//}
//#endif



__device__ float3 findIntersection(float3 p1, float v1, float3 p2, float v2, float isovalue){
	return (p1 + (isovalue - v1) * (p2 - p1) / (v2 - v1));
}

//kernels

//functionValues doit être de taille (nbX + 2) * (nbY + 2) * (nbZ + 2)
__global__ void evalGrid(float *functionValues, unsigned int nbX, unsigned int nbY, unsigned int nbZ, float ox, float oy, float oz, float dx, float dy, float dz ){
	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(i<(nbX+2)*(nbY+2)*(nbZ+2)){
		unsigned int x = i/((nbY+2) * (nbZ+2));
		unsigned int y = (i - x * (nbY+2) * (nbZ+2)) / (nbZ+2);
		unsigned int z = i - x * (nbY+2) * (nbZ+2) - y * (nbZ+2);
		functionValues[i] = eval(ox + x * dx, oy + y * dy, oz + z * dz);
	}
}

//index de taille (nbX+1)*(nbY+1)*(nbZ+1)
__global__ void computeIndex (float *functionValues, unsigned char *index, 
															unsigned int *nonEmptyCube, 
															unsigned int nbX, unsigned int nbY, unsigned int nbZ, 
															float isovalue, float invalidIsovalue){
	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(i<(nbX+1)*(nbY+1)*(nbZ+1)){
		unsigned int x = i/((nbY+1) * (nbZ+1));
		unsigned int y = (i - x * (nbY+1) * (nbZ+1)) / (nbZ+1);
		unsigned int z = i - x * (nbY+1) * (nbZ+1) - y * (nbZ+1);

		unsigned char ind = 0;

		float v0, v1, v2, v3, v4, v5, v6, v7;
		v0 = functionValues[(nbY + 2) * (nbZ + 2) * x + (nbZ + 2) * y + z];
		v1 = functionValues[(nbY + 2) * (nbZ + 2) * (x + 1) + (nbZ + 2) * y + z];
		v2 = functionValues[(nbY + 2) * (nbZ + 2) * (x + 1) + (nbZ + 2) * (y + 1) + z];
		v3 = functionValues[(nbY + 2) * (nbZ + 2) * x + (nbZ + 2) * (y + 1) + z];
		v4 = functionValues[(nbY + 2) * (nbZ + 2) * x + (nbZ + 2) * y + z + 1];
		v5 = functionValues[(nbY + 2) * (nbZ + 2) * (x + 1) + (nbZ + 2) * y + z + 1];
		v6 = functionValues[(nbY + 2) * (nbZ + 2) * (x + 1) + (nbZ + 2) * (y + 1) + z + 1];
		v7 = functionValues[(nbY + 2) * (nbZ + 2) * x + (nbZ + 2) * (y + 1) + z + 1];

		if(v0 >= isovalue) ind |= 1;
		if(v1 >= isovalue) ind |= 2;
		if(v2 >= isovalue) ind |= 4;
		if(v3 >= isovalue) ind |= 8;
		if(v4 >= isovalue) ind |= 16;
		if(v5 >= isovalue) ind |= 32;
		if(v6 >= isovalue) ind |= 64;
		if(v7 >= isovalue) ind |= 128;

		index[i] = ind;

		nonEmptyCube[i] = (ind != 0 && ind != 255) ? 1 : 0;
	}
}

__global__ void compactCubes(unsigned char *index, unsigned int *nonEmptyCubeScan, unsigned int *compactNonEmptyCubeScan, unsigned int nbX, unsigned int nbY, unsigned int nbZ){
	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(i<(nbX+1)*(nbY+1)*(nbZ+1)){
		if(index[i]!=0 && index[i]!=255){
			compactNonEmptyCubeScan[nonEmptyCubeScan[i]] = i;
		}
	}
}

__global__ void countTrianglesAndVertices(unsigned int *nbVerticesCube, unsigned int *nbTrianglesCube, unsigned int *compactNonEmptyCubeScan, unsigned char *index, unsigned int nbNonEmptyCubes, unsigned int nbX, unsigned int nbY, unsigned int nbZ){
	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(i<nbNonEmptyCubes){		
		unsigned int indice = compactNonEmptyCubeScan[i];
		unsigned int x = indice/((nbY+1) * (nbZ+1));
		unsigned int y = (indice - x * (nbY+1) * (nbZ+1)) / (nbZ+1);
		unsigned int z = indice - x * (nbY+1) * (nbZ+1) - y * (nbZ+1);

		int nbVerts = 0;
		if ((edgeTable[index[indice]] & 1) && (x != nbX)) nbVerts++; 
		if ((edgeTable[index[indice]] & 8) && (y != nbY)) nbVerts++; 
		if ((edgeTable[index[indice]] & 256) && (z != nbZ)) nbVerts++;
		nbVerticesCube[i] = nbVerts;

		nbTrianglesCube[i] = (x != nbX && y != nbY && z != nbZ) ? numTriTable[index[indice]] : 0;
	}
}

__global__ void createTrianglesAndVertices (float3 *vertices, uint3 *triangles, float *functionValues, 
																						unsigned int *nbVerticesCube, unsigned int *nbTrianglesCube, 
																						unsigned int *compactNonEmptyCubeScan, unsigned int *nonEmptyCubeScan, 
																						unsigned char *index, unsigned int nbNonEmptyCubes, 
																						unsigned int nbX, unsigned int nbY, unsigned int nbZ, float ox, float oy, float oz, float dx, float dy, float dz, 
																						float isovalue, float invalidIsovalue) {
	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if(i<nbNonEmptyCubes){
		unsigned int ind = compactNonEmptyCubeScan[i];
		unsigned int x = ind/((nbY+1) * (nbZ+1));
		unsigned int y = (ind - x * (nbY+1) * (nbZ+1)) / (nbZ+1);
		unsigned int z = ind - x * (nbY+1) * (nbZ+1) - y * (nbZ+1);

		unsigned char indexCube = index[ind];

		unsigned int vert[12];
		float3 point = make_float3(ox + x * dx, oy + y * dy, oz + z * dz);
		unsigned int indice = 0;
		short edge = edgeTable[indexCube];

		float v0, v1, v2, v3, v4, v5, v6, v7;
		//		unsigned int id0, id1, id2, id3, id4, id5, id6, id7;
		unsigned int id1, id2, id3, id4, id5, id7;
		//		id0 = (nbY + 1)*(nbZ + 1)*x 			+ (nbZ + 1)*y 			+ z;
		id1 = (nbY + 1)*(nbZ + 1)*(x + 1)	+ (nbZ + 1)*y				+ z;
		id2 = (nbY + 1)*(nbZ + 1)*(x + 1)	+ (nbZ + 1)*(y + 1)	+ z;
		id3 = (nbY + 1)*(nbZ + 1)*x				+ (nbZ + 1)*(y + 1)	+ z;
		id4 = (nbY + 1)*(nbZ + 1)*x				+ (nbZ + 1)*y				+ z + 1;
		id5 = (nbY + 1)*(nbZ + 1)*(x + 1)	+ (nbZ + 1)*y				+ z + 1;
		//		id6 = (nbY + 1)*(nbZ + 1)*(x + 1)	+ (nbZ + 1)*(y + 1)	+ z + 1;
		id7 = (nbY + 1)*(nbZ + 1)*x				+ (nbZ + 1)*(y + 1)	+ z + 1;
		v0 = functionValues[(nbY + 2)*(nbZ + 2)*x				+ (nbZ + 2)*y				+ z];
		v1 = functionValues[(nbY + 2)*(nbZ + 2)*(x + 1)	+ (nbZ + 2)*y				+ z];
		v2 = functionValues[(nbY + 2)*(nbZ + 2)*(x + 1)	+ (nbZ + 2)*(y + 1)	+ z];
		v3 = functionValues[(nbY + 2)*(nbZ + 2)*x				+ (nbZ + 2)*(y + 1)	+ z];
		v4 = functionValues[(nbY + 2)*(nbZ + 2)*x				+ (nbZ + 2)*y				+ z + 1];
		v5 = functionValues[(nbY + 2)*(nbZ + 2)*(x + 1)	+ (nbZ + 2)*y				+ z + 1];
		v6 = functionValues[(nbY + 2)*(nbZ + 2)*(x + 1)	+ (nbZ + 2)*(y + 1)	+ z + 1];
		v7 = functionValues[(nbY + 2)*(nbZ + 2)*x				+ (nbZ + 2)*(y + 1)	+ z + 1];

		bool valid = false;
		//		if (fabs (v0) > invalidIsovalue 
		//				|| fabs (v1) > invalidIsovalue 
		//				|| fabs (v2) > invalidIsovalue 
		//				|| fabs (v3) > invalidIsovalue
		//				|| fabs (v4) > invalidIsovalue
		//				|| fabs (v5) > invalidIsovalue 
		//				|| fabs (v6) > invalidIsovalue 
		//				|| fabs (v7) > invalidIsovalue
		//				|| isnan (v0) || isnan (v1) || isnan (v2) || isnan (v3)
		//				|| isnan (v4) || isnan (v5) || isnan (v6) || isnan (v7) 	
		//			 ) valid = false;
		//
		if ((edge & 1) && (x != nbX)){
			valid = valid || (fabs (v0) > invalidIsovalue) || (fabs (v1) > invalidIsovalue);
			vert[0] = nbVerticesCube[i] + indice;
			vertices[vert[0]] = findIntersection(point, v0, 
																					 point + make_float3(dx, 0, 0), v1, 
																					 isovalue);
			indice ++;

		}
		if ((edge & 2) && (x != nbX) && (y != nbY)){
			valid = valid || (fabs (v1) > invalidIsovalue) || (fabs (v2) > invalidIsovalue);
			vert[1] = nbVerticesCube[nonEmptyCubeScan[id1]];			
			if(edgeTable[index[id1]]&1 && (x+1)!=nbX){
				vert[1] ++;
			}
		}
		if ((edge & 4) && (x != nbX) && (y != nbY)){
			valid |= (fabs (v2) > invalidIsovalue) || (fabs (v3) > invalidIsovalue);
			vert[2] = nbVerticesCube[nonEmptyCubeScan[id3]];
		}
		if ((edge & 8) && (y != nbY)){
			valid = valid || (fabs (v3) > invalidIsovalue) || (fabs (v0) > invalidIsovalue);
			vert[3] = nbVerticesCube[i] + indice;
			vertices[vert[3]] = findIntersection(point + make_float3(0, dy, 0), v3, 
																					 point, v0, 
																					 isovalue);
			indice ++;
		}
		if ((edge & 16) && (z != nbZ) && (x != nbX)){
			valid = valid || (fabs (v4) > invalidIsovalue) || (fabs (v5) > invalidIsovalue);
			vert[4] = nbVerticesCube[nonEmptyCubeScan[id4]];
		}

		if ((edge & 32) && (x != nbX) && (y != nbY) && (z != nbZ)){
			valid = valid || (fabs (v5) > invalidIsovalue) || (fabs (v6) > invalidIsovalue);
			vert[5] = nbVerticesCube[nonEmptyCubeScan[id5]];
			if(edgeTable[index[id5]]&1 && (x+1)!=nbX){
				vert[5] ++;
			}
		}
		if ((edge & 64) && (x != nbX) && (y != nbY) && (z != nbZ)){
			valid = valid || (fabs (v6) > invalidIsovalue) || (fabs (v7) > invalidIsovalue);
			vert[6] = nbVerticesCube[nonEmptyCubeScan[id7]];
		}
		if ((edge & 128) && (y != nbY) && (z != nbZ)){
			valid = valid || (fabs (v7) > invalidIsovalue) || (fabs (v4) > invalidIsovalue);
			vert[7] = nbVerticesCube[nonEmptyCubeScan[id4]];
			if(edgeTable[index[id4]]&1 && x!=nbX){
				vert[7] ++;
			}
		}		
		if ((edge & 256) && (z != nbZ)){
			valid = valid || (fabs (v0) > invalidIsovalue) || (fabs (v4) > invalidIsovalue);
			vert[8] = nbVerticesCube[i] + indice;
			vertices[vert[8]] = findIntersection(point , v0, 
																					 point + make_float3(0, 0, dz), v4, 
																					 isovalue);
			indice ++;
		}
		unsigned int d;
		short edge1;
		if ((edge & 512) && (x != nbX) && (z != nbZ)){
			valid = valid || (fabs (v1) > invalidIsovalue) || (fabs (v5) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id1]];
			if(edge1 & 1 && (x+1)!=nbX) d++;
			if(edge1 & 8 && (y!=nbY)) d++; 
			vert[9] = nbVerticesCube[nonEmptyCubeScan[id1]] + d;
		}
		if ((edge & 1024) && (x != nbX) && (y != nbY) && (z != nbZ)){
			valid = valid || (fabs (v2) > invalidIsovalue) || (fabs (v6) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id2]];
			if(edge1 & 1 && (x+1)!=nbX) d++;
			if(edge1 & 8 && (y+1)!=nbY) d++;
			vert[10] = nbVerticesCube[nonEmptyCubeScan[id2]] + d;
		}
		if ((edge & 2048) && (y != nbY) && (z != nbZ)){
			valid = valid || (fabs (v3) > invalidIsovalue) || (fabs (v7) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id3]];
			if(edge1 & 1 && x!=nbX) d++;
			if(edge1 & 8 && (y+1)!=nbY) d++;
			vert[11] = nbVerticesCube[nonEmptyCubeScan[id3]] + d;
		}

		unsigned int tri = 0;	
		if (indexCube != 0 && indexCube != 255 && x != nbX && y != nbY && z != nbZ){
			unsigned int nbTri = nbTrianglesCube[i];
			for (int s = 0; triTable[indexCube * 16 + s] != -1; s+=3){
				if (!valid)
					triangles[nbTri + tri] = make_uint3 (vert[triTable[indexCube * 16 + s + 2]], 
																							 vert[triTable[indexCube * 16 + s + 1]], 
																							 vert[triTable[indexCube * 16 + s]]);
				else
					triangles[nbTri + tri] = make_uint3 (UINT32_MAX, UINT32_MAX, UINT32_MAX);
				tri++;
			}
		}
	}
}

//fonctions host

void MarchingCubesMesher::Mesh::saveOFF (const string & filename) {
	ofstream out (filename.c_str ());
	if (!out) 
		exit (EXIT_FAILURE);
	out << "OFF" << endl;
	out << VSize << " " << TSize << " 0" << endl;
	for (unsigned int i = 0; i < VSize; i++)
		out << V[i*3] << " " << V[i*3 + 1] << " " << V[i*3 +2] << endl;
	for (unsigned int i = 0; i < TSize; i++)
		out << "3 " << T[i*3] << " " << T[i*3 + 1] << " " << T[i*3 + 2] << endl;
	out.close ();
}

void MarchingCubesMesher::createMesh (float * evalValuesGPU, float isovalue, 
																			float invalidIsovalue) {
	//informations de la grille
	float ox = (grid->getOrigin())[0];
	float oy = (grid->getOrigin())[1];
	float oz = (grid->getOrigin())[2];

	unsigned int nbX = grid->getNbCellsX();
	unsigned int nbY = grid->getNbCellsY();
	unsigned int nbZ = grid->getNbCellsZ();

	float dx = grid->getCellSizeX();
	float dy = grid->getCellSizeY();
	float dz = grid->getCellSizeZ();

	int nbThreadsPerBlock = 512;
	dim3 nbBlocks;

	//creation timer
	//	cudaEvent_t start,stop;
	//	float elapsedTime;
	//	cudaEventCreate(&start);
	//	cudaEventCreate(&stop);


	//calcul des valeurs de la fonction pour chaque point de la grille//////////////////////////////////////////////
	//	cudaEventRecord(start, 0);	
	//	size_t functionValuesSize = (nbX + 2) * (nbY + 2) * (nbZ + 2) * sizeof(float);
	float *deviceFunctionValues = evalValuesGPU;
	//	cudaMalloc((void**)&deviceFunctionValues, functionValuesSize); 
	nbBlocks = ceil((float)((nbX + 2) * (nbY + 2) * (nbZ + 2))/(float)nbThreadsPerBlock);
	if (nbBlocks.x > 65535){
		nbBlocks.y = ceil((float)nbBlocks.x / (float)65535);
		nbBlocks.x = 65535;
	}
	//	evalGrid<<<nbBlocks, nbThreadsPerBlock>>>(deviceFunctionValues, nbX, nbY, nbZ, ox, oy, oz, dx, dy, dz);
	//	cout <<"nbBlocks : " << nbBlocks.x << "x" << nbBlocks.y << " nbThreadsPerBlock " << nbThreadsPerBlock << endl;
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps calcul functionValues: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << functionValuesSize /(1024*1024) << " Mo" << endl;
	//cudaMemcpy(functionValues, deviceFunctionValues, functionValuesSize, cudaMemcpyDeviceToHost);


	//calcul de nonEmptyCube et index ///////////////////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);	
	size_t indexSize = (nbX + 1) * (nbY + 1) * (nbZ + 1) * sizeof(unsigned char);
	unsigned char *deviceIndex;
	cudaMalloc((void**)&deviceIndex, indexSize); 
	size_t nonEmptyCubeSize = (nbX + 1) * (nbY + 1) * (nbZ + 1) * sizeof(unsigned int);
	unsigned int *deviceNonEmptyCube;
	cudaMalloc((void**)&deviceNonEmptyCube, nonEmptyCubeSize);
	nbBlocks.y = 0;
	nbBlocks = ceil((float)((nbX + 1) * (nbY + 1) * (nbZ + 1))/(float)nbThreadsPerBlock);
	if (nbBlocks.x > 65535){
		nbBlocks.y = ceil((float)nbBlocks.x / (float)65535);
		nbBlocks.x = 65535;
	}
	computeIndex<<<nbBlocks, nbThreadsPerBlock>>> (deviceFunctionValues, deviceIndex, deviceNonEmptyCube, 
																								 nbX, nbY, nbZ, isovalue, invalidIsovalue);
	unsigned int lastValue;
	cudaMemcpy(&lastValue, deviceNonEmptyCube + (nbX + 1) * (nbY + 1) * (nbZ + 1) - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps calcul index et nonEmptyCube: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize)/(1024*1024) << " Mo" << endl;


	//scan et compactage nonEmptyCube//////////////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);	
	//size_t nonEmptyCubeScanSize = nonEmptyCubeSize;
	//unsigned int *deviceNonEmptyCubeScan;
	//cudaMalloc((void**)&deviceNonEmptyCubeScan, nonEmptyCubeScanSize);
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(deviceNonEmptyCube), thrust::device_ptr<unsigned int>(deviceNonEmptyCube + (nbX + 1) * (nbY + 1) * (nbZ + 1)), thrust::device_ptr<unsigned int>(deviceNonEmptyCube));
	unsigned int lastValueScan;
	cudaMemcpy(&lastValueScan, deviceNonEmptyCube + (nbX + 1) * (nbY + 1) * (nbZ + 1) - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbNonEmptyCubes = lastValue + lastValueScan;
	cout << "[Marching Cube Mesher] : "<< "number of non empty cells : " << nbNonEmptyCubes << " " << lastValue << " " << lastValueScan << endl;
	//	cout << "nombre de cubes non vides : " << nbNonEmptyCubes << " " << lastValue << " " << lastValueScan << endl;
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps scan nonEmptyCube: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize)/(1024*1024) << " Mo" << endl;

	//	cudaEventRecord(start, 0);	
	size_t compactNonEmptyCubeScanSize = nbNonEmptyCubes * sizeof(unsigned int);
	unsigned int *deviceCompactNonEmptyCubeScan;
	cudaMalloc((void**)&deviceCompactNonEmptyCubeScan, compactNonEmptyCubeScanSize);

	nbBlocks.y = 0;
	nbBlocks = ceil((float)((nbX + 1) * (nbY + 1) * (nbZ + 1))/(float)nbThreadsPerBlock);
	if (nbBlocks.x > 65535){
		nbBlocks.y = ceil((float)nbBlocks.x / (float)65535);
		nbBlocks.x = 65535;
	}
	compactCubes<<<nbBlocks, nbThreadsPerBlock>>>(deviceIndex, deviceNonEmptyCube, deviceCompactNonEmptyCubeScan, nbX, nbY, nbZ);
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps compact nonEmptyCube: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize)/(1024*1024) << " Mo" << endl;


	//calcul nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);
	size_t nbVerticesCubeSize = nbNonEmptyCubes * sizeof(unsigned int);
	unsigned int *deviceNbVerticesCube;
	cudaMalloc((void**)&deviceNbVerticesCube, nbVerticesCubeSize);
	size_t nbTrianglesCubeSize = nbNonEmptyCubes * sizeof(unsigned int);
	unsigned int *deviceNbTrianglesCube;
	cudaMalloc((void**)&deviceNbTrianglesCube, nbTrianglesCubeSize);
	nbBlocks.y = 0;
	nbBlocks = ceil((float)(nbNonEmptyCubes)/(float)nbThreadsPerBlock);
	if (nbBlocks.x > 65535){
		nbBlocks.y = ceil((float)nbBlocks.x / (float)65535);
		nbBlocks.x = 65535;
	}
	countTrianglesAndVertices<<<nbBlocks, nbThreadsPerBlock>>>(deviceNbVerticesCube, deviceNbTrianglesCube, deviceCompactNonEmptyCubeScan, deviceIndex, nbNonEmptyCubes, nbX, nbY, nbZ);
	unsigned int lastValue1;
	cudaMemcpy(&lastValue1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastValue2;
	cudaMemcpy(&lastValue2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps calcul nbVerticesCube et nbTrianglesCube: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize)/(1024*1024) << " Mo" << endl;


	//scan nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(deviceNbTrianglesCube), thrust::device_ptr<unsigned int>(deviceNbTrianglesCube + nbNonEmptyCubes), thrust::device_ptr<unsigned int>(deviceNbTrianglesCube));
	thrust::exclusive_scan(thrust::device_ptr<unsigned int>(deviceNbVerticesCube), thrust::device_ptr<unsigned int>(deviceNbVerticesCube + nbNonEmptyCubes), thrust::device_ptr<unsigned int>(deviceNbVerticesCube));
	unsigned int lastValueScan1;
	cudaMemcpy(&lastValueScan1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastValueScan2;
	cudaMemcpy(&lastValueScan2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbVertices =  lastValueScan2 + lastValue2;
	unsigned int nbTriangles = lastValueScan1 + lastValue1;
	//	cout << "nbTriangles : " << nbTriangles << endl;
	mesh->TSize = nbTriangles;
	//	cout << "nbVertices : " << nbVertices << endl;
	mesh->VSize = nbVertices;
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps scan nbVerticesCube et nbTrianglesCube: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize)/(1024*1024) << " Mo" << endl;


	//creation triangles et vertices /////////////////////////////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);
	size_t verticesSize = nbVertices * sizeof(float3);
	float3 *vertices = (float3*)malloc(verticesSize);
	float3 *deviceVertices;
	cudaMalloc((void**)&deviceVertices, verticesSize);
	size_t trianglesSize = nbTriangles * sizeof(uint3);
	uint3 *triangles = (uint3*)malloc(trianglesSize);
	uint3 *deviceTriangles;
	cudaMalloc((void**)&deviceTriangles, trianglesSize);
	nbBlocks.y = 0;
	nbBlocks = ceil((float)(nbNonEmptyCubes)/(float)nbThreadsPerBlock);
	if (nbBlocks.x > 65535){
		nbBlocks.y = ceil((float)nbBlocks.x / (float)65535);
		nbBlocks.x = 65535;
	}
	createTrianglesAndVertices<<<nbBlocks, nbThreadsPerBlock>>>(deviceVertices, deviceTriangles, 
																															deviceFunctionValues, deviceNbVerticesCube, 
																															deviceNbTrianglesCube, deviceCompactNonEmptyCubeScan, 
																															deviceNonEmptyCube, deviceIndex, nbNonEmptyCubes, nbX, nbY, nbZ, ox, oy, oz, dx, dy, dz, 
																															isovalue, invalidIsovalue);

	thrust::device_ptr<unsigned int> newEnd;
	thrust::device_ptr<unsigned int> thrustDeviceTriangles ((unsigned int *)deviceTriangles);
	newEnd = thrust::remove (thrustDeviceTriangles, 
													 thrustDeviceTriangles + 3*nbTriangles, 
													 UINT32_MAX);

	nbTriangles = newEnd - thrustDeviceTriangles;
	nbTriangles /= 3;
	trianglesSize = nbTriangles*sizeof (uint3);
	mesh->TSize = nbTriangles;
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "temps creation vertices et triangles: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize + verticesSize + trianglesSize)/(1024*1024) << " Mo" << endl;
	//	cudaEventRecord(start, 0);
	cudaMemcpy(vertices, deviceVertices, verticesSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(triangles, deviceTriangles, trianglesSize, cudaMemcpyDeviceToHost);
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "readback vertices et triangles: " << elapsedTime << " ms" << endl;
	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize + verticesSize + trianglesSize)/(1024*1024) << " Mo" << endl;


	mesh->T = (unsigned int*)triangles;
	mesh->V = (float*)vertices;

	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	cout << "---temps total: " << elapsedTime << " ms" << endl;


	//clean up
	//	cudaEventDestroy(start);
	//	cudaEventDestroy(stop);

	cudaFree(deviceNonEmptyCube);
	cudaFree(deviceFunctionValues);
	cudaFree(deviceIndex);
	cudaFree(deviceCompactNonEmptyCubeScan);
	cudaFree(deviceNbVerticesCube);
	cudaFree(deviceNbTrianglesCube);
	cudaFree(deviceVertices);
	cudaFree(deviceTriangles);
}

__global__ void computeIndex3D (unsigned char * voxelGrid, unsigned char * index, 
																unsigned int * nonEmptyCube, 
																uint3 res, 
																unsigned char isovalue, float invalidIsovalue){

	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z; 
	unsigned int i = (res.x - 1)*(res.y - 1)*idZ + (res.x - 1)*idY + idX;

	if (idX >= (res.x - 1) || idY >= (res.y - 1) || idZ >= (res.z - 1))
		return;

	unsigned char ind = 0;
	unsigned char v[8];

	v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
	v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
	v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
	v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

	//	if(v[0] >= isovalue) ind |= 1;
	//	if(v[1] >= isovalue) ind |= 2;
	//	if(v[2] >= isovalue) ind |= 4;
	//	if(v[3] >= isovalue) ind |= 8;
	//	if(v[4] >= isovalue) ind |= 16;
	//	if(v[5] >= isovalue) ind |= 32;
	//	if(v[6] >= isovalue) ind |= 64;
	//	if(v[7] >= isovalue) ind |= 128;

	if(v[0] <= isovalue) ind |= 1;
	if(v[1] <= isovalue) ind |= 2;
	if(v[2] <= isovalue) ind |= 4;
	if(v[3] <= isovalue) ind |= 8;
	if(v[4] <= isovalue) ind |= 16;
	if(v[5] <= isovalue) ind |= 32;
	if(v[6] <= isovalue) ind |= 64;
	if(v[7] <= isovalue) ind |= 128;

	index[i] = ind;
	nonEmptyCube[i] = (ind != 0 && ind != 255) ? 1 : 0;
}

__global__ void compactCubes3D (unsigned char * index, unsigned int * nonEmptyCubeScan,
																unsigned int * compactNonEmptyCubeScan, unsigned int indexSize){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < indexSize)
		if (index[i] != 0 && index[i] != 255)
			compactNonEmptyCubeScan[nonEmptyCubeScan[i]] = i;
}

__global__ void countTrianglesAndVertices3D (unsigned int * nbVerticesCube, unsigned int * nbTrianglesCube, 
																						 unsigned int * compactNonEmptyCubeScan, unsigned char * index, 
																						 unsigned int nbNonEmptyCubes, uint3 res) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbNonEmptyCubes) {		
		unsigned int indice = compactNonEmptyCubeScan[i];

		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int remainder, idX, idY, idZ;
		idZ = indice/(resM1.x*resM1.y);
		remainder = indice % (resM1.x*resM1.y);
		idY = remainder/resM1.x;
		idX = remainder % resM1.x;

		int nbVerts = 0;
		if ((edgeTable[index[indice]] & 1) && (idX != resM2.x)) nbVerts++; 
		if ((edgeTable[index[indice]] & 8) && (idY != resM2.y)) nbVerts++; 
		if ((edgeTable[index[indice]] & 256) && (idZ != resM2.z)) nbVerts++;
		nbVerticesCube[i] = nbVerts;

		nbTrianglesCube[i] = (idX != resM2.x && idY != resM2.y && idZ != resM2.z) ? numTriTable[index[indice]] : 0;
	}

	//	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	//	if(i<nbNonEmptyCubes){		
	//		unsigned int indice = compactNonEmptyCubeScan[i];
	//		unsigned int x = indice/((nbY+1) * (nbZ+1));
	//		unsigned int y = (indice - x * (nbY+1) * (nbZ+1)) / (nbZ+1);
	//		unsigned int z = indice - x * (nbY+1) * (nbZ+1) - y * (nbZ+1);
	//
	//		int nbVerts = 0;
	//		if ((edgeTable[index[indice]] & 1) && (x != nbX)) nbVerts++; 
	//		if ((edgeTable[index[indice]] & 8) && (y != nbY)) nbVerts++; 
	//		if ((edgeTable[index[indice]] & 256) && (z != nbZ)) nbVerts++;
	//		nbVerticesCube[i] = nbVerts;
	//
	//		nbTrianglesCube[i] = (x != nbX && y != nbY && z != nbZ) ? numTriTable[index[indice]] : 0;
	//	}
}

__global__ void createTrianglesAndVertices3D (float3 * vertices, uint3 * triangles, unsigned char * voxelGrid, 
																							unsigned int * nbVerticesCube, unsigned int * nbTrianglesCube, 
																							unsigned int * compactNonEmptyCubeScan, unsigned int * nonEmptyCubeScan, 
																							unsigned char * index, unsigned int nbNonEmptyCubes, 
																							uint3 res, float3 minC, float cellSize, 
																							float isovalue, float invalidIsovalue) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbNonEmptyCubes) {
		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int ind = compactNonEmptyCubeScan[i];
		unsigned int remainder, idX, idY, idZ;
		idZ = ind/(resM1.x*resM1.y);
		remainder = ind % (resM1.x*resM1.y);
		idY = remainder/resM1.x;
		idX = remainder % resM1.x;
		unsigned char indexCube = index[ind];

		unsigned int vert[12];
		float3 point = make_float3 (minC.x + idX * cellSize, minC.y + idY * cellSize, minC.z + idZ * cellSize);
		unsigned int indice = 0;
		short edge = edgeTable[indexCube];

		float v[8];
		unsigned int id[8];

		id[0] = idX 				+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[1] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[2] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[3] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[4] = idX					+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[5] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[6] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);
		id[7] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);

		v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
		v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
		v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
		v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

		bool valid = false;
		if ((edge & 1) && (idX != resM2.x)) {
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[1]) > invalidIsovalue);
			vert[0] = nbVerticesCube[i] + indice;
			vertices[vert[0]] = findIntersection(point, v[0], 
																					 point + make_float3(cellSize, 0, 0), v[1], 
																					 isovalue);
			indice ++;

		}
		if ((edge & 2) && (idX != resM2.x) && (idY != resM2.y)) {
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[2]) > invalidIsovalue);
			vert[1] = nbVerticesCube[nonEmptyCubeScan[id[1]]];			
			if (edgeTable[index[id[1]]]&1 && (idX+1)!=resM2.x) {
				vert[1] ++;
			}
		}
		if ((edge & 4) && (idX != resM2.x) && (idY != resM2.y)) {
			valid |= (fabs (v[2]) > invalidIsovalue) || (fabs (v[3]) > invalidIsovalue);
			vert[2] = nbVerticesCube[nonEmptyCubeScan[id[3]]];
		}
		if ((edge & 8) && (idY != resM2.y)) {
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[0]) > invalidIsovalue);
			vert[3] = nbVerticesCube[i] + indice;
			vertices[vert[3]] = findIntersection(point + make_float3(0, cellSize, 0), v[3], 
																					 point, v[0], 
																					 isovalue);
			indice ++;
		}
		if ((edge & 16) && (idZ != resM2.z) && (idX != resM2.x)){
			valid = valid || (fabs (v[4]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			vert[4] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
		}

		if ((edge & 32) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[5]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			vert[5] = nbVerticesCube[nonEmptyCubeScan[id[5]]];
			if(edgeTable[index[id[5]]]&1 && (idX+1)!=resM2.x){
				vert[5] ++;
			}
		}
		if ((edge & 64) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[6]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			vert[6] = nbVerticesCube[nonEmptyCubeScan[id[7]]];
		}
		if ((edge & 128) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[7]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[7] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
			if(edgeTable[index[id[4]]]&1 && idX!=resM2.x){
				vert[7] ++;
			}
		}		
		if ((edge & 256) && (idZ != resM2.z)){
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[8] = nbVerticesCube[i] + indice;
			vertices[vert[8]] = findIntersection(point , v[0], 
																					 point + make_float3(0, 0, cellSize), v[4], 
																					 isovalue);
			indice ++;
		}
		unsigned int d;
		short edge1;
		if ((edge & 512) && (idX != resM2.x) && (idZ != resM2.z)){
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[1]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY!=resM2.y)) d++; 
			vert[9] = nbVerticesCube[nonEmptyCubeScan[id[1]]] + d;
		}
		if ((edge & 1024) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[2]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[2]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[10] = nbVerticesCube[nonEmptyCubeScan[id[2]]] + d;
		}
		if ((edge & 2048) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[3]]];
			if(edge1 & 1 && idX!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[11] = nbVerticesCube[nonEmptyCubeScan[id[3]]] + d;
		}

		unsigned int tri = 0;	
		if (indexCube != 0 && indexCube != 255 && idX != resM2.x && idY != resM2.y && idZ != resM2.z){
			unsigned int nbTri = nbTrianglesCube[i];
			for (int s = 0; triTable[indexCube * 16 + s] != -1; s+=3){
				if (!valid)
					triangles[nbTri + tri] = make_uint3 (vert[triTable[indexCube * 16 + s + 2]], 
																							 vert[triTable[indexCube * 16 + s + 1]], 
																							 vert[triTable[indexCube * 16 + s]]);
				else
					triangles[nbTri + tri] = make_uint3 (UINT32_MAX, UINT32_MAX, UINT32_MAX);
				tri++;
			}
		}
	}
}

void MarchingCubesMesher::createMesh3D (unsigned char * evalValuesGPU, unsigned char isovalue, 
																				float invalidIsovalue) {
	uint3 res = make_uint3 (grid->getNbCellsX (), grid->getNbCellsY (), grid->getNbCellsZ ());
	float3 minC = make_float3 ((grid->getOrigin ())[0], (grid->getOrigin ())[1], (grid->getOrigin ())[2]);
	float cellSize = grid->getCellSizeX ();
	//	unsigned int gridSize = res.x*res.y*res.z;
	unsigned int indexSize = (res.x - 1)*(res.y - 1)*(res.z - 1);

	dim3 blockDim, gridDim;

	blockDim = dim3 (8, 8, 8);
	gridDim = dim3 (((res.x - 1)/blockDim.x)+1, ((res.y - 1)/blockDim.y)+1, 
									((res.z - 1)/blockDim.z)+1);

	unsigned char * deviceFunctionValues = evalValuesGPU;
	unsigned char * deviceIndex = NULL;
	unsigned int * deviceNonEmptyCube = NULL;
	unsigned int * deviceCompactNonEmptyCubeScan = NULL;
	unsigned int * deviceNbVerticesCube = NULL;
	unsigned int * deviceNbTrianglesCube = NULL;
	float3 * deviceVertices = NULL;
	uint3 * deviceTriangles = NULL;
	float3 * vertices = NULL;
	uint3 * triangles = NULL;

	//calcul de nonEmptyCube et index ///////////////////////////////////////////////////////////////////////////////
	cudaMalloc ((void**)&deviceIndex, indexSize*sizeof (unsigned char)); 
	cudaMalloc ((void**)&deviceNonEmptyCube, indexSize*sizeof (unsigned int));
	checkCUDAError ();

	computeIndex3D<<<gridDim, blockDim>>> (deviceFunctionValues, deviceIndex, deviceNonEmptyCube, 
																				 res, isovalue, invalidIsovalue);
	unsigned int lastValue;
	cudaMemcpy (&lastValue, deviceNonEmptyCube + indexSize - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);

	std::cout << "[Marching Cube Mesher] : "<< "index computed " << indexSize 
		<< std::endl;
	//	//scan et compactage nonEmptyCube//////////////////////////////////////////////////////////////////////////
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNonEmptyCube), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube + indexSize), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube));
	unsigned int lastValueScan;
	cudaMemcpy (&lastValueScan, deviceNonEmptyCube + indexSize - 1, 
							sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbNonEmptyCubes = lastValue + lastValueScan;
	std::cout << "[Marching Cube Mesher] : "<< "number of non empty cells : " 
		<< nbNonEmptyCubes << std::endl;

	cudaMalloc ((void**)&deviceCompactNonEmptyCubeScan, nbNonEmptyCubes * sizeof(unsigned int));
	checkCUDAError ();

	compactCubes3D<<<(indexSize/512)+1, 512>>> (deviceIndex, deviceNonEmptyCube, deviceCompactNonEmptyCubeScan, indexSize);

	//calcul nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	//	cudaEventRecord(start, 0);
	cudaMalloc ((void**)&deviceNbVerticesCube, nbNonEmptyCubes*sizeof (unsigned int));
	cudaMalloc ((void**)&deviceNbTrianglesCube, nbNonEmptyCubes*sizeof (unsigned int));
	checkCUDAError ();

	countTrianglesAndVertices3D
		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceNbVerticesCube, deviceNbTrianglesCube, 
																				deviceCompactNonEmptyCubeScan, deviceIndex, nbNonEmptyCubes, 
																				res);
	unsigned int lastValue1;
	cudaMemcpy(&lastValue1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastValue2;
	cudaMemcpy(&lastValue2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	std::cout << "[Marching Cube Mesher] : "<< "triangles and vertices counted" << std::endl;

	//scan nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbTrianglesCube), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube));
	std::cout << "[Marching Cube Mesher] : "<< "triangles count scanned" << std::endl;
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbVerticesCube), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube));
	std::cout << "[Marching Cube Mesher] : "<< "vertices count scanned" << std::endl;
	unsigned int lastValueScan1, lastValueScan2;
	cudaMemcpy (&lastValueScan1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy (&lastValueScan2, deviceNbVerticesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbVertices =  lastValueScan2 + lastValue2;
	unsigned int nbTriangles = lastValueScan1 + lastValue1;
	mesh->TSize = nbTriangles;
	mesh->VSize = nbVertices;
	std::cout << "[Marching Cube Mesher] : "<< "there are " << nbTriangles << " triangles and "
		<< nbVertices << " vertices" << std::endl;

	//creation triangles et vertices /////////////////////////////////////////////////////////////////////////////////////////
	vertices = (float3 *) malloc (nbVertices*sizeof (float3));
	cudaMalloc ((void**)&deviceVertices, nbVertices*sizeof (float3));

	triangles = (uint3 *) malloc (nbTriangles*sizeof (uint3));
	cudaMalloc ((void**)&deviceTriangles, nbTriangles*sizeof (uint3));

	createTrianglesAndVertices3D
		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceVertices, deviceTriangles, 
																				deviceFunctionValues, deviceNbVerticesCube, 
																				deviceNbTrianglesCube, deviceCompactNonEmptyCubeScan, 
																				deviceNonEmptyCube, deviceIndex, nbNonEmptyCubes, 
																				res, minC, cellSize, 
																				isovalue, invalidIsovalue);

	//	thrust::device_ptr<unsigned int> newEnd;
	//	thrust::device_ptr<unsigned int> thrustDeviceTriangles ((unsigned int *)deviceTriangles);
	//	newEnd = thrust::remove (thrustDeviceTriangles, 
	//													 thrustDeviceTriangles + 3*nbTriangles, 
	//													 UINT32_MAX);
	//
	//	nbTriangles = newEnd - thrustDeviceTriangles;
	//	nbTriangles /= 3;
	//	trianglesSize = nbTriangles*sizeof (uint3);
	//	mesh->TSize = nbTriangles;
	//	//	cudaEventRecord(stop, 0);
	//	//	cudaEventSynchronize(stop);
	//	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	//	cout << "temps creation vertices et triangles: " << elapsedTime << " ms" << endl;
	//	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize + verticesSize + trianglesSize)/(1024*1024) << " Mo" << endl;
	//	//	cudaEventRecord(start, 0);
	cudaMemcpy (vertices, deviceVertices, nbVertices*sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (triangles, deviceTriangles, nbTriangles*sizeof (uint3), cudaMemcpyDeviceToHost);
	mesh->T = (unsigned int*) triangles;
	mesh->V = (float*) vertices;

	cudaFree (deviceNonEmptyCube);
	cudaFree (deviceIndex);
	cudaFree (deviceCompactNonEmptyCubeScan);
	cudaFree (deviceNbVerticesCube);
	cudaFree (deviceNbTrianglesCube);
	cudaFree (deviceVertices);
	cudaFree (deviceTriangles);
}

__global__ void computeIndex3D (float * voxelGrid, unsigned char * index, 
																unsigned int * nonEmptyCube, 
																uint3 res, 
																float isovalue, float invalidIsovalue){

	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z; 
	unsigned int i = (res.x - 1)*(res.y - 1)*idZ + (res.x - 1)*idY + idX;

	if (idX >= (res.x - 1) || idY >= (res.y - 1) || idZ >= (res.z - 1))
		return;

	unsigned char ind = 0;
	float v[8];

	v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
	v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
	v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
	v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

	//	if(v[0] >= isovalue) ind |= 1;
	//	if(v[1] >= isovalue) ind |= 2;
	//	if(v[2] >= isovalue) ind |= 4;
	//	if(v[3] >= isovalue) ind |= 8;
	//	if(v[4] >= isovalue) ind |= 16;
	//	if(v[5] >= isovalue) ind |= 32;
	//	if(v[6] >= isovalue) ind |= 64;
	//	if(v[7] >= isovalue) ind |= 128;

	if(v[0] <= isovalue) ind |= 1;
	if(v[1] <= isovalue) ind |= 2;
	if(v[2] <= isovalue) ind |= 4;
	if(v[3] <= isovalue) ind |= 8;
	if(v[4] <= isovalue) ind |= 16;
	if(v[5] <= isovalue) ind |= 32;
	if(v[6] <= isovalue) ind |= 64;
	if(v[7] <= isovalue) ind |= 128;

	index[i] = ind;
	nonEmptyCube[i] = (ind != 0 && ind != 255) ? 1 : 0;
}

__global__ void createTrianglesAndVertices3D (float3 * vertices, uint3 * triangles, float * voxelGrid, 
																							unsigned int * nbVerticesCube, unsigned int * nbTrianglesCube, 
																							unsigned int * compactNonEmptyCubeScan, unsigned int * nonEmptyCubeScan, 
																							unsigned char * index, unsigned int nbNonEmptyCubes, 
																							uint3 res, float3 minC, float cellSize, 
																							float isovalue, float invalidIsovalue) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbNonEmptyCubes) {
		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int ind = compactNonEmptyCubeScan[i];
		unsigned int remainder, idX, idY, idZ;
		idZ = ind/(resM1.x*resM1.y);
		remainder = ind % (resM1.x*resM1.y);
		idY = remainder/resM1.x;
		idX = remainder % resM1.x;
		unsigned char indexCube = index[ind];

		unsigned int vert[12];
		float3 point = make_float3 (minC.x + idX * cellSize, minC.y + idY * cellSize, minC.z + idZ * cellSize);
		unsigned int indice = 0;
		short edge = edgeTable[indexCube];

		float v[8];
		unsigned int id[8];

		id[0] = idX 				+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[1] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[2] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[3] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[4] = idX					+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[5] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[6] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);
		id[7] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);

		v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
		v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
		v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
		v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

		bool valid = false;
		if ((edge & 1) && (idX != resM2.x)) {
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[1]) > invalidIsovalue);
			vert[0] = nbVerticesCube[i] + indice;
			vertices[vert[0]] = findIntersection(point, v[0], 
																					 point + make_float3(cellSize, 0, 0), v[1], 
																					 isovalue);
			indice ++;

		}
		if ((edge & 2) && (idX != resM2.x) && (idY != resM2.y)) {
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[2]) > invalidIsovalue);
			vert[1] = nbVerticesCube[nonEmptyCubeScan[id[1]]];			
			if (edgeTable[index[id[1]]]&1 && (idX+1)!=resM2.x) {
				vert[1] ++;
			}
		}
		if ((edge & 4) && (idX != resM2.x) && (idY != resM2.y)) {
			valid |= (fabs (v[2]) > invalidIsovalue) || (fabs (v[3]) > invalidIsovalue);
			vert[2] = nbVerticesCube[nonEmptyCubeScan[id[3]]];
		}
		if ((edge & 8) && (idY != resM2.y)) {
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[0]) > invalidIsovalue);
			vert[3] = nbVerticesCube[i] + indice;
			vertices[vert[3]] = findIntersection(point + make_float3(0, cellSize, 0), v[3], 
																					 point, v[0], 
																					 isovalue);
			indice ++;
		}
		if ((edge & 16) && (idZ != resM2.z) && (idX != resM2.x)){
			valid = valid || (fabs (v[4]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			vert[4] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
		}

		if ((edge & 32) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[5]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			vert[5] = nbVerticesCube[nonEmptyCubeScan[id[5]]];
			if(edgeTable[index[id[5]]]&1 && (idX+1)!=resM2.x){
				vert[5] ++;
			}
		}
		if ((edge & 64) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[6]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			vert[6] = nbVerticesCube[nonEmptyCubeScan[id[7]]];
		}
		if ((edge & 128) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[7]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[7] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
			if(edgeTable[index[id[4]]]&1 && idX!=resM2.x){
				vert[7] ++;
			}
		}		
		if ((edge & 256) && (idZ != resM2.z)){
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[8] = nbVerticesCube[i] + indice;
			vertices[vert[8]] = findIntersection(point , v[0], 
																					 point + make_float3(0, 0, cellSize), v[4], 
																					 isovalue);
			indice ++;
		}
		unsigned int d;
		short edge1;
		if ((edge & 512) && (idX != resM2.x) && (idZ != resM2.z)){
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[1]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY!=resM2.y)) d++; 
			vert[9] = nbVerticesCube[nonEmptyCubeScan[id[1]]] + d;
		}
		if ((edge & 1024) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[2]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[2]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[10] = nbVerticesCube[nonEmptyCubeScan[id[2]]] + d;
		}
		if ((edge & 2048) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[3]]];
			if(edge1 & 1 && idX!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[11] = nbVerticesCube[nonEmptyCubeScan[id[3]]] + d;
		}

		unsigned int tri = 0;	
		if (indexCube != 0 && indexCube != 255 && idX != resM2.x && idY != resM2.y && idZ != resM2.z){
			unsigned int nbTri = nbTrianglesCube[i];
			for (int s = 0; triTable[indexCube * 16 + s] != -1; s+=3){
				if (!valid)
					triangles[nbTri + tri] = make_uint3 (vert[triTable[indexCube * 16 + s + 2]], 
																							 vert[triTable[indexCube * 16 + s + 1]], 
																							 vert[triTable[indexCube * 16 + s]]);
				else
					triangles[nbTri + tri] = make_uint3 (UINT32_MAX, UINT32_MAX, UINT32_MAX);
				tri++;
			}
		}
	}
}

void MarchingCubesMesher::createMesh3D (float * evalValuesGPU, float isovalue, 
																				float invalidIsovalue) {
	double time1, time2;
	uint3 res = make_uint3 (grid->getNbCellsX (), grid->getNbCellsY (), grid->getNbCellsZ ());
	float3 minC = make_float3 ((grid->getOrigin ())[0], (grid->getOrigin ())[1], (grid->getOrigin ())[2]);
	float cellSize = grid->getCellSizeX ();
	//	unsigned int gridSize = res.x*res.y*res.z;
	unsigned int indexSize = (res.x - 1)*(res.y - 1)*(res.z - 1);

	dim3 blockDim, gridDim;

	blockDim = dim3 (8, 8, 8);
	gridDim = dim3 (((res.x - 1)/blockDim.x)+1, ((res.y - 1)/blockDim.y)+1, 
									((res.z - 1)/blockDim.z)+1);

	float * deviceFunctionValues = evalValuesGPU;
	unsigned char * deviceIndex = NULL;
	unsigned int * deviceNonEmptyCube = NULL;
	unsigned int * deviceCompactNonEmptyCubeScan = NULL;
	unsigned int * deviceNbVerticesCube = NULL;
	unsigned int * deviceNbTrianglesCube = NULL;
	float3 * deviceVertices = NULL;
	uint3 * deviceTriangles = NULL;
	float3 * vertices = NULL;
	uint3 * triangles = NULL;

	//calcul de nonEmptyCube et index ///////////////////////////////////////////////////////////////////////////////
	cudaMalloc ((void**)&deviceIndex, indexSize*sizeof (unsigned char)); 
	cudaMalloc ((void**)&deviceNonEmptyCube, indexSize*sizeof (unsigned int));
	checkCUDAError ();

	time1 = GET_TIME ();
	computeIndex3D<<<gridDim, blockDim>>> (deviceFunctionValues, deviceIndex, deviceNonEmptyCube, 
																				 res, isovalue, invalidIsovalue);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	unsigned int lastValue;
	cudaMemcpy (&lastValue, deviceNonEmptyCube + indexSize - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);

	std::cout << "[Marching Cube Mesher] : "<< indexSize << " index computed in "
		<< time2 - time1 << " ms." << std::endl;
	//	//scan et compactage nonEmptyCube//////////////////////////////////////////////////////////////////////////
	time1 = GET_TIME ();
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNonEmptyCube), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube + indexSize), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube));
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	unsigned int lastValueScan;
	cudaMemcpy (&lastValueScan, deviceNonEmptyCube + indexSize - 1, 
							sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbNonEmptyCubes = lastValue + lastValueScan;
	std::cout << "[Marching Cube Mesher] : " << nbNonEmptyCubes << " non empty cells compacted in " 
		<< time2 - time1 << " ms." << std::endl;

	cudaMalloc ((void**)&deviceCompactNonEmptyCubeScan, nbNonEmptyCubes * sizeof(unsigned int));
	checkCUDAError ();

	compactCubes3D<<<(indexSize/512)+1, 512>>> (deviceIndex, deviceNonEmptyCube, deviceCompactNonEmptyCubeScan, indexSize);

	//calcul nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	cudaMalloc ((void**)&deviceNbVerticesCube, nbNonEmptyCubes*sizeof (unsigned int));
	cudaMalloc ((void**)&deviceNbTrianglesCube, nbNonEmptyCubes*sizeof (unsigned int));
	checkCUDAError ();

	time1 = GET_TIME ();
	countTrianglesAndVertices3D
		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceNbVerticesCube, deviceNbTrianglesCube, 
																				deviceCompactNonEmptyCubeScan, deviceIndex, nbNonEmptyCubes, 
																				res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	unsigned int lastValue1;
	cudaMemcpy(&lastValue1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastValue2;
	cudaMemcpy(&lastValue2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	std::cout << "[Marching Cube Mesher] : "<< "triangles and vertices counted in "
		<< time2 - time1 << " ms." << std::endl;

	//scan nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbTrianglesCube), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube));
	std::cout << "[Marching Cube Mesher] : "<< "triangles count scanned" << std::endl;
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbVerticesCube), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube));
	std::cout << "[Marching Cube Mesher] : "<< "vertices count scanned" << std::endl;
	unsigned int lastValueScan1, lastValueScan2;
	cudaMemcpy (&lastValueScan1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy (&lastValueScan2, deviceNbVerticesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbVertices =  lastValueScan2 + lastValue2;
	unsigned int nbTriangles = lastValueScan1 + lastValue1;
	mesh->TSize = nbTriangles;
	mesh->VSize = nbVertices;
	std::cout << "[Marching Cube Mesher] : "<< "there are " << nbTriangles << " triangles and "
		<< nbVertices << " vertices" << std::endl;

	//creation triangles et vertices /////////////////////////////////////////////////////////////////////////////////////////
	vertices = (float3 *) malloc (nbVertices*sizeof (float3));
	cudaMalloc ((void**)&deviceVertices, nbVertices*sizeof (float3));

	triangles = (uint3 *) malloc (nbTriangles*sizeof (uint3));
	cudaMalloc ((void**)&deviceTriangles, nbTriangles*sizeof (uint3));

	createTrianglesAndVertices3D
		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceVertices, deviceTriangles, 
																				deviceFunctionValues, deviceNbVerticesCube, 
																				deviceNbTrianglesCube, deviceCompactNonEmptyCubeScan, 
																				deviceNonEmptyCube, deviceIndex, nbNonEmptyCubes, 
																				res, minC, cellSize, 
																				isovalue, invalidIsovalue);

	//	thrust::device_ptr<unsigned int> newEnd;
	//	thrust::device_ptr<unsigned int> thrustDeviceTriangles ((unsigned int *)deviceTriangles);
	//	newEnd = thrust::remove (thrustDeviceTriangles, 
	//													 thrustDeviceTriangles + 3*nbTriangles, 
	//													 UINT32_MAX);
	//
	//	nbTriangles = newEnd - thrustDeviceTriangles;
	//	nbTriangles /= 3;
	//	trianglesSize = nbTriangles*sizeof (uint3);
	//	mesh->TSize = nbTriangles;
	//	//	cudaEventRecord(stop, 0);
	//	//	cudaEventSynchronize(stop);
	//	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	//	cout << "temps creation vertices et triangles: " << elapsedTime << " ms" << endl;
	//	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize + verticesSize + trianglesSize)/(1024*1024) << " Mo" << endl;
	//	//	cudaEventRecord(start, 0);

	cudaMemcpy (vertices, deviceVertices, nbVertices*sizeof (float3), cudaMemcpyDeviceToHost);
	cudaMemcpy (triangles, deviceTriangles, nbTriangles*sizeof (uint3), cudaMemcpyDeviceToHost);
	mesh->T = (unsigned int*) triangles;
	mesh->V = (float*) vertices;

	cudaFree (deviceNonEmptyCube);
	cudaFree (deviceIndex);
	cudaFree (deviceCompactNonEmptyCubeScan);
	cudaFree (deviceNbVerticesCube);
	cudaFree (deviceNbTrianglesCube);
	cudaFree (deviceVertices);
	cudaFree (deviceTriangles);
}

__global__ void countTrianglesAndVertices3D (unsigned int * nbVerticesCube, 
																						 unsigned int * nbTrianglesCube, 
																						 unsigned char * contour_values, 
																						 unsigned int contour_size, uint3 res) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < contour_size) {		
		int nbVerts = 0;
		if ((edgeTable[contour_values[i]] & 1)) nbVerts++; 
		if ((edgeTable[contour_values[i]] & 8)) nbVerts++; 
		if ((edgeTable[contour_values[i]] & 256)) nbVerts++;
		nbVerticesCube[i] = nbVerts;
		nbTrianglesCube[i] = numTriTable[contour_values[i]];
	}
}

__global__ void createTrianglesAndVertices3D (float3 * vertices, 
																							float3 * normals, 
																							uint3 * triangles, 
																							unsigned int * nbVerticesCube, 
																							unsigned int * nbTrianglesCube, 
																							unsigned char * contour_values, 
																							unsigned int * contour_indices, 
																							unsigned int * contour_neigh_indices, 
																							unsigned int * contour_neigh_morpho_centroids, 
																							unsigned int contour_size, 
																							uint3 res, float3 minC, float cellSize, 
																							float isovalue, float invalidIsovalue) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < contour_size) {
		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int contour_index = contour_indices[i];
		unsigned char contour_value = contour_values[i];
		unsigned int remainder, idX, idY, idZ;
		idZ = contour_index/(res.x*res.y);
		remainder = contour_index % (res.x*res.y);
		idY = remainder/res.x;
		idX = remainder % res.x;

		unsigned int vert[12];
		float3 point = make_float3 (minC.x + idX * cellSize, 
																minC.y + idY * cellSize, 
																minC.z + idZ * cellSize);
		unsigned int indice = 0;
		short edge = edgeTable[contour_value];

		float v[8];
		unsigned int id[8];

		id[0] = contour_neigh_indices[8*i + 0];
		id[1] = contour_neigh_indices[8*i + 1];
		id[2] = contour_neigh_indices[8*i + 2];
		id[3] = contour_neigh_indices[8*i + 3];
		id[4] = contour_neigh_indices[8*i + 4];
		id[5] = contour_neigh_indices[8*i + 5];
		id[6] = contour_neigh_indices[8*i + 6];
		id[7] = contour_neigh_indices[8*i + 7];

		v[0] = (contour_value & MC0) != 0 ? 1.f : 0.f;
		v[1] = (contour_value & MC1) != 0 ? 1.f : 0.f;
		v[2] = (contour_value & MC2) != 0 ? 1.f : 0.f;
		v[3] = (contour_value & MC3) != 0 ? 1.f : 0.f;
		v[4] = (contour_value & MC4) != 0 ? 1.f : 0.f;
		v[5] = (contour_value & MC5) != 0 ? 1.f : 0.f;
		v[6] = (contour_value & MC6) != 0 ? 1.f : 0.f;
		v[7] = (contour_value & MC7) != 0 ? 1.f : 0.f;

		for (int k = 0; k < 12; k++)
			vert[k] = 0;

		if (edge & 1) {
			vert[0] = nbVerticesCube[i] + indice;
			vertices[vert[0]] = findIntersection(point, v[0], 
																					 point + make_float3(cellSize, 0, 0), v[1], 
																					 isovalue);
			unsigned int morpho_centroid_morton;
			if (v[0] == 0.f) {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 0];
			} else {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 1];
			}
			float3 morpho_centroid = make_float3 (DecodeMorton3X (morpho_centroid_morton), 
																						DecodeMorton3Y (morpho_centroid_morton), 
																						DecodeMorton3Z (morpho_centroid_morton));
			float3 morpho_normal = make_float3 (idX, idY, idZ);
			morpho_normal = morpho_centroid - morpho_normal;
			normalize (morpho_normal);
			normals[vert[0]] = morpho_normal;

			indice ++;

		}
		if (edge & 2) {
			if (id[1] != 0xffffffff) {
				vert[1] = nbVerticesCube[id[1]];	
				if (edgeTable[contour_values[id[1]]]&1) {
					vert[1] ++;
				}
			}		
		}
		if (edge & 4) {
			vert[2] = (id[3] != 0xffffffff) ? nbVerticesCube[id[3]] : 0;
		}
		if (edge & 8) {
			vert[3] = nbVerticesCube[i] + indice;
			vertices[vert[3]] = findIntersection(point + make_float3(0, cellSize, 0), v[3], 
																					 point, v[0], 
																					 isovalue);
			unsigned int morpho_centroid_morton;
			if (v[0] == 0.f) {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 0];
			} else {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 3];
			}
			float3 morpho_centroid = make_float3 (DecodeMorton3X (morpho_centroid_morton), 
																						DecodeMorton3Y (morpho_centroid_morton), 
																						DecodeMorton3Z (morpho_centroid_morton));
			float3 morpho_normal = make_float3 (idX, idY, idZ);
			morpho_normal = morpho_centroid - morpho_normal;
			normalize (morpho_normal);
			normals[vert[3]] = morpho_normal;

			indice ++;
		}
		if (edge & 16){
			vert[4] = (id[4] != 0xffffffff) ? nbVerticesCube[id[4]] : 0;
		}

		if (edge & 32) {
			if (id[5] != 0xffffffff) {
				vert[5] = nbVerticesCube[id[5]];	
				if (edgeTable[contour_values[id[5]]]&1) {
					vert[5] ++;
				}
			}		
		}
		if (edge & 64) {
			vert[6] = (id[7] != 0xffffffff) ? nbVerticesCube[id[7]] : 0;
		}
		if (edge & 128){
			if (id[4] != 0xffffffff) {
				vert[7] = nbVerticesCube[id[4]];	
				if (edgeTable[contour_values[id[4]]]&1) {
					vert[7] ++;
				}
			}		
		}		
		if (edge & 256){
			vert[8] = nbVerticesCube[i] + indice;
			vertices[vert[8]] = findIntersection(point , v[0], 
																					 point + make_float3(0, 0, cellSize), v[4], 
																					 isovalue);

			unsigned int morpho_centroid_morton;
			if (v[0] == 0.f) {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 0];
			} else {
				morpho_centroid_morton = contour_neigh_morpho_centroids[8*i + 4];
			}
			float3 morpho_centroid = make_float3 (DecodeMorton3X (morpho_centroid_morton), 
																						DecodeMorton3Y (morpho_centroid_morton), 
																						DecodeMorton3Z (morpho_centroid_morton));
			float3 morpho_normal = make_float3 (idX, idY, idZ);
			morpho_normal = morpho_centroid - morpho_normal;
			normalize (morpho_normal);
			normals[vert[8]] = morpho_normal;

			indice ++;
		}
		unsigned int d;
		short edge1;
		if (edge & 512){
			d = 0;
			edge1 = (id[1] != 0xffffffff) ? edgeTable[contour_values[id[1]]] : 0;
			if(edge1 & 1) d++;
			if(edge1 & 8) d++; 
			vert[9] = (id[1] != 0xffffffff) ? (nbVerticesCube[id[1]] + d) : d;
		}
		if (edge & 1024){
			d = 0;
			edge1 = (id[2] != 0xffffffff) ? edgeTable[contour_values[id[2]]] : 0;
			if(edge1 & 1) d++;
			if(edge1 & 8) d++;
			vert[10] = (id[2] != 0xffffffff) ? (nbVerticesCube[id[2]] + d) : d;
		}
		if (edge & 2048){
			d = 0;
			edge1 = (id[3] != 0xffffffff) ? edgeTable[contour_values[id[3]]] : 0;
			if(edge1 & 1) d++;
			if(edge1 & 8) d++;
			vert[11] = (id[3] != 0xffffffff) ? (nbVerticesCube[id[3]] + d) : d;
		}
		unsigned int tri = 0;	
		unsigned int nbTri = nbTrianglesCube[i];
		for (int s = 0; triTable[contour_value * 16 + s] != -1; s+=3){
			uint3 tri_s = make_uint3 (vert[triTable[contour_value * 16 + s + 2]], 
																vert[triTable[contour_value * 16 + s + 1]], 
																vert[triTable[contour_value * 16 + s]]);
			triangles[nbTri + tri] = tri_s;
			tri++;
		}
	}
}

void MarchingCubesMesher::createMesh3D (unsigned char * contour_values, 
																				unsigned int * contour_indices, 
																				unsigned int * contour_neigh_indices, 
																				unsigned int * contour_neigh_morpho_centroids, 
																				unsigned int contour_size, 
																				float isovalue, 
																				float invalidIsovalue) {
	double time1, time2;
	uint3 res = make_uint3 (grid->getNbCellsX (), grid->getNbCellsY (), grid->getNbCellsZ ());
	float3 minC = make_float3 ((grid->getOrigin ())[0], (grid->getOrigin ())[1], (grid->getOrigin ())[2]);
	float cellSize = grid->getCellSizeX ();
	//	unsigned int gridSize = res.x*res.y*res.z;
	//	unsigned int indexSize = (res.x - 1)*(res.y - 1)*(res.z - 1);

	dim3 blockDim, gridDim;

	blockDim = dim3 (8, 8, 8);
	gridDim = dim3 (((res.x - 1)/blockDim.x)+1, ((res.y - 1)/blockDim.y)+1, 
									((res.z - 1)/blockDim.z)+1);

	//	unsigned char * deviceIndex = NULL;
	//	unsigned int * deviceNonEmptyCube = NULL;
	//	unsigned int * deviceCompactNonEmptyCubeScan = NULL;
	unsigned int * deviceNbVerticesCube = NULL;
	unsigned int * deviceNbTrianglesCube = NULL;
	//	float3 * deviceVertices = NULL;
	//	float3 * deviceNormals = NULL;
	//	uint3 * deviceTriangles = NULL;
	//	float3 * vertices = NULL;
	//	float3 * normals = NULL;
	//	uint3 * triangles = NULL;

	unsigned int nbNonEmptyCubes = contour_size;
	//	deviceCompactNonEmptyCubeScan = contour_indices;


	//calcul nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	cudaMalloc ((void**)&deviceNbVerticesCube, nbNonEmptyCubes*sizeof (unsigned int));
	cudaMalloc ((void**)&deviceNbTrianglesCube, nbNonEmptyCubes*sizeof (unsigned int));
	checkCUDAError ();

	time1 = GET_TIME ();
	countTrianglesAndVertices3D
		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceNbVerticesCube, deviceNbTrianglesCube, 
																				contour_values, contour_size, res);
	cudaDeviceSynchronize ();
	time2 = GET_TIME ();
	unsigned int lastValue1;
	cudaMemcpy(&lastValue1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int lastValue2;
	cudaMemcpy(&lastValue2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	std::cout << "[Marching Cube Mesher] : "<< "triangles and vertices counted in "
		<< time2 - time1 << " ms." << std::endl;

	//scan nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbTrianglesCube), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube));
	std::cout << "[Marching Cube Mesher] : "<< "triangles count scanned" << std::endl;
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbVerticesCube), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube + nbNonEmptyCubes), 
													thrust::device_ptr<unsigned int> (deviceNbVerticesCube));
	std::cout << "[Marching Cube Mesher] : "<< "vertices count scanned" << std::endl;
	unsigned int lastValueScan1, lastValueScan2;
	cudaMemcpy (&lastValueScan1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy (&lastValueScan2, deviceNbVerticesCube + nbNonEmptyCubes - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbVertices =  lastValueScan2 + lastValue2;
	unsigned int nbTriangles = lastValueScan1 + lastValue1;
	mesh->TSize = nbTriangles;
	mesh->VSize = nbVertices;
	std::cout << "[Marching Cube Mesher] : "<< "there are " << nbTriangles << " triangles and "
		<< nbVertices << " vertices" << std::endl;

	//creation triangles et vertices /////////////////////////////////////////////////////////////////////////////////////////
	//	vertices = (float3 *) malloc (nbVertices*sizeof (float3));
	//	normals = (float3 *) malloc (nbVertices*sizeof (float3));
	//	triangles = (uint3 *) malloc (nbTriangles*sizeof (uint3));

	//	cudaMalloc ((void**)&mesh->devV, nbVertices*sizeof (float3));
	//	cudaMalloc ((void**)&mesh->devN, nbVertices*sizeof (float3));
	//	cudaMalloc ((void**)&mesh->devT, nbTriangles*sizeof (uint3));

	createTrianglesAndVertices3D
		<<<(contour_size/512)+1, 512>>> (mesh->devV, mesh->devN, 
																		 mesh->devT, 
																		 deviceNbVerticesCube, 
																		 deviceNbTrianglesCube, 
																		 contour_values, 
																		 contour_indices, 
																		 contour_neigh_indices, 
																		 contour_neigh_morpho_centroids, 
																		 contour_size, 
																		 res, minC, cellSize, 
																		 isovalue, invalidIsovalue);

	//	mesh->devT = deviceTriangles;
	//	mesh->devV = deviceVertices;
	//	mesh->devN = deviceNormals;

	//	cudaMemcpy (vertices, deviceVertices, nbVertices*sizeof (float3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy (normals, deviceNormals, nbVertices*sizeof (float3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy (triangles, deviceTriangles, nbTriangles*sizeof (uint3), cudaMemcpyDeviceToHost);
	//	mesh->T = (unsigned int*) triangles;
	//	mesh->V = (float*) vertices;
	//	mesh->N = (float*) normals;

	//
	//	cudaFree (deviceNonEmptyCube);
	//	cudaFree (deviceIndex);
	//	cudaFree (deviceCompactNonEmptyCubeScan);
	cudaFree (deviceNbVerticesCube);
	cudaFree (deviceNbTrianglesCube);
	//	cudaFree (deviceVertices);
	//	cudaFree (deviceTriangles);
}

__global__ void computeIndex3DByBit (unsigned char * voxelGrid, unsigned char * index, 
																		 unsigned int * nonEmptyCube, 
																		 uint3 res, 
																		 unsigned char isovalue, float invalidIsovalue){

	unsigned int idX = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int idY = blockIdx.y * blockDim.y + threadIdx.y; 
	unsigned int idZ = blockIdx.z * blockDim.z + threadIdx.z; 
	unsigned int i = (res.x - 1)*(res.y - 1)*idZ + (res.x - 1)*idY + idX;

	if (idX >= (res.x - 1) || idY >= (res.y - 1) || idZ >= (res.z - 1))
		return;

	unsigned char ind = 0;
	unsigned char v[8];

	v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
	v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
	v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
	v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
	v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
	v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

	//	if(v[0] >= isovalue) ind |= 1;
	//	if(v[1] >= isovalue) ind |= 2;
	//	if(v[2] >= isovalue) ind |= 4;
	//	if(v[3] >= isovalue) ind |= 8;
	//	if(v[4] >= isovalue) ind |= 16;
	//	if(v[5] >= isovalue) ind |= 32;
	//	if(v[6] >= isovalue) ind |= 64;
	//	if(v[7] >= isovalue) ind |= 128;

	if(v[0] <= isovalue) ind |= 1;
	if(v[1] <= isovalue) ind |= 2;
	if(v[2] <= isovalue) ind |= 4;
	if(v[3] <= isovalue) ind |= 8;
	if(v[4] <= isovalue) ind |= 16;
	if(v[5] <= isovalue) ind |= 32;
	if(v[6] <= isovalue) ind |= 64;
	if(v[7] <= isovalue) ind |= 128;

	index[i] = ind;
	nonEmptyCube[i] = (ind != 0 && ind != 255) ? 1 : 0;
}

__global__ void compactCubes3DByBit (unsigned char * index, unsigned int * nonEmptyCubeScan,
																		 unsigned int * compactNonEmptyCubeScan, unsigned int indexSize){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < indexSize)
		if (index[i] != 0 && index[i] != 255)
			compactNonEmptyCubeScan[nonEmptyCubeScan[i]] = i;
}

__global__ void countTrianglesAndVertices3DByBit (unsigned int * nbVerticesCube, unsigned int * nbTrianglesCube, 
																									unsigned int * compactNonEmptyCubeScan, unsigned char * index, 
																									unsigned int nbNonEmptyCubes, uint3 res) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbNonEmptyCubes) {		
		unsigned int indice = compactNonEmptyCubeScan[i];

		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int remainder, idX, idY, idZ;
		idZ = indice/(resM1.x*resM1.y);
		remainder = indice % (resM1.x*resM1.y);
		idY = remainder/resM1.x;
		idX = remainder % resM1.x;

		int nbVerts = 0;
		if ((edgeTable[index[indice]] & 1) && (idX != resM2.x)) nbVerts++; 
		if ((edgeTable[index[indice]] & 8) && (idY != resM2.y)) nbVerts++; 
		if ((edgeTable[index[indice]] & 256) && (idZ != resM2.z)) nbVerts++;
		nbVerticesCube[i] = nbVerts;

		nbTrianglesCube[i] = (idX != resM2.x && idY != resM2.y && idZ != resM2.z) ? numTriTable[index[indice]] : 0;
	}

	//	unsigned int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	//	if(i<nbNonEmptyCubes){		
	//		unsigned int indice = compactNonEmptyCubeScan[i];
	//		unsigned int x = indice/((nbY+1) * (nbZ+1));
	//		unsigned int y = (indice - x * (nbY+1) * (nbZ+1)) / (nbZ+1);
	//		unsigned int z = indice - x * (nbY+1) * (nbZ+1) - y * (nbZ+1);
	//
	//		int nbVerts = 0;
	//		if ((edgeTable[index[indice]] & 1) && (x != nbX)) nbVerts++; 
	//		if ((edgeTable[index[indice]] & 8) && (y != nbY)) nbVerts++; 
	//		if ((edgeTable[index[indice]] & 256) && (z != nbZ)) nbVerts++;
	//		nbVerticesCube[i] = nbVerts;
	//
	//		nbTrianglesCube[i] = (x != nbX && y != nbY && z != nbZ) ? numTriTable[index[indice]] : 0;
	//	}
}

__global__ void createTrianglesAndVertices3DByBit (float3 * vertices, uint3 * triangles, unsigned char * voxelGrid, 
																									 unsigned int * nbVerticesCube, unsigned int * nbTrianglesCube, 
																									 unsigned int * compactNonEmptyCubeScan, unsigned int * nonEmptyCubeScan, 
																									 unsigned char * index, unsigned int nbNonEmptyCubes, 
																									 uint3 res, float3 minC, float cellSize, 
																									 float isovalue, float invalidIsovalue) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < nbNonEmptyCubes) {
		uint3 resM1 = make_uint3 (res.x - 1, res.y - 1, res.z - 1);
		uint3 resM2 = make_uint3 (res.x - 2, res.y - 2, res.z - 2);

		unsigned int ind = compactNonEmptyCubeScan[i];
		unsigned int remainder, idX, idY, idZ;
		idZ = ind/(resM1.x*resM1.y);
		remainder = ind % (resM1.x*resM1.y);
		idY = remainder/resM1.x;
		idX = remainder % resM1.x;
		unsigned char indexCube = index[ind];

		unsigned int vert[12];
		float3 point = make_float3 (minC.x + idX * cellSize, minC.y + idY * cellSize, minC.z + idZ * cellSize);
		unsigned int indice = 0;
		short edge = edgeTable[indexCube];

		float v[8];
		unsigned int id[8];

		id[0] = idX 				+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[1] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*idZ;
		id[2] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[3] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*idZ;
		id[4] = idX					+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[5] = idX + 1			+ resM1.x*idY				+ resM1.x*resM1.y*(idZ + 1);
		id[6] = idX + 1			+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);
		id[7] = idX					+ resM1.x*(idY + 1)	+ resM1.x*resM1.y*(idZ + 1);

		v[0] = voxelGrid[idX 				+ res.x*idY				+ res.x*res.y*idZ];
		v[1] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*idZ];
		v[2] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[3] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*idZ];
		v[4] = voxelGrid[idX				+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[5] = voxelGrid[idX + 1		+ res.x*idY				+ res.x*res.y*(idZ + 1)];
		v[6] = voxelGrid[idX + 1		+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];
		v[7] = voxelGrid[idX				+ res.x*(idY + 1)	+ res.x*res.y*(idZ + 1)];

		bool valid = false;
		if ((edge & 1) && (idX != resM2.x)) {
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[1]) > invalidIsovalue);
			vert[0] = nbVerticesCube[i] + indice;
			vertices[vert[0]] = findIntersection(point, v[0], 
																					 point + make_float3(cellSize, 0, 0), v[1], 
																					 isovalue);
			indice ++;

		}
		if ((edge & 2) && (idX != resM2.x) && (idY != resM2.y)) {
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[2]) > invalidIsovalue);
			vert[1] = nbVerticesCube[nonEmptyCubeScan[id[1]]];			
			if (edgeTable[index[id[1]]]&1 && (idX+1)!=resM2.x) {
				vert[1] ++;
			}
		}
		if ((edge & 4) && (idX != resM2.x) && (idY != resM2.y)) {
			valid |= (fabs (v[2]) > invalidIsovalue) || (fabs (v[3]) > invalidIsovalue);
			vert[2] = nbVerticesCube[nonEmptyCubeScan[id[3]]];
		}
		if ((edge & 8) && (idY != resM2.y)) {
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[0]) > invalidIsovalue);
			vert[3] = nbVerticesCube[i] + indice;
			vertices[vert[3]] = findIntersection(point + make_float3(0, cellSize, 0), v[3], 
																					 point, v[0], 
																					 isovalue);
			indice ++;
		}
		if ((edge & 16) && (idZ != resM2.z) && (idX != resM2.x)){
			valid = valid || (fabs (v[4]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			vert[4] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
		}

		if ((edge & 32) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[5]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			vert[5] = nbVerticesCube[nonEmptyCubeScan[id[5]]];
			if(edgeTable[index[id[5]]]&1 && (idX+1)!=resM2.x){
				vert[5] ++;
			}
		}
		if ((edge & 64) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[6]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			vert[6] = nbVerticesCube[nonEmptyCubeScan[id[7]]];
		}
		if ((edge & 128) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[7]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[7] = nbVerticesCube[nonEmptyCubeScan[id[4]]];
			if(edgeTable[index[id[4]]]&1 && idX!=resM2.x){
				vert[7] ++;
			}
		}		
		if ((edge & 256) && (idZ != resM2.z)){
			valid = valid || (fabs (v[0]) > invalidIsovalue) || (fabs (v[4]) > invalidIsovalue);
			vert[8] = nbVerticesCube[i] + indice;
			vertices[vert[8]] = findIntersection(point , v[0], 
																					 point + make_float3(0, 0, cellSize), v[4], 
																					 isovalue);
			indice ++;
		}
		unsigned int d;
		short edge1;
		if ((edge & 512) && (idX != resM2.x) && (idZ != resM2.z)){
			valid = valid || (fabs (v[1]) > invalidIsovalue) || (fabs (v[5]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[1]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY!=resM2.y)) d++; 
			vert[9] = nbVerticesCube[nonEmptyCubeScan[id[1]]] + d;
		}
		if ((edge & 1024) && (idX != resM2.x) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[2]) > invalidIsovalue) || (fabs (v[6]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[2]]];
			if(edge1 & 1 && (idX+1)!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[10] = nbVerticesCube[nonEmptyCubeScan[id[2]]] + d;
		}
		if ((edge & 2048) && (idY != resM2.y) && (idZ != resM2.z)){
			valid = valid || (fabs (v[3]) > invalidIsovalue) || (fabs (v[7]) > invalidIsovalue);
			d = 0;
			edge1 = edgeTable[index[id[3]]];
			if(edge1 & 1 && idX!=resM2.x) d++;
			if(edge1 & 8 && (idY+1)!=resM2.y) d++;
			vert[11] = nbVerticesCube[nonEmptyCubeScan[id[3]]] + d;
		}

		unsigned int tri = 0;	
		if (indexCube != 0 && indexCube != 255 && idX != resM2.x && idY != resM2.y && idZ != resM2.z){
			unsigned int nbTri = nbTrianglesCube[i];
			for (int s = 0; triTable[indexCube * 16 + s] != -1; s+=3){
				if (!valid)
					triangles[nbTri + tri] = make_uint3 (vert[triTable[indexCube * 16 + s + 2]], 
																							 vert[triTable[indexCube * 16 + s + 1]], 
																							 vert[triTable[indexCube * 16 + s]]);
				else
					triangles[nbTri + tri] = make_uint3 (UINT32_MAX, UINT32_MAX, UINT32_MAX);
				tri++;
			}
		}
	}
}

void MarchingCubesMesher::createMesh3DByBit (unsigned char * evalValuesGPU, unsigned char isovalue, 
																						 float invalidIsovalue) {
	uint3 res = make_uint3 (grid->getNbCellsX (), grid->getNbCellsY (), grid->getNbCellsZ ());
	float3 minC = make_float3 ((grid->getOrigin ())[0], (grid->getOrigin ())[1], (grid->getOrigin ())[2]);
	float cellSize = grid->getCellSizeX ();
	//	unsigned int gridSize = res.x*res.y*res.z;
	unsigned int indexSize = (res.x - 1)*(res.y - 1)*(res.z - 1);

	dim3 blockDim, gridDim;

	blockDim = dim3 (4, 4, 4);
	gridDim = dim3 (((res.x - 1)/blockDim.x)+1, ((res.y - 1)/blockDim.y)+1, 
									((res.z - 1)/blockDim.z)+1);

	unsigned char * deviceFunctionValues = evalValuesGPU;
	unsigned char * deviceIndex = NULL;
	unsigned int * deviceNonEmptyCube = NULL;
	//	unsigned int * deviceCompactNonEmptyCubeScan = NULL;
	//	unsigned int * deviceNbVerticesCube = NULL;
	//	unsigned int * deviceNbTrianglesCube = NULL;
	//	float3 * deviceVertices = NULL;
	//	uint3 * deviceTriangles = NULL;
	//	float3 * vertices = NULL;
	//	uint3 * triangles = NULL;

	//calcul de nonEmptyCube et index ///////////////////////////////////////////////////////////////////////////////
	cudaMalloc ((void**)&deviceIndex, indexSize*sizeof (unsigned char)); 
	cudaMalloc ((void**)&deviceNonEmptyCube, indexSize*sizeof (unsigned int));
	checkCUDAError ();

	computeIndex3DByBit<<<gridDim, blockDim>>> (deviceFunctionValues, deviceIndex, deviceNonEmptyCube, 
																							res, isovalue, invalidIsovalue);
	unsigned int lastValue;
	cudaMemcpy (&lastValue, deviceNonEmptyCube + indexSize - 1, 
							sizeof (unsigned int), cudaMemcpyDeviceToHost);

	std::cout << "[Marching Cube Mesher] : "<< "index computed " << indexSize 
		<< std::endl;
	//	//scan et compactage nonEmptyCube//////////////////////////////////////////////////////////////////////////
	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNonEmptyCube), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube + indexSize), 
													thrust::device_ptr<unsigned int> (deviceNonEmptyCube));
	unsigned int lastValueScan;
	cudaMemcpy (&lastValueScan, deviceNonEmptyCube + indexSize - 1, 
							sizeof(unsigned int), cudaMemcpyDeviceToHost);
	unsigned int nbNonEmptyCubes = lastValue + lastValueScan;
	std::cout << "[Marching Cube Mesher] : "<< "number of non empty cells : " 
		<< nbNonEmptyCubes << std::endl;

	//	cudaMalloc ((void**)&deviceCompactNonEmptyCubeScan, nbNonEmptyCubes * sizeof(unsigned int));
	//	checkCUDAError ();
	//
	//	compactCubes3D<<<(indexSize/512)+1, 512>>> (deviceIndex, deviceNonEmptyCube, deviceCompactNonEmptyCubeScan, indexSize);
	//
	//	//calcul nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	//	//	cudaEventRecord(start, 0);
	//	cudaMalloc ((void**)&deviceNbVerticesCube, nbNonEmptyCubes*sizeof (unsigned int));
	//	cudaMalloc ((void**)&deviceNbTrianglesCube, nbNonEmptyCubes*sizeof (unsigned int));
	//	checkCUDAError ();
	//
	//	countTrianglesAndVertices3D
	//		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceNbVerticesCube, deviceNbTrianglesCube, 
	//																				deviceCompactNonEmptyCubeScan, deviceIndex, nbNonEmptyCubes, 
	//																				res);
	//	unsigned int lastValue1;
	//	cudaMemcpy(&lastValue1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//	unsigned int lastValue2;
	//	cudaMemcpy(&lastValue2, deviceNbVerticesCube + nbNonEmptyCubes - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	//	std::cout << "[Marching Cube Mesher] : "<< "triangles and vertices counted" << std::endl;
	//
	//	//scan nbVerticesCube et nbTrianglesCube///////////////////////////////////////////////////////////////////
	//	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbTrianglesCube), 
	//													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube + nbNonEmptyCubes), 
	//													thrust::device_ptr<unsigned int> (deviceNbTrianglesCube));
	//	std::cout << "[Marching Cube Mesher] : "<< "triangles count scanned" << std::endl;
	//	thrust::exclusive_scan (thrust::device_ptr<unsigned int> (deviceNbVerticesCube), 
	//													thrust::device_ptr<unsigned int> (deviceNbVerticesCube + nbNonEmptyCubes), 
	//													thrust::device_ptr<unsigned int> (deviceNbVerticesCube));
	//	std::cout << "[Marching Cube Mesher] : "<< "vertices count scanned" << std::endl;
	//	unsigned int lastValueScan1, lastValueScan2;
	//	cudaMemcpy (&lastValueScan1, deviceNbTrianglesCube + nbNonEmptyCubes - 1, 
	//							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	//	cudaMemcpy (&lastValueScan2, deviceNbVerticesCube + nbNonEmptyCubes - 1, 
	//							sizeof (unsigned int), cudaMemcpyDeviceToHost);
	//	unsigned int nbVertices =  lastValueScan2 + lastValue2;
	//	unsigned int nbTriangles = lastValueScan1 + lastValue1;
	//	mesh->TSize = nbTriangles;
	//	mesh->VSize = nbVertices;
	//	std::cout << "[Marching Cube Mesher] : "<< "there are " << nbTriangles << " triangles and "
	//		<< nbVertices << " vertices" << std::endl;
	//
	//	//creation triangles et vertices /////////////////////////////////////////////////////////////////////////////////////////
	//	vertices = (float3 *) malloc (nbVertices*sizeof (float3));
	//	cudaMalloc ((void**)&deviceVertices, nbVertices*sizeof (float3));
	//
	//	triangles = (uint3 *) malloc (nbTriangles*sizeof (uint3));
	//	cudaMalloc ((void**)&deviceTriangles, nbTriangles*sizeof (uint3));
	//
	//	createTrianglesAndVertices3D
	//		<<<(nbNonEmptyCubes/512)+1, 512>>> (deviceVertices, deviceTriangles, 
	//																				deviceFunctionValues, deviceNbVerticesCube, 
	//																				deviceNbTrianglesCube, deviceCompactNonEmptyCubeScan, 
	//																				deviceNonEmptyCube, deviceIndex, nbNonEmptyCubes, 
	//																				res, minC, cellSize, 
	//																				isovalue, invalidIsovalue);
	//
	//	//	thrust::device_ptr<unsigned int> newEnd;
	//	//	thrust::device_ptr<unsigned int> thrustDeviceTriangles ((unsigned int *)deviceTriangles);
	//	//	newEnd = thrust::remove (thrustDeviceTriangles, 
	//	//													 thrustDeviceTriangles + 3*nbTriangles, 
	//	//													 UINT32_MAX);
	//	//
	//	//	nbTriangles = newEnd - thrustDeviceTriangles;
	//	//	nbTriangles /= 3;
	//	//	trianglesSize = nbTriangles*sizeof (uint3);
	//	//	mesh->TSize = nbTriangles;
	//	//	//	cudaEventRecord(stop, 0);
	//	//	//	cudaEventSynchronize(stop);
	//	//	//	cudaEventElapsedTime(&elapsedTime, start, stop); 
	//	//	//	cout << "temps creation vertices et triangles: " << elapsedTime << " ms" << endl;
	//	//	//	cout << "memoire allouee device : " << (functionValuesSize + indexSize + nonEmptyCubeSize + compactNonEmptyCubeScanSize + nbVerticesCubeSize + nbTrianglesCubeSize + verticesSize + trianglesSize)/(1024*1024) << " Mo" << endl;
	//	//	//	cudaEventRecord(start, 0);
	//	cudaMemcpy (vertices, deviceVertices, nbVertices*sizeof (float3), cudaMemcpyDeviceToHost);
	//	cudaMemcpy (triangles, deviceTriangles, nbTriangles*sizeof (uint3), cudaMemcpyDeviceToHost);
	//	mesh->T = (unsigned int*) triangles;
	//	mesh->V = (float*) vertices;
	//
	//	cudaFree (deviceNonEmptyCube);
	//	cudaFree (deviceIndex);
	//	cudaFree (deviceCompactNonEmptyCubeScan);
	//	cudaFree (deviceNbVerticesCube);
	//	cudaFree (deviceNbTrianglesCube);
	//	cudaFree (deviceVertices);
	//	cudaFree (deviceTriangles);
}
