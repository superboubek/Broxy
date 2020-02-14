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

#include <omp.h>
#define GET_TIME  1e3*omp_get_wtime

#include <eigen3/Eigen/Dense>

#include "Decimator.h"

double c_qem_timing;
unsigned int c_qem_counter;

// save diagnostic state
#pragma GCC diagnostic push 

// turn off the specific warning. Can also use "-Wall"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"


// stuff to define the mesh
#include <vcg/complex/complex.h>

// io
#include <wrap/io_trimesh/import.h>

// local optimization
#include <vcg/complex/algorithms/local_optimization.h>
#include <vcg/complex/algorithms/local_optimization/tri_edge_collapse_quadric.h>

// turn the warnings back on
#pragma GCC diagnostic pop

// disable "unused-parameter" warning only
#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wunused-parameter"

using namespace vcg;
using namespace tri;

//-----------------------------------------------------------------------------
// Quadratic Programming :
// a) using FORTRAN routine
// b) defintion of a workspace utility class
//-----------------------------------------------------------------------------
#define MMAX 1000
#define NMAX 3
#define MNN (MMAX + NMAX + NMAX)
#define LWAR (3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX + 1)
#define LIWAR NMAX

// the fortran routine
extern "C" int ql0001_ (int *m, int *me, int *mmax, int *
												n, int *nmax, int *mnn, double *c__, double *d__, 
												double *a, double *b, double *xl, double *xu, 
												double *x, double *u, int *iout, int *ifail, int *
												iprint, double *war, int *lwar, int *iwar, int *liwar,
												double *eps);

class QPWorkspace {
	public:
		QPWorkspace () {
			qp_c_ = new double [NMAX*NMAX];
			qp_d_ = new double [NMAX];
			qp_a_ = new double [MMAX*NMAX];
			qp_b_ = new double [MMAX];
			qp_xl_ = new double [NMAX];
			qp_xu_ = new double [NMAX];
			qp_x_ = new double [NMAX];
			qp_u_ = new double [MNN];
			qp_war_ = new double [LWAR];
			qp_iwar_ = new int [LIWAR];
			qp_a_vec_ = new double [3*MMAX];
			qp_b_vec_ = new double [MMAX];
		}
		~QPWorkspace () {
			delete qp_c_;
			delete qp_d_;
			delete qp_a_;
			delete qp_b_;
			delete qp_xl_;
			delete qp_xu_;
			delete qp_x_;
			delete qp_u_;
			delete qp_war_;
			delete qp_iwar_;
			delete qp_a_vec_;
			delete qp_b_vec_;
		}
		inline double * qp_c () { return qp_c_; }
		inline double * qp_d () { return qp_d_; }
		inline double * qp_a () { return qp_a_; }
		inline double * qp_b () { return qp_b_; }
		inline double * qp_xl () { return qp_xl_; }
		inline double * qp_xu () { return qp_xu_; }
		inline double * qp_x () { return qp_x_; }
		inline double * qp_u () { return qp_u_; }
		inline double * qp_war () { return qp_war_; }
		inline int * qp_iwar () { return qp_iwar_; }
		inline double * qp_a_vec () { return qp_a_vec_; }
		inline double * qp_b_vec () { return qp_b_vec_; }
	private:
		double * qp_c_;
		double * qp_d_;
		double * qp_a_;
		double * qp_b_;
		double * qp_xl_;
		double * qp_xu_;
		double * qp_x_;
		double * qp_u_;
		double * qp_war_;
		int * qp_iwar_;
		double * qp_a_vec_;
		double * qp_b_vec_;
};

/**********************************************************
	Mesh Classes for Quadric Edge collapse based simplification

	For edge collpases we need verteses with:
	- V->F adjacency
	- per vertex incremental mark
	- per vertex Normal


	Moreover for using a quadric based collapse the vertex class
	must have also a Quadric member Q();
	Otherwise the user have to provide an helper function object
	to recover the quadric.

 ******************************************************/
// The class prototypes.
class MyVertex;
class MyEdge;
class MyFace;

struct MyUsedTypes: public UsedTypes<Use<MyVertex>::AsVertexType,Use<MyEdge>::AsEdgeType,Use<MyFace>::AsFaceType>{};

class MyVertex  : public Vertex< MyUsedTypes,
	vertex::VFAdj,
	vertex::Coord3f,
	vertex::Normal3f,
	vertex::Mark,
	vertex::BitFlags  >{
		public:
			vcg::math::Quadric<double> &Qd() {return q;}
			bool is_feature () { return is_feature_; }
			void set_is_feature (bool is_feature) { is_feature_ = is_feature; }
		private:
			math::Quadric<double> q;
			bool is_feature_;
	};

class MyEdge : public Edge< MyUsedTypes> {};

typedef BasicVertexPair<MyVertex> VertexPair;

class MyFace    : public Face< MyUsedTypes,
	face::VFAdj,
	face::VertexRef,
	face::BitFlags > {};

// the main mesh class
class MyMesh    : public vcg::tri::TriMesh<std::vector<MyVertex>, std::vector<MyFace> > {};
typedef typename MyMesh::ScalarType ScalarType;
typedef typename MyMesh::CoordType CoordType;
typedef MyMesh::VertexType::EdgeType EdgeType;
typedef typename MyMesh::VertexIterator VertexIterator;
typedef typename MyMesh::VertexPointer VertexPointer;
typedef typename MyMesh::FaceIterator FaceIterator;
typedef typename MyMesh::FacePointer FacePointer;

class MyTriEdgeCollapseQuadricParameter : public TriEdgeCollapseQuadricParameter {
	public:
		bool use_linear_constraints_;
		float max_edge_length_;
		MorphoGraphics::Vec3f bbox_min_;
		MorphoGraphics::Vec3f bbox_max_;
		MorphoGraphics::Vec3<unsigned int> res_;
		float cell_size_;
		char * scale_grid_;
		float w_qem_;
		float w_edge_length_;
		bool use_features_;
		QPWorkspace * qp_ws_;
};

class CQEMEdgeCollapse: public vcg::tri::TriEdgeCollapseQuadric< MyMesh, VertexPair, CQEMEdgeCollapse, QInfoStandard<MyVertex>  > {
	public:
		typedef  vcg::tri::TriEdgeCollapseQuadric< MyMesh,  VertexPair, CQEMEdgeCollapse, QInfoStandard<MyVertex>  > TECQ;
		CoordType _c_qem_x;
		bool _is_feasible;
		bool _is_optimal;
		bool _is_bounded;
		double _quadric_error;

		inline CQEMEdgeCollapse (const VertexPair &p, int i, BaseParameterClass *pp)/* :TECQ(p,i,pp)*/ {
			this->localMark = i;
			this->pos = p;
			this->_priority = ComputePriority (pp, _is_feasible, _is_optimal, _is_bounded, _c_qem_x);
		}

		static void InitQuadric(MyMesh &m,BaseParameterClass *_pp);
		static void Init(MyMesh &m, HeapType &h_ret, BaseParameterClass *_pp);
		void Execute(MyMesh &m, BaseParameterClass *_pp);
		bool IsFeasible(BaseParameterClass *_pp);
		ScalarType ComputePriority (BaseParameterClass *_pp);
		ScalarType ComputePriority (BaseParameterClass *_pp, 
																bool & is_feasible, bool & is_optimal, bool & is_bounded, 
																CoordType & c_qem_x);
		CoordType ComputeMinimal ();
		CoordType ComputeMinimal (BaseParameterClass *_pp);
		CoordType ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints);
		CoordType ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, bool & is_feasible, bool & is_optimal, bool & is_bounded, bool & rt);
		CoordType ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, bool & is_feasible, bool & is_optimal, bool & is_bounded, bool & rt, ScalarType & objective);
		void SolveQP (Eigen::Matrix3d & c, Eigen::Vector3d & d, double & d0, Eigen::Matrix3d & omega,
									Eigen::Vector3d & sol, bool use_linear_constraints, 
									bool & is_feasible, bool & is_optimal, bool & is_bounded, 
									QPWorkspace * qp_ws);
		CoordType ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, bool & is_feasible, bool & is_optimal, bool & is_bounded, 
															bool & rt, ScalarType & objective, bool print_debug);
		void UpdateHeap(HeapType & h_ret,BaseParameterClass *_pp);
};

void CQEMEdgeCollapse::Execute(MyMesh &m, BaseParameterClass *_pp) {
	//			QParameter *pp=(QParameter *)_pp;
	CoordType newPos;
	newPos = _c_qem_x;
	//	std::cout << "_c_qem_x : " << _c_qem_x[0] << " " << _c_qem_x[1] << " " << _c_qem_x[2] << std::endl;
	//	std::cout << "v0 : " << this->pos.V (0)->P ()[0] << " " << this->pos.V (0)->P ()[1] << " " << this->pos.V (0)->P ()[1] << std::endl;
	//	typename MyMesh::VertexType * v[2];
	//	v[0] = this->pos.V(0);
	//	v[1] = this->pos.V(1);
	QH::Qd(this->pos.V(1))+=QH::Qd(this->pos.V(0));

	typename MyMesh::VertexType * v[2];
	v[0] = this->pos.V(0);
	v[1] = this->pos.V(1);

	bool is_feature[2];
	is_feature[0] = v[0]->is_feature ();
	is_feature[1] = v[1]->is_feature ();

	v[1]->set_is_feature (is_feature[0] || is_feature[1]);

	EdgeCollapser<MyMesh,VertexPair>::Do (m, this->pos, newPos); // v0 is deleted and v1 take the new position
}

bool CQEMEdgeCollapse::IsFeasible (BaseParameterClass *_pp){
	//	QParameter *pp=(QParameter *)_pp;
	//	if(!pp->PreserveTopology) return true;
	//
	//	bool res = ( EdgeCollapser<MyMesh, VertexPair>::LinkConditions(this->pos) );
	//	if(!res) ++( TEC::FailStat::LinkConditionEdge() );
	//

	QParameter *pp=(QParameter *)_pp;
	ScalarType error;
	typename vcg::face::VFIterator<FaceType> x;
	std::vector<CoordType> on; // original normals
	typename MyMesh::VertexType * v[2];
	v[0] = this->pos.V(0);
	v[1] = this->pos.V(1);

	if(pp->NormalCheck){ // Compute maximal normal variation
		// store the old normals for non-collapsed face in v0
		for(x.F() = v[0]->VFp(), x.I() = v[0]->VFi(); x.F()!=0; ++x )	 // for all faces in v0		
			if(x.F()->V(0)!=v[1] && x.F()->V(1)!=v[1] && x.F()->V(2)!=v[1] ) // skip faces with v1
				on.push_back(NormalizedTriangleNormal(*x.F()));
		// store the old normals for non-collapsed face in v1
		for(x.F() = v[1]->VFp(), x.I() = v[1]->VFi(); x.F()!=0; ++x )	 // for all faces in v1	
			if(x.F()->V(0)!=v[0] && x.F()->V(1)!=v[0] && x.F()->V(2)!=v[0] ) // skip faces with v0
				on.push_back(NormalizedTriangleNormal(*x.F()));
	}

	//// Move the two vertexe  into new position (storing the old ones)
	CoordType OldPos0=v[0]->P();
	CoordType OldPos1=v[1]->P();
	bool is_feasible = _is_feasible, is_optimal = _is_optimal, is_bounded = _is_bounded;
	//	bool rt;
	ScalarType objective;
	//			if(pp->OptimalPlacement) { 
	//				v[0]->P() = ComputeMinimal(is_feasible, is_optimal, is_bounded, rt, objective); v[1]->P()=v[0]->P();
	//			} else  {
	//				v[0]->P() = v[1]->P();
	//			}
	v[0]->P () = _c_qem_x;
	//// Rescan faces and compute quality and difference between normals
	int i;
	double ndiff,MinCos  = 1e100; // minimo coseno di variazione di una normale della faccia 
	// (e.g. max angle) Mincos varia da 1 (normali coincidenti) a
	// -1 (normali opposte);
	//	double qt, MinQual = 1e100;
	CoordType nn;
	for(x.F() = v[0]->VFp(), x.I() = v[0]->VFi(),i=0; x.F()!=0; ++x )	// for all faces in v0		
		if(x.F()->V(0)!=v[1] && x.F()->V(1)!=v[1] && x.F()->V(2)!=v[1] )		// skip faces with v1
		{
			if(pp->NormalCheck){
				nn=NormalizedTriangleNormal(*x.F());
				ndiff=nn.dot(on[i++]);
				if(ndiff<MinCos) MinCos=ndiff;
			}
		}
	for(x.F() = v[1]->VFp(), x.I() = v[1]->VFi(),i=0; x.F()!=0; ++x )		// for all faces in v1	
		if(x.F()->V(0)!=v[0] && x.F()->V(1)!=v[0] && x.F()->V(2)!=v[0] )			// skip faces with v0
		{
			if(pp->NormalCheck){
				nn=NormalizedTriangleNormal(*x.F());
				ndiff=nn.dot(on[i++]);
				if(ndiff<MinCos) MinCos=ndiff;
			}
		}
	//	double QuadErr = objective;
	bool res;
	if(pp->NormalCheck) 
		res = (MinCos >= 0.0) && is_feasible && is_optimal && is_bounded;
	else
		res = is_feasible && is_optimal && is_bounded;

	//Rrestore old position of v0 and v1
	v[0]->P()=OldPos0;
	v[1]->P()=OldPos1;

	double edgeLength = sqrt (SquaredDistance (v[0]->P (), v[1]->P ()));

	MorphoGraphics::Vec3f v_0 (v[0]->P ()[0], v[0]->P ()[1], v[0]->P ()[2]);
	MorphoGraphics::Vec3f v_1 (v[1]->P ()[0], v[1]->P ()[1], v[1]->P ()[2]);
	MorphoGraphics::Vec3f bbox_min = ((MyTriEdgeCollapseQuadricParameter *)pp)->bbox_min_;
	MorphoGraphics::Vec3f bbox_max = ((MyTriEdgeCollapseQuadricParameter *)pp)->bbox_max_;
	MorphoGraphics::Vec3<unsigned int> res_grid = ((MyTriEdgeCollapseQuadricParameter *)pp)->res_;
	float cell_size = ((MyTriEdgeCollapseQuadricParameter *)pp)->cell_size_;
	char * scale_grid = ((MyTriEdgeCollapseQuadricParameter *)pp)->scale_grid_;

	MorphoGraphics::Vec3f grid_coord_0 = (v_0 - bbox_min)/cell_size;
	MorphoGraphics::Vec3f grid_coord_1 = (v_1 - bbox_min)/cell_size;
	MorphoGraphics::Vec3<int> id_0 (min (max (floor (grid_coord_0[0]), 0.f), (float)res_grid[0]), 
																	min (max (floor (grid_coord_0[1]), 0.f), (float)res_grid[1]), 
																	min (max (floor (grid_coord_0[2]), 0.f), (float)res_grid[2]));
	MorphoGraphics::Vec3<int> id_1 (min (max (floor (grid_coord_1[0]), 0.f), (float)res_grid[0]), 
																	min (max (floor (grid_coord_1[1]), 0.f), (float)res_grid[1]), 
																	min (max (floor (grid_coord_1[2]), 0.f), (float)res_grid[2]));
	float max_edge_length_0 = scale_grid[id_0[0] + id_0[1]*res_grid[0] 
		+ id_0[2]*res_grid[0]*res_grid[1]];
	float max_edge_length_1 = scale_grid[id_1[0] + id_1[1]*res_grid[0] 
		+ id_1[2]*res_grid[0]*res_grid[1]];
	float min_max_edge_length = min (max_edge_length_0, max_edge_length_1);
	float max_edge_length_alpha = ((MyTriEdgeCollapseQuadricParameter *)pp)->max_edge_length_; // TODO: change variable name


	if (edgeLength > max_edge_length_alpha*min_max_edge_length*cell_size && 
			((MyTriEdgeCollapseQuadricParameter *)pp)->use_linear_constraints_)
		res = false;

	return res;
}

ScalarType CQEMEdgeCollapse::ComputePriority (BaseParameterClass *_pp) {
	CoordType c_qem_x;	
	bool is_feasible, is_optimal, is_bounded;
	return ComputePriority (_pp, is_feasible, is_optimal, is_bounded, c_qem_x);
}

ScalarType CQEMEdgeCollapse::ComputePriority (BaseParameterClass *_pp, 
																							bool & is_feasible, bool & is_optimal, bool & is_bounded, 
																							CoordType & c_qem_x) {
	QParameter *pp=(QParameter *)_pp;
	ScalarType error = 0;
	typename vcg::face::VFIterator<FaceType> x;
	std::vector<CoordType> on; // original normals
	typename MyMesh::VertexType * v[2];
	v[0] = this->pos.V(0);
	v[1] = this->pos.V(1);

	if(pp->NormalCheck){ // Compute maximal normal variation
		// store the old normals for non-collapsed face in v0
		for(x.F() = v[0]->VFp(), x.I() = v[0]->VFi(); x.F()!=0; ++x )	 // for all faces in v0		
			if(x.F()->V(0)!=v[1] && x.F()->V(1)!=v[1] && x.F()->V(2)!=v[1] ) // skip faces with v1
			{
				on.push_back(NormalizedTriangleNormal(*x.F()));
			}
		// store the old normals for non-collapsed face in v1
		for(x.F() = v[1]->VFp(), x.I() = v[1]->VFi(); x.F()!=0; ++x )	 // for all faces in v1	
			if(x.F()->V(0)!=v[0] && x.F()->V(1)!=v[0] && x.F()->V(2)!=v[0] ) // skip faces with v0
			{
				on.push_back(NormalizedTriangleNormal(*x.F()));
			}
	}

	//// Move the two vertexe  into new position (storing the old ones)
	CoordType OldPos0=v[0]->P();
	CoordType OldPos1=v[1]->P();
	bool rt;
	ScalarType objective;
	if(pp->OptimalPlacement) { 
		c_qem_x = ComputeMinimal (pp, ((MyTriEdgeCollapseQuadricParameter *)pp)->use_linear_constraints_, is_feasible, is_optimal, is_bounded, rt, objective);
		v[0]->P() = c_qem_x; v[1]->P()=v[0]->P();
	} else  {
		v[0]->P() = v[1]->P();
	}
	//// Rescan faces and compute quality and difference between normals
	int i = 0;
	double ndiff,MinCos  = 1e100; // minimo coseno di variazione di una normale della faccia 
	// (e.g. max angle) Mincos varia da 1 (normali coincidenti) a
	// -1 (normali opposte);
	double qt,   MinQual = 1e100;
	CoordType nn;
	for(x.F() = v[0]->VFp(), x.I() = v[0]->VFi(),i=0; x.F()!=0; ++x )	// for all faces in v0		
		if(x.F()->V(0)!=v[1] && x.F()->V(1)!=v[1] && x.F()->V(2)!=v[1] )		// skip faces with v1
		{
			if(pp->NormalCheck){
				nn=NormalizedTriangleNormal(*x.F());
				ndiff=nn.dot(on[i++]);
				if(ndiff<MinCos) MinCos=ndiff;
			}

			if(pp->QualityCheck){
				qt= QualityFace(*x.F());
				if(qt<MinQual) MinQual=qt;
			}
		}

	for(x.F() = v[1]->VFp(), x.I() = v[1]->VFi(); x.F()!=0; ++x )		// for all faces in v1	
		if(x.F()->V(0)!=v[0] && x.F()->V(1)!=v[0] && x.F()->V(2)!=v[0] )			// skip faces with v0
		{
			if(pp->NormalCheck){
				nn=NormalizedTriangleNormal(*x.F());
				ndiff=nn.dot(on[i++]);
				if(ndiff<MinCos) MinCos=ndiff;
			}

			if(pp->QualityCheck){
				qt= QualityFace(*x.F());
				if(qt<MinQual) MinQual=qt;
			}
		}

	QuadricType qq=QH::Qd(v[0]);
	qq+=QH::Qd(v[1]);
	//	Point3d tpd=Point3d::Construct(v[1]->P());
	double QuadErr = pp->ScaleFactor*objective;

	// All collapses involving triangles with quality larger than <QualityThr> has no penalty;
	if(MinQual>pp->QualityThr) MinQual=pp->QualityThr;

	if(QuadErr<pp->QuadricEpsilon) QuadErr=pp->QuadricEpsilon;
	this->_quadric_error = objective;

	QuadErr = !is_feasible ? 10e10 : QuadErr;
	if(pp->NormalCheck) QuadErr = MinCos < 0 ? 10e10 : QuadErr;

	error = (ScalarType)(QuadErr / MinQual);

	//Rrestore old position of v0 and v1
	v[0]->P()=OldPos0;
	v[1]->P()=OldPos1;

	double edgeLength = sqrt (SquaredDistance (v[0]->P (), v[1]->P ()));
	//	error *= (2.f + edgeLength)/(1.f + exp (-(edgeLength/0.07f)));
	//	if (edgeLength > ((MyTriEdgeCollapseQuadricParameter *)pp)->max_edge_length_)
	//		error = 10e10;
	MorphoGraphics::Vec3f v_0 (v[0]->P ()[0], v[0]->P ()[1], v[0]->P ()[2]);
	MorphoGraphics::Vec3f v_1 (v[1]->P ()[0], v[1]->P ()[1], v[1]->P ()[2]);
	MorphoGraphics::Vec3f bbox_min = ((MyTriEdgeCollapseQuadricParameter *)pp)->bbox_min_;
	MorphoGraphics::Vec3f bbox_max = ((MyTriEdgeCollapseQuadricParameter *)pp)->bbox_max_;
	MorphoGraphics::Vec3<unsigned int> res_grid = ((MyTriEdgeCollapseQuadricParameter *)pp)->res_;
	float cell_size = ((MyTriEdgeCollapseQuadricParameter *)pp)->cell_size_;
	char * scale_grid = ((MyTriEdgeCollapseQuadricParameter *)pp)->scale_grid_;

	MorphoGraphics::Vec3f grid_coord_0 = (v_0 - bbox_min)/cell_size;
	MorphoGraphics::Vec3f grid_coord_1 = (v_1 - bbox_min)/cell_size;
	MorphoGraphics::Vec3<int> id_0 (min (max (floor (grid_coord_0[0]), 0.f), (float)res_grid[0]), 
																	min (max (floor (grid_coord_0[1]), 0.f), (float)res_grid[1]), 
																	min (max (floor (grid_coord_0[2]), 0.f), (float)res_grid[2]));
	MorphoGraphics::Vec3<int> id_1 (min (max (floor (grid_coord_1[0]), 0.f), (float)res_grid[0]), 
																	min (max (floor (grid_coord_1[1]), 0.f), (float)res_grid[1]), 
																	min (max (floor (grid_coord_1[2]), 0.f), (float)res_grid[2]));
	float max_edge_length_0 = scale_grid[id_0[0] + id_0[1]*res_grid[0] 
		+ id_0[2]*res_grid[0]*res_grid[1]];
	float max_edge_length_1 = scale_grid[id_1[0] + id_1[1]*res_grid[0] 
		+ id_1[2]*res_grid[0]*res_grid[1]];
	float min_max_edge_length = min (max_edge_length_0, max_edge_length_1);
	float max_edge_length_alpha = ((MyTriEdgeCollapseQuadricParameter *)pp)->max_edge_length_; // TODO: change variable name

	if (!((MyTriEdgeCollapseQuadricParameter *)pp)->use_features_)
		if (edgeLength > max_edge_length_alpha*min_max_edge_length*cell_size)
			error = 10e10;

	//	QParameter *q_pp=(QParameter *)_pp;
	//	CoordType newPos;
	//			ScalarType objective;
	//			bool is_feasible, is_optimal, is_bounded, rt;

	this->_priority = error;
	return this->_priority;
}

CoordType CQEMEdgeCollapse::ComputeMinimal () {
	CoordType x;
	std::cout << "[CQEMEdgeCollapse] : warning !!!!! using the dummy ComputeMinimal method." << std::endl;
	return x;
}

CoordType CQEMEdgeCollapse::ComputeMinimal (BaseParameterClass *_pp) {
	bool use_linear_constraints = true;
	bool is_feasible, is_optimal, is_bounded, rt;
	CoordType x = ComputeMinimal (_pp, use_linear_constraints, is_feasible, is_optimal, is_bounded, rt);
	return x;
}

CoordType CQEMEdgeCollapse::ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints) {
	bool is_feasible, is_optimal, is_bounded, rt;
	CoordType x = ComputeMinimal (_pp, use_linear_constraints, is_feasible, is_optimal, is_bounded, rt);
	return x;
}

CoordType CQEMEdgeCollapse::ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, 
																						bool & is_feasible, bool & is_optimal, bool & is_bounded, bool & rt) {
	bool print_debug = false;
	ScalarType obj;
	CoordType x = ComputeMinimal (_pp, use_linear_constraints, is_feasible, is_optimal, is_bounded, rt, obj, print_debug);	
	return x;
}

CoordType CQEMEdgeCollapse::ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, 
																						bool & is_feasible, bool & is_optimal, bool & is_bounded, bool & rt, ScalarType & objective) {
	bool print_debug = false;
	CoordType x = ComputeMinimal (_pp, use_linear_constraints, is_feasible, is_optimal, is_bounded, rt, objective, print_debug);	
	return x;
}

CoordType CQEMEdgeCollapse::ComputeMinimal (BaseParameterClass *_pp, bool use_linear_constraints, 
																						bool & is_feasible, bool & is_optimal, bool & is_bounded, bool & rt, 
																						ScalarType & objective, bool print_debug) {

	MyTriEdgeCollapseQuadricParameter *pp = (MyTriEdgeCollapseQuadricParameter *)_pp;
	//	double time1, time2;
	typename MyMesh::VertexType * v[2];
	v[0] = this->pos.V(0);
	v[1] = this->pos.V(1);

	// Compute the edge Quadric by summing the vertices quadrics
	Point3<QuadricType::ScalarType> x (0.0, 0.0, 0.0);
	Point3<QuadricType::ScalarType> c_x (0.0, 0.0, 0.0);
	QuadricType q=QH::Qd(v[0]);
	q+=QH::Qd(v[1]);

	Point3<QuadricType::ScalarType> qem_regularizer;
	//	Point3<QuadricType::ScalarType> x0=Point3d::Construct(v[0]->P());
	//	Point3<QuadricType::ScalarType> x1=Point3d::Construct(v[1]->P());
	qem_regularizer.Import((v[0]->P()+v[1]->P())/2);
	//	double qvx=q.Apply(qem_regularizer);
	//	double qv0=q.Apply(x0);
	//	double qv1=q.Apply(x1);
	//			if(qv0<qvx) qem_regularizer=x0;
	//			if(qv1<qvx && qv1<qv0) qem_regularizer=x1;


	// Compute regularizer
	double lambda;
	Eigen::Vector3d regularizer;
	regularizer = Eigen::Vector3d (qem_regularizer[0], qem_regularizer[1], qem_regularizer[2]);

	// Eigen decomposition of the edge Quadric quadratic term
	double ratio_e_val = 1e-3;
	Eigen::Matrix3d d, omega;
	for (int curs = 0, j = 0; j < 3; j++)
		for (int i = j; i < 3; i++) {
			d (i, j) = q.a[curs];
			d (j, i) = d (i, j);
			curs++;
		}

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es (d);
	omega = es.eigenvectors ().transpose ();

	// Prepare the transformed optimization problem
	double max_e_val = es.eigenvalues () (2);
	//			lambda = 10.0/(SquaredDistance (v[0]->P (), v[1]->P ()));
	lambda = 1.0/(SquaredDistance (v[0]->P (), v[1]->P ()));
	//			lambda = 10*fabs (max_e_val);

	Eigen::Matrix3d t_d = Eigen::Matrix3d::Zero ();
	Eigen::Vector3d b, t_b, t_regularizer;
	//	double c;
	double t_c;
	//	c = q.c;
	b = Eigen::Vector3d (q.b[0], q.b[1], q.b[2]);
	t_regularizer = omega*regularizer;
	t_c = q.c;
	t_b = omega*b;
	for (int k = 0; k < 3; k++)	{
		double e_val_k = es.eigenvalues () (k); 
		if (fabs (e_val_k)/fabs (max_e_val) < ratio_e_val) {
			t_d (k, k) = e_val_k + lambda;
			t_b (k) = t_b (k) - lambda*2*t_regularizer (k); 
			t_c = t_c + lambda*(t_regularizer (k)*t_regularizer (k));
		} else {
			t_d (k, k) = e_val_k;
		}
	}

	// Solve the program
	Eigen::Vector3d t_sol, sol;
	SolveQP (t_d, t_b, t_c, omega, t_sol, use_linear_constraints, 
					 is_feasible, is_optimal, is_bounded, 
					 pp->qp_ws_);
	sol = (omega.transpose ())*t_sol;
	for (int i = 0; i < 3; i++) 
		c_x[i] = sol (i);

	double objective_q = q.Apply (c_x);
	objective = objective_q;

	return CoordType::Construct(c_x);
}

void CQEMEdgeCollapse::SolveQP (Eigen::Matrix3d & c, Eigen::Vector3d & d, double & d0, Eigen::Matrix3d & omega,
																Eigen::Vector3d & sol, bool use_linear_constraints, 
																bool & is_feasible, bool & is_optimal, bool & is_bounded, 
																QPWorkspace * qp_ws) {
	double time1, time2;
	typename MyMesh::VertexType * v[2];
	time1 = GET_TIME ();
	v[0] = this->pos.V(0);
	v[1] = this->pos.V(1);

	// Get the Quadratic Programming Workspace
	double * qp_a_vec = qp_ws->qp_a_vec ();
	double * qp_b_vec = qp_ws->qp_b_vec ();
	double * qp_c = qp_ws->qp_c ();
	double * qp_d = qp_ws->qp_d ();
	double * qp_a = qp_ws->qp_a ();
	double * qp_b = qp_ws->qp_b ();
	double * qp_xl = qp_ws->qp_xl ();
	double * qp_xu = qp_ws->qp_xu ();
	double * qp_x = qp_ws->qp_x ();
	double * qp_u = qp_ws->qp_u ();
	double * qp_war = qp_ws->qp_war ();
	int * qp_iwar = qp_ws->qp_iwar ();


	int curs_const = 0;
	// Constraints retrieval
	typename vcg::face::VFIterator<FaceType> fiter;
	for(fiter.F() = v[0]->VFp(), fiter.I() = v[0]->VFi(); fiter.F()!=0; ++fiter )	{		// for all faces in v0		
		if(fiter.F()->V(0)!=v[1] && fiter.F()->V(1)!=v[1] && fiter.F()->V(2)!=v[1] ) {	// skip faces with v1
			CoordType vcg_nn = NormalizedTriangleNormal(*fiter.F());
			CoordType vcg_p = fiter.F ()->V (0)->P ();
			Eigen::Vector3d nn (vcg_nn[0], vcg_nn[1], vcg_nn[2]);
			Eigen::Vector3d p (vcg_p[0], vcg_p[1], vcg_p[2]);
			Eigen::Vector3d t_nn (nn[0], nn[1], nn[2]);
			t_nn = omega*t_nn;
			qp_a_vec[3*curs_const] = t_nn[0]; 
			qp_a_vec[3*curs_const + 1] = t_nn[1]; 
			qp_a_vec[3*curs_const + 2] = t_nn[2];
			qp_b_vec[curs_const] = -nn.dot (p);
			curs_const++;
		}
	}

	for(fiter.F() = v[1]->VFp(), fiter.I() = v[1]->VFi(); fiter.F()!=0; ++fiter ) {		// for all faces in v1	
		CoordType vcg_nn = NormalizedTriangleNormal(*fiter.F());
		CoordType vcg_p = fiter.F ()->V (0)->P ();
		Eigen::Vector3d nn (vcg_nn[0], vcg_nn[1], vcg_nn[2]);
		Eigen::Vector3d p (vcg_p[0], vcg_p[1], vcg_p[2]);
		Eigen::Vector3d t_nn (nn[0], nn[1], nn[2]);
		t_nn = omega*t_nn;
		qp_a_vec[3*curs_const] = t_nn[0]; 
		qp_a_vec[3*curs_const + 1] = t_nn[1]; 
		qp_a_vec[3*curs_const + 2] = t_nn[2];
		qp_b_vec[curs_const] = -nn.dot (p);
		curs_const++;
	}

	int m = use_linear_constraints ? curs_const : 0;
	int me = 0;
	int mmax = m;
	int n = 3;
	int nmax = n;
	int mnn = m + n + n;
	int iout, ifail, iprint = 0;
	int lwar = 3*nmax*nmax/2 + 10*nmax + 2*mmax + 1;
	int liwar = n;
	double eps = 1e-15;

	qp_iwar[0] = 1;

	// Objective Set Up
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			// fortran77 by column storage
			qp_c[3*j + i] = 2*c (i, j);
		}
		qp_d[i] = d[i];
	}
	// Constraints Set Up
	for (int l = 0; l < m; l++) {
		for (int j = 0; j < 3; j++) {
			qp_a[m*j + l]	= qp_a_vec[3*l + j];
		}
		qp_b[l] = qp_b_vec[l];
	}
	for (int j = 0; j < 3; j++) {
		qp_xu[j] = 1e10;
		qp_xl[j] = -1e10;
	}

	ql0001_ (&m, &me, &mmax, &n, &nmax, &mnn, 
					 qp_c, qp_d, qp_a, qp_b, qp_xl, qp_xu, qp_x, qp_u, 
					 &iout, &ifail, &iprint, qp_war, &lwar, qp_iwar, &liwar, &eps);

	is_feasible = false;
	is_optimal = false;
	is_bounded = false;

	if (ifail == 0) {
		// All good
		is_feasible = true;
		is_optimal = true;
		is_bounded = true;
		for (int j = 0; j < 3; j++)
			sol[j] = qp_x[j];
	} else if (ifail == 5) {
		// Lenght of working array too short
		std::cout << "Siconos : length of working array is too short" << std::endl;
	}

	time2 = GET_TIME ();
	c_qem_timing += (time2 - time1);
	c_qem_counter++;
}
void CQEMEdgeCollapse::UpdateHeap(HeapType & h_ret,BaseParameterClass *_pp) {
	//	QParameter *pp=(QParameter *)_pp;
	this->GlobalMark()++;
	VertexType *v[2];
	v[0]= this->pos.V(0);
	v[1]= this->pos.V(1);
	v[1]->IMark() = this->GlobalMark();

	// First Pass to clear vertices of the 2-ring
	// Primary loop around the remaining vertex to unmark visited flags
	vcg::face::VFIterator<FaceType> vfi(v[1]);
	vcg::face::VFIterator<FaceType> vfi1;
	vcg::face::VFIterator<FaceType> vfi2;
	while (!vfi.End()) {
		vfi.V1()->ClearV();
		vfi.V2()->ClearV();
		// Secondary loops for the 2-ring neighborhood
		vfi1 = vcg::face::VFIterator<FaceType> (vfi.V1());
		vfi2 = vcg::face::VFIterator<FaceType> (vfi.V2());
		while (!vfi1.End()) {
			vfi1.V1()->ClearV();
			vfi1.V2()->ClearV();
			++vfi1;
		}
		while (!vfi2.End()) {
			vfi2.V1()->ClearV();
			vfi2.V2()->ClearV();
			++vfi2;
		}
		++vfi;
	}

	// Second Pass to store vertices of the 1-ring, 
	// add all its edges to the heap and mark all the
	// vertices
	std::vector<MyVertex*> one_ring_vertices;
	std::vector<VertexPair> two_ring_edges;
	vfi = face::VFIterator<FaceType> (v[1]);

	v[1]->SetV ();
	while (!vfi.End ()) {
		assert(!vfi.F ()->IsD ());
		if (!(vfi.V1 ()->IsV ()) && vfi.V1 ()->IsRW ()) {
			vfi.V1 ()->SetV ();
			vfi.V1 ()->IMark() = this->GlobalMark();
			two_ring_edges.push_back (VertexPair(vfi.V0(),vfi.V1()));
			//					h_ret.push_back(HeapElem(new CQEMEdgeCollapse(VertexPair(vfi.V0(),vfi.V1()), this->GlobalMark(),_pp)));
			//					std::push_heap(h_ret.begin(),h_ret.end());
			one_ring_vertices.push_back (vfi.V1 ());
		}

		if (!(vfi.V2()->IsV()) && vfi.V2()->IsRW()) {
			vfi.V2()->SetV();
			vfi.V2()->IMark() = this->GlobalMark();
			two_ring_edges.push_back (VertexPair(vfi.V0(),vfi.V2()));
			//					h_ret.push_back(HeapElem(new CQEMEdgeCollapse(VertexPair(vfi.V0(),vfi.V2()),this->GlobalMark(),_pp)));
			//					std::push_heap(h_ret.begin(),h_ret.end());
			one_ring_vertices.push_back (vfi.V2 ());
		}

		if (vfi.V1 ()->IsRW () && vfi.V2 ()->IsRW ()) {
			two_ring_edges.push_back (VertexPair(vfi.V1(),vfi.V2()));
			//					h_ret.push_back(HeapElem(new CQEMEdgeCollapse(VertexPair(vfi.V1(),vfi.V2()),this->GlobalMark(),_pp)));
			//					std::push_heap(h_ret.begin(),h_ret.end());
		}
		++vfi;
	}

	for(int i = 0; i < (int)one_ring_vertices.size (); i++) {
		MyVertex * vi = one_ring_vertices[i];
		if (!(*vi).IsD () && (*vi).IsRW ()) {
			vcg::face::VFIterator<FaceType> x;
			for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++x )
			{
				assert(x.F()->V(x.I())==&(*vi));
				if(x.V1()->IsRW() && !x.V1()->IsV()){
					x.V1()->SetV();
					two_ring_edges.push_back (VertexPair(x.V0(),x.V1()));
					//							h_ret.push_back(HeapElem(new CQEMEdgeCollapse (VertexPair(x.V0(), x.V1()),
					//																															this->GlobalMark(), _pp )));
					//							std::push_heap(h_ret.begin(),h_ret.end());
				}
				if(x.V2()->IsRW()&& !x.V2()->IsV()){
					x.V2()->SetV();
					two_ring_edges.push_back (VertexPair(x.V0(),x.V2()));
					//							h_ret.push_back(HeapElem(new CQEMEdgeCollapse (VertexPair(x.V0(),x.V2()),
					//																															this->GlobalMark(), _pp )));
					//							std::push_heap(h_ret.begin(),h_ret.end());
				}
			}
			for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++ x){
				x.V1()->ClearV();
				x.V2()->ClearV();
			}
			v[1]->SetV ();
			for(int j = 0; j < (int)one_ring_vertices.size (); j++)
				one_ring_vertices[j]->SetV ();
		}
	}

	std::vector<CQEMEdgeCollapse*> edge_collapses (two_ring_edges.size ());

	for (int k = 0; k < (int)two_ring_edges.size (); k++) {
		edge_collapses[k] = new CQEMEdgeCollapse (two_ring_edges[k],
																							this->GlobalMark(), _pp );
		//				h_ret.push_back(HeapElem(new CQEMEdgeCollapse (two_ring_edges[k],
		//																												this->GlobalMark(), _pp )));
		//				std::push_heap(h_ret.begin(),h_ret.end());
	}

	for (int k = 0; k < (int)edge_collapses.size (); k++) {
		h_ret.push_back(HeapElem(edge_collapses[k]));
		std::push_heap(h_ret.begin(),h_ret.end());
	}
}

void CQEMEdgeCollapse::InitQuadric(MyMesh &m,BaseParameterClass *_pp)
{
	QParameter *pp=(QParameter *)_pp;
	typename MyMesh::FaceIterator pf;
	typename MyMesh::VertexIterator pv;
	int j;
	QH::Init();
	//	m.ClearFlags();
	for(pv=m.vert.begin();pv!=m.vert.end();++pv)		// Azzero le quadriche
		if( ! (*pv).IsD() && (*pv).IsW())
			QH::Qd(*pv).SetZero();

	//	for(pv=m.vert.begin();pv!=m.vert.end();++pv)		// Azzero le quadriche
	//		if( !(*pv).IsD() && (*pv).IsW() ) {
	//			QuadricType q;
	//			Plane3<ScalarType,false> p;
	//			p.SetDirection((*pv).N ());
	//			p.SetOffset(p.Direction().dot((*pv).P()));
	//			q.ByPlane(p);
	//
	//			QH::Qd(*pv) = q;				// Sommo la quadrica ai vertici
	//		}

	for(pf=m.face.begin();pf!=m.face.end();++pf)
		if( !(*pf).IsD() && (*pf).IsR() )
			if((*pf).V(0)->IsR() &&(*pf).V(1)->IsR() &&(*pf).V(2)->IsR())
			{
				QuadricType q;
				Plane3<ScalarType,false> p;
				// Calcolo piano
				p.SetDirection( ( (*pf).V(1)->cP() - (*pf).V(0)->cP() ) ^  ( (*pf).V(2)->cP() - (*pf).V(0)->cP() ));

				//				Point3<ScalarType> nDir0 = Point3<ScalarType>::Construct((*pf).V(0)->N());
				//				Point3<ScalarType> nDir1 = Point3<ScalarType>::Construct((*pf).V(1)->N());
				//				Point3<ScalarType> nDir2 = Point3<ScalarType>::Construct((*pf).V(2)->N());
				//				Point3<ScalarType> nDir = nDir0 + nDir1 + nDir2;
				//				Point3<ScalarType> faceDir = ((*pf).V(1)->cP() - (*pf).V(0)->cP() ) ^  ( (*pf).V(2)->cP() - (*pf).V(0)->cP() );				
				//				nDir[0] = (faceDir.Norm ()/nDir.Norm ())*nDir[0];
				//				nDir[1] = (faceDir.Norm ()/nDir.Norm ())*nDir[1];
				//				nDir[2] = (faceDir.Norm ()/nDir.Norm ())*nDir[2];
				//				p.SetDirection(nDir);
				//			
				//				MorphoGraphics::Vec3f test_p (-0.167266, -0.699645, -0.30523);
				//				if (
				//						fabs ((*pf).V (0)->P ()[0] - test_p[0]) < 0.000001
				//						&& fabs ((*pf).V (0)->P ()[1] - test_p[1]) < 0.000001
				//						&& fabs ((*pf).V (0)->P ()[2] - test_p[2]) < 0.000001
				//						) {
				//
				//					std::cout << "test normal " << (*pf).V (0)->N ()[0] << " " << (*pf).V (0)->N ()[1] << " " << (*pf).V (0)->N ()[2]  << std::endl;
				//				}

				// Se normalizzo non dipende dall'area

				if(!pp->UseArea)
					p.Normalize();

				p.SetOffset( p.Direction().dot((*pf).V(0)->cP()));

				// Calcolo quadrica	delle facce
				q.ByPlane(p);

				for(j=0;j<3;++j)
					if( (*pf).V(j)->IsW() )
					{
						if(pp->QualityWeight)
							q*=(*pf).V(j)->Q();
						QH::Qd((*pf).V(j)) += q;				// Sommo la quadrica ai vertici
					}

				for(j=0;j<3;++j)
					if( (*pf).IsB(j) || pp->QualityQuadric )				// Bordo!
					{
						Plane3<ScalarType,false> pb;						// Piano di bordo

						// Calcolo la normale al piano di bordo e la sua distanza
						// Nota che la lunghezza dell'edge DEVE essere Normalizzata
						// poiche' la pesatura in funzione dell'area e'gia fatta in p.Direction()
						// Senza la normalize il bordo e' pesato in funzione della grandezza della mesh (mesh grandi non decimano sul bordo)
						pb.SetDirection(p.Direction() ^ ( (*pf).V1(j)->cP() - (*pf).V(j)->cP() ).normalized());
						if(  (*pf).IsB(j) ) pb.SetDirection(pb.Direction()* (ScalarType)pp->BoundaryWeight);        // amplify border planes
						else pb.SetDirection(pb.Direction()* (ScalarType)(pp->BoundaryWeight/100.0)); // and consider much less quadric for quality
						pb.SetOffset(pb.Direction().dot((*pf).V(j)->cP()));
						q.ByPlane(pb);

						if( (*pf).V (j)->IsW() )	QH::Qd((*pf).V (j)) += q;			// Sommo le quadriche
						if( (*pf).V1(j)->IsW() )	QH::Qd((*pf).V1(j)) += q;
					}
			}

	if(pp->ScaleIndependent)
	{
		vcg::tri::UpdateBounding<MyMesh>::Box(m);
		//Make all quadric independent from mesh size
		pp->ScaleFactor = 1e8*pow(1.0/m.bbox.Diag(),6); // scaling factor
		//pp->ScaleFactor *=pp->ScaleFactor ;
		//pp->ScaleFactor *=pp->ScaleFactor ;
		//printf("Scale factor =%f\n",pp->ScaleFactor );
		//printf("bb (%5.2f %5.2f %5.2f)-(%5.2f %5.2f %5.2f) Diag %f\n",m.bbox.min[0],m.bbox.min[1],m.bbox.min[2],m.bbox.max[0],m.bbox.max[1],m.bbox.max[2],m.bbox.Diag());
	}
}


void CQEMEdgeCollapse::Init(MyMesh &m, HeapType &h_ret, BaseParameterClass *_pp)
{
	QParameter *pp=(QParameter *)_pp;

	typename 	MyMesh::VertexIterator  vi;
	typename 	MyMesh::FaceIterator  pf;

	pp->CosineThr=cos(pp->NormalThrRad);

	vcg::tri::UpdateTopology<MyMesh>::VertexFace(m);
	vcg::tri::UpdateFlags<MyMesh>::FaceBorderFromVF(m);

	if(pp->FastPreserveBoundary)
	{
		for(pf=m.face.begin();pf!=m.face.end();++pf)
			if( !(*pf).IsD() && (*pf).IsW() )
				for(int j=0;j<3;++j)
					if((*pf).IsB(j))
					{
						(*pf).V(j)->ClearW();
						(*pf).V1(j)->ClearW();
					}
	}

	if(pp->PreserveBoundary)
	{
		WV().clear();
		for(pf=m.face.begin();pf!=m.face.end();++pf)
			if( !(*pf).IsD() && (*pf).IsW() )
				for(int j=0;j<3;++j)
					if((*pf).IsB(j))
					{
						if((*pf).V(j)->IsW())  {(*pf).V(j)->ClearW(); WV().push_back((*pf).V(j));}
						if((*pf).V1(j)->IsW()) {(*pf).V1(j)->ClearW();WV().push_back((*pf).V1(j));}
					}
	}

	InitQuadric(m,pp);

	// Initialize the heap with all the possible collapses
	if(IsSymmetric(pp))
	{ // if the collapse is symmetric (e.g. u->v == v->u)
		for(vi=m.vert.begin();vi!=m.vert.end();++vi)
			if(!(*vi).IsD() && (*vi).IsRW())
			{
				vcg::face::VFIterator<FaceType> x;
				for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++ x){
					x.V1()->ClearV();
					x.V2()->ClearV();
				}
				for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++x )
				{
					assert(x.F()->V(x.I())==&(*vi));
					if((x.V0()<x.V1()) && x.V1()->IsRW() && !x.V1()->IsV()){
						x.V1()->SetV();
						h_ret.push_back(HeapElem(new CQEMEdgeCollapse(VertexPair(x.V0(),x.V1()),TriEdgeCollapse< MyMesh,VertexPair,CQEMEdgeCollapse>::GlobalMark(),_pp )));
					}
					if((x.V0()<x.V2()) && x.V2()->IsRW()&& !x.V2()->IsV()){
						x.V2()->SetV();
						h_ret.push_back(HeapElem(new CQEMEdgeCollapse(VertexPair(x.V0(),x.V2()),TriEdgeCollapse< MyMesh,VertexPair,CQEMEdgeCollapse>::GlobalMark(),_pp )));
					}
				}
			}
	}
	else
	{ // if the collapse is A-symmetric (e.g. u->v != v->u)
		for(vi=m.vert.begin();vi!=m.vert.end();++vi)
			if(!(*vi).IsD() && (*vi).IsRW())
			{
				vcg::face::VFIterator<FaceType> x;
				UnMarkAll(m);
				for( x.F() = (*vi).VFp(), x.I() = (*vi).VFi(); x.F()!=0; ++ x)
				{
					assert(x.F()->V(x.I())==&(*vi));
					if(x.V()->IsRW() && x.V1()->IsRW() && !IsMarked(m,x.F()->V1(x.I()))){
						h_ret.push_back( HeapElem( new CQEMEdgeCollapse( VertexPair (x.V(),x.V1()),TriEdgeCollapse< MyMesh,VertexPair,CQEMEdgeCollapse>::GlobalMark(),_pp)));
					}
					if(x.V()->IsRW() && x.V2()->IsRW() && !IsMarked(m,x.F()->V2(x.I()))){
						h_ret.push_back( HeapElem( new CQEMEdgeCollapse( VertexPair (x.V(),x.V2()),TriEdgeCollapse< MyMesh,VertexPair,CQEMEdgeCollapse>::GlobalMark(),_pp)));
					}
				}
			}
	}
}

class Decimator::VCGOptimSession {
	public:
		VCGOptimSession (const MorphoGraphics::Mesh & mesh, int target_num_faces, 
										 float target_error, float max_edge_length, 
										 bool use_linear_constraints, bool use_features, 
										 const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
										 const MorphoGraphics::Vec3<unsigned int> & res, 
										 float cell_size, char * scale_grid, 
										 const std::vector<bool> & feature_taggs, 
										 int num_hc_iters = 3);
		~VCGOptimSession () { delete qparams.qp_ws_; };
		bool Optimize ();
		void GetMesh (MorphoGraphics::Mesh & output_mesh);
		void ExportMesh (const std::string & filename);
		int num_faces () const { return num_faces_; }
		inline int target_num_faces () { return target_num_faces_; }
		inline float target_error () { return target_error_; }
		inline float error () { return error_; }
		inline bool use_linear_constraints () { return use_linear_constraints_; }
		inline bool use_features () { return use_features_; }
		inline void set_num_faces (int num_faces) { num_faces_ = num_faces; }
		inline void set_target_num_faces (int target_num_faces) { target_num_faces_ = target_num_faces; }
		inline void set_target_error (float target_error) { target_error_ = target_error; }
		inline void set_error (float error) { error_ = error; }
		inline void set_use_linear_constraints (float use_linear_constraints) { use_linear_constraints_ = use_linear_constraints; }
		inline void set_use_features (float use_features) { use_features_ = use_features; }
	private:
		int num_faces_;
		int target_num_faces_;
		float error_;
		float target_error_;
		bool use_linear_constraints_;
		bool use_features_;
		vcg::LocalOptimization<MyMesh> * deci_session_;
		MyMesh decimated_mesh_;
		MyTriEdgeCollapseQuadricParameter qparams;
};


Decimator::VCGOptimSession::VCGOptimSession (const MorphoGraphics::Mesh & mesh, 
																						 int target_num_faces, 
																						 float target_error, 
																						 float max_edge_length, 
																						 bool use_linear_constraints, 
																						 bool use_features, 
																						 const MorphoGraphics::Vec3f & bbox_min, 
																						 const MorphoGraphics::Vec3f & bbox_max, 
																						 const MorphoGraphics::Vec3<unsigned int> & res, 
																						 float cell_size, 
																						 char * scale_grid, 
																						 const std::vector<bool> & feature_taggs, 
																						 int num_hc_iters) {
	c_qem_counter = 0;
	c_qem_timing = 0.0;
	target_num_faces_ = target_num_faces;
	target_error_ = target_error;
	int FinalSize = target_num_faces_;

	decimated_mesh_.Clear ();

	Allocator<MyMesh>::AddVertices (decimated_mesh_, mesh.P ().size ());
	Allocator<MyMesh>::AddFaces (decimated_mesh_, mesh.T ().size ());

	int i = 0;
	const std::vector<MorphoGraphics::Vec3f> & p = mesh.P ();
	const std::vector<MorphoGraphics::Vec3f> & n = mesh.N ();
	VertexIterator vi = decimated_mesh_.vert.begin ();

	for (i = 0; i < (int) p.size (); i++) {
		(*vi).P () = CoordType (p[i][0], p[i][1], p[i][2]);
		(*vi).N () = CoordType (n[i][0], n[i][1], n[i][2]);
		(*vi).set_is_feature (feature_taggs[i]);
		vi++;
	}

	std::vector<VertexPointer> index (decimated_mesh_.vn);
	for (i = 0, vi = decimated_mesh_.vert.begin (); i < decimated_mesh_.vn; ++i, ++vi) {
		index[i] = &(*vi);
	}

	const std::vector< MorphoGraphics::Vec3<unsigned int> > & t = mesh.T ();
	FaceIterator fi = decimated_mesh_.face.begin ();
	for (i = 0; i < (int) t.size (); i++) {
		(*fi).V (0) = index[t[i][0]];
		(*fi).V (1) = index[t[i][1]];
		(*fi).V (2) = index[t[i][2]];
		fi++;
	}

	//ExportMesh ("before_hc.ply");

	tri::Smooth<MyMesh>::VertexCoordLaplacianHC (decimated_mesh_, num_hc_iters);

	//ExportMesh ("before_decimation.ply");

	qparams.QualityThr  = .3;
	float TargetError = std::numeric_limits<float>::max ();
	qparams.QualityCheck	= true;  printf("Using Quality Checking\n");
	qparams.NormalCheck	= true;  printf("Using Normal Deviation Checking\n");
	qparams.OptimalPlacement	= true;  printf("Using OptimalPlacement\n");
	qparams.ScaleIndependent	= true;  printf("Using ScaleIndependent\n");
	qparams.max_edge_length_ = max_edge_length;
	qparams.bbox_min_ = bbox_min;
	qparams.bbox_max_ = bbox_max;
	qparams.res_ = res;
	qparams.cell_size_ = cell_size;
	qparams.scale_grid_ = scale_grid;
	qparams.use_linear_constraints_ = use_linear_constraints;
	qparams.w_qem_ = 1.f;
	qparams.w_edge_length_ = 0.f;
	qparams.use_features_ = use_features;
	qparams.qp_ws_ = new QPWorkspace (); // Quadratic Programming Workspace Allocation

	printf ("reducing it to %i\n",FinalSize);

	vcg::tri::UpdateBounding<MyMesh>::Box(decimated_mesh_);

	// decimator initialization
	deci_session_ = new vcg::LocalOptimization<MyMesh> (decimated_mesh_,&qparams);

	deci_session_->Init<CQEMEdgeCollapse>();
	printf ("Initial Heap Size %i\n",int (deci_session_->h.size()));

	deci_session_->SetTargetSimplices (FinalSize);
	deci_session_->SetTimeBudget (0.5f);

	TargetError = target_error_;

	if (TargetError < std::numeric_limits<float>::max ()) deci_session_->SetTargetMetric (TargetError);

	std::cout << "Target Error : " << TargetError << std::endl;

	num_faces_ = decimated_mesh_.fn;
	error_ = 0.f;

	//	delete deci_session_;
}

bool Decimator::VCGOptimSession::Optimize () {
	bool heap_not_empty = deci_session_->DoOptimization();
	num_faces_ = decimated_mesh_.fn;
	error_ = deci_session_->currMetric;
	return heap_not_empty;
}

void Decimator::VCGOptimSession::GetMesh (MorphoGraphics::Mesh & output_mesh) {
	std::vector<MorphoGraphics::Vec3f> & output_p = output_mesh.P ();
	SimpleTempData<typename MyMesh::VertContainer,int> indices (decimated_mesh_.vert);

	int i;
	VertexIterator vi;
	FaceIterator fi;
	output_p.resize (decimated_mesh_.vn);
	VertexPointer vp;
	for (i = 0, vi = decimated_mesh_.vert.begin (); vi != decimated_mesh_.vert.end ();) {
		vp = &(*vi);
		indices[vi] = i;
		if (!vp->IsD ()) {
			output_p[i][0] = vp->P ()[0];
			output_p[i][1] = vp->P ()[1];
			output_p[i][2] = vp->P ()[2];
			i++;
		}
		vi++;
	}

	std::vector< MorphoGraphics::Vec3<unsigned int> > & output_t = output_mesh.T ();
	FacePointer fp;
	output_t.resize (decimated_mesh_.fn);
	for (i = 0, fi = decimated_mesh_.face.begin (); fi != decimated_mesh_.face.end ();) {
		fp=&(*fi);
		if (!fp->IsD ()) {
			output_t[i][0] = indices[fp->cV (0)];
			output_t[i][1] = indices[fp->cV (1)];
			output_t[i][2] = indices[fp->cV (2)];
			i++;
		}
		fi++;
	}

	//	std::cout << c_qem_counter << " cqem_timing : " << c_qem_timing << std::endl;
}

Decimator::Decimator () {
}

Decimator::~Decimator () {
}

Decimator::Decimator (const MorphoGraphics::Mesh & mesh, int target_num_faces, 
											float target_error, float max_edge_length, 
											const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
											const MorphoGraphics::Vec3<unsigned int> & res, 
											float cell_size, char * scale_grid) {
	use_linear_constraints_ = true;
	use_features_ = false;
	std::vector<bool> feature_taggs;
	vcg_optim_session_ = new Decimator::VCGOptimSession (mesh, target_num_faces, 
																											 target_error, max_edge_length, 
																											 use_linear_constraints_,
																											 use_features_, 
																											 bbox_min, bbox_max, res, 
																											 cell_size, scale_grid, 
																											 feature_taggs);
	num_faces_ = vcg_optim_session_->num_faces ();
	target_num_faces_ = target_num_faces;
	error_ = vcg_optim_session_->error ();
	target_error_ = target_error;

}

Decimator::Decimator (const MorphoGraphics::Mesh & mesh, int target_num_faces, 
											float target_error, 
											bool use_linear_constraints, 
											bool use_features, 
											float max_edge_length, 
											const MorphoGraphics::Vec3f & bbox_min, const MorphoGraphics::Vec3f & bbox_max, 
											const MorphoGraphics::Vec3<unsigned int> & res, 
											float cell_size, char * scale_grid, 
											const std::vector<bool> & feature_taggs, 
											int num_hc_iters) {
	vcg_optim_session_ = new Decimator::VCGOptimSession (mesh, target_num_faces, 
																											 target_error, max_edge_length,
																											 use_linear_constraints, 
																											 use_features, 
																											 bbox_min, bbox_max, res, 
																											 cell_size, scale_grid, 
																											 feature_taggs, 
																											 num_hc_iters);
	num_faces_ = vcg_optim_session_->num_faces ();
	target_num_faces_ = target_num_faces;
	error_ = vcg_optim_session_->error ();
	target_error_ = target_error;
	use_linear_constraints_ = use_linear_constraints;
	use_features_ = use_features;
}

bool  Decimator::Optimize () {
	bool heap_not_empty = vcg_optim_session_->Optimize ();
	num_faces_ = vcg_optim_session_->num_faces ();
	error_ = vcg_optim_session_->error ();
	return heap_not_empty;
}

void Decimator::GetMesh (MorphoGraphics::Mesh & output_mesh) {
	vcg_optim_session_->GetMesh (output_mesh);
}

#pragma GCC diagnostic pop
