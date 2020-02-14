TEMPLATE = lib
TARGET   = GMorpho

CONFIG += warn_on thread release rtti
CONFIG -= debug profile
QT -= core widgets opengl xml gui

INCLUDEPATH += '.'
INCLUDEPATH += '..'
DESTDIR		= ../Bin
MOC_DIR = .tmp
OBJECTS_DIR = .tmp

QMAKE_CC = gcc-4.8
QMAKE_CXX = g++-4.8
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -march=native 
unix:LIBS += -lGLU -L/usr/local/lib64 -lGLEW -lgomp
unix:LIBS += -lcusolver -lcusparse -lgfortran

INCLUDEPATH += /usr/local/include/GL # needed for glew
INCLUDEPATH += eigen3

# Set your CUDA gpu architecture
CUDA_COMPUTE_ARCH = 52
include (cuda.prf)
add_cuda_source (FrameField.cu)
add_cuda_source (FrameFieldSparseBiharmonic.cu)
add_cuda_source (FrameFieldLocalOptimization.cu)
add_cuda_source (FrameFieldQuotientSpace.cu)
add_cuda_source (ScaleField.cu)
add_cuda_source (Grid.cu)
add_cuda_source (BVH.cu)
add_cuda_source (Voxelizer.cu)
add_cuda_source (GMorpho.cu)
add_cuda_source (MarchingCubesMesher.cu)

HEADERS += Grid.h 
HEADERS += BVH.h 
HEADERS += Voxelizer.h 
HEADERS += FrameField.h 
HEADERS += ScaleField.h 
HEADERS += GMorpho.h 
HEADERS += MarchingCubesMesher.h 

