TEMPLATE = app
TARGET   = Broxy
CONFIG += qt core gui warn_on thread release rtti
QT *= opengl xml

INCLUDEPATH += ./
DESTDIR		= ./Bin
MOC_DIR = .tmp
OBJECTS_DIR = .tmp

QMAKE_CC = gcc-4.8
QMAKE_CXX = g++-4.8
QMAKE_CXXFLAGS += -std=c++11 -fopenmp -march=native -O2 -use_fast_math -m64
unix:LIBS += -lGLU -L/usr/local/lib64 -lGLEW -lQGLViewer -lgomp
unix:LIBS += -lcusolver -lcusparse -lgfortran
unix:LIBS += -LBin/ -lGMorpho

INCLUDEPATH += ./GMorpho
INCLUDEPATH += ./Decimator 
INCLUDEPATH += ./vcglib 
INCLUDEPATH += ./CPoly
INCLUDEPATH += /usr/local/include/GL # needed for glew
INCLUDEPATH += eigen3

# Set your CUDA gpu architecture
#CUDA_COMPUTE_ARCH = 52
include (cuda.prf)
#add_cuda_source (GMorpho/FrameField.cu)
#add_cuda_source (GMorpho/FrameFieldSparseBiharmonic.cu)
#add_cuda_source (GMorpho/FrameFieldLocalOptimization.cu)
#add_cuda_source (GMorpho/FrameFieldQuotientSpace.cu)
#add_cuda_source (GMorpho/ScaleField.cu)
#add_cuda_source (GMorpho/Grid.cu)
#add_cuda_source (GMorpho/BVH.cu)
#add_cuda_source (GMorpho/Voxelizer.cu)
#add_cuda_source (GMorpho/GMorpho.cu)
#add_cuda_source (GMorpho/MarchingCubesMesher.cu)

HEADERS = BroxyApp.h
HEADERS += BroxyViewer.h
#HEADERS += GMorpho/Grid.h 
#HEADERS += GMorpho/BVH.h 
#HEADERS += GMorpho/Voxelizer.h 
#HEADERS += GMorpho/FrameField.h 
#HEADERS += GMorpho/ScaleField.h 
#HEADERS += GMorpho/GMorpho.h 
#HEADERS += GMorpho/MarchingCubesMesher.h 
HEADERS += Decimator/Decimator.h 
HEADERS += CPoly/CPoly.h

SOURCES = Main.cpp 
SOURCES += BroxyApp.cpp
SOURCES += BroxyViewer.cpp
SOURCES += Decimator/Decimator.cpp 
SOURCES += CPoly/CPoly.cpp

# Fortran routine integration
FORTRAN_SOURCES += QP/ql0001.f 
fortran.output = ${QMAKE_FILE_BASE}.o
fortran.commands = gfortran -O3 -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
fortran.input = FORTRAN_SOURCES
QMAKE_EXTRA_COMPILERS += fortran

