# Broxy
Bounding Proxies for Shape Approximation (SIGGRAPH 2017) - Reference Implementation

by Stéphane Calderon and Tamy Boubekeur
ACM Transaction on Graphics - Proc. SIGGRAPH 2017

## Release Notes

### Temporary release!

For now, only the core of the paper will be released online. This is mainly the GMorpho library which allows to voxelize, perform high speed morphology on GPU and extract a high res. mesh from it.
The rest of the code (graphics interface and constrained mesh optimizer) will be released in the future, as soon as we figure out the right license to use.

## About

This program allows to load a surface mesh in OFF format and generate a proxy at arbitrary, spatially varying resolution. 
The user can globally control the proxy geometry using a global scaleand locally tune it using a brush.
The user can intereact with the proxy in its volumetric format and then mesh it adaptively to export it in mesh format.

## COMPILING

This program has been tested on Linux Ubuntu 14.04 and 16.04.
* Install gcc >=4.8, gfortran, Qt >=5.2.1, OpenGL >=4.5, GLEW >= 2.0, Eigen >=3, CUDA 7.5 
* Edit Broxy.pro to adjust the paths and options to your environment
* From the command line:
  * user@machine: cd <path-to-broxy>/GMorpho
  * user@machine: qmake
  * user@machine: make
  * user@machine: cd ..
  * user@machine: qmake
  * user@machine: make

## RUNNING

From the command line:

* user@machine: cd <path-to-broxy>/Bin 
* user@machine: ./Broxy

By default, the program loads <path-to-broxy>/Bin/Resources/Models/Beast.off, use the menu to load other models. 

## Known bug and limitations

The release of this code is still work in progress, with NUMEROUS things to fix:
* when loading a new model, the screen does ot resize properly; for a quick fix, just resize your window a bit
* when loading a new model, the previous proxy may not be cleaned (not really a bug, as it allows to compare proxies for several versions of the same mesh, but definitely not a feature)
* the optional rotation field experiments are not fully integrated in the UI, this may be disturing when accidentally pressing some keys
* indeed, may keys are used for quick hacks and for enabling some modes, which are not reflected in the UI
* to try alternative base resolution, you need to recompile
* the program may not compile with recent versions of CUDA in windows, due to weird incompatibilities between Eigen and the Visual Studio versions that can actually run with recent CUDA release
* many more..., but so far nothing linked to the core of the method (the morphological kernel)

## Authors

* [**Stéphane Calderon**](https://www.linkedin.com/in/st%C3%A9phane-calderon-509ab628/?ppe=1) 
* [**Tamy Boubekeur**](https://www.telecom-paristech.fr/~boubek)

## Citation

Please cite the following paper in case you are using this code:
>**Bounding Proxies for Shape Approximation.** *Stéphane Calderon and Tamy Boubekeur.* ACM Transaction on Graphics (Proc. SIGGRAPH 2017), vol. 36, no. 5, art. 57, 2017.

## License

See the [LICENSE.txt](LICENSE.txt) file for details. 
