# Dominant Set Clustering and Pooling for Multi-View 3D Object Recognition
This repository contains the Matlab implementation of the paper [*"Dominant Set Clustering and Pooling for
Multi-View 3D Object Recognition."*](http://www.cim.mcgill.ca/~chuwang/files/bmvc2017/0253.pdf) The paper is accepted to BMVC 2017.

![DSCNN pipeline](http://www.cim.mcgill.ca/~chuwang/files/bmvc2017/System_cluster_pooling_tight.svg)

**Code Credit Clarification:**

The folder GraphPool-master contains implementation of Dominant Set clustering and pooling layer and its dependencies. And it is implemented by Chu Wang http://www.cim.mcgill.ca/~chuwang/ . The CNN code framework is originally adopted from MVCNN https://github.com/suhangpro/mvcnn, but it is significantly modified to support end-to-end dominant set clustering during CNN training. The core forward/backward propagation rule for DS layer is defined in cnn_shape_init.m located in folder cnn_code.

**Setup:**

The code has been tested in 
  1) Ubuntu 14.04 with cuda7.5/8.0. 
  2) MacOS 10.10 with cuda 7.5. 

The results in the paper is acquired using 2) setup. 

To proceed to the following steps, you need Matlab and Cuda ready in your system. And for the dependencies, please consider using the provided ones in this repository. For the newest versions of vlfeat and Matconvnet, you may have to configure them yourself to make it work with this implementation.


After getting cuda and Matlab, you can follow the steps below to setup the code
Build dependencies. The dependencies include: matconvnet, liblinear, vlfeat. The provided code versions that are compatible with my framework.

1. Install matconvnet follow the instructions in 
http://www.vlfeat.org/matconvnet/install/
recommended cuda version 7.5, but 8.0 also compiles and works fine.
```
cd dependencies/matconvnet/
```
In matlab:
```
> addpath(genpath('matlab'))
> vl_compilenn('enableGpu', true, 'cudaRoot', '/Developer/NVIDIA/CUDA-8.0')
*change the path to your cuda library path.*
```
2. Compile vlfeat and liblinear
In matlab, do
```
> setup(true)
```

**Run experiments**

The experiment scripts are included in folder exp_scripts. See comments in the scripts for detail. Before you proceed, please download Modelnet40 dataset from http://modelnet.cs.princeton.edu/ and render it using tools provided in utils (credit to MVCNN authors). Here we provide rendered rgb images of ModelNet40 dataset in a [tarball file](http://www.cim.mcgill.ca/dscnn-data/ModelNet40_rendered_rgb.tar), rendered depth maps in a [tarball file](http://www.cim.mcgill.ca/dscnn-data/Modelnet40_Depth.tar), rendered surface normal maps in a [tarball file](http://www.cim.mcgill.ca/dscnn-data/Modelnet40_Surf.tar) for your convenience.

After you download the tarball, 

1) extract it and name the folder as Modelnet40_off.
2) create a folder named as data in your local directory.
2) move the dataset folder to data/

You are all set to run the experiments.

In matlab,
```
> run_experiments
> SVM_v_fast_approx
> SVM_v_end_to_end
```
**Reproduce the results**

We found that different system configuration may lead to different liblinear performance. So the accuracy you acquired at your system maybe not be exactly the same as presented in our paper. But you should observe a general trend that is mvcnn-fast-approx < mvcnn-end-to-end < dscnn-fast-approx < dscnn-end-to-end. 

In our experiment, we created separate repository for MVCNN end-to-end experiment, cloned from https://github.com/suhangpro/mvcnn. Therefore you should use their code for MVCNN end-to-end results.

