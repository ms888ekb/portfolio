# Head Yaw Angle Detection

The aim of this project is to gauge the yaw angle of the human head with neural networks.

### Details

In the project, I used created a functional neural network using Keras package. The first branch of the network processes numeric data, like face landmarks proportions and so on. The second branch is a headless VGG network with several additional layers. 
On the top, two branches are concatenated and extended by several fully connected dense layers.

The most interesting part of this project (besides NN construction) is a training dataset. It was made by using 3D modeling software where a camera rotated around the human heads taking a shot every single angle. Then, using the in-built Python scripting engine, these shots were named and saved on a hard drive.
The names of these shots contain the following data: angle, shot number, head number. In the project, I parsed this data when loaded the shots into the training dataset.
It was enough just 6 heads to build a quite robust and relatively precise model.

### Features

- detects face angle relatively the camera view.
- uses 3D modeled faces to train
- applies the affine transformation to vertically align the face for better prediction.
- the NN model consists of numeric and visual processing branches.

### Example

![](images/example.gif)