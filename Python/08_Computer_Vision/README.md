# Liveness detection

The aim of this project is to try to recignize not only a person, but also liveness of its face. 

There are many approaches to solving this task, all of them are different in the matter of implementation cost. Mine is considering face landmarks movements, though it is not a 100% method due to imperfection of the landmarks detection algorithm.

### Library used

In the project I used dlib as a landmarks detector tool along with face_recognition library.

### Functionality

- known faces detection
- liveness detection (do not work 100% correct when the face roll angle is more than 40 degrees. 
- known faces session counter
- head roll angle detection

### Example

![](images/example.gif)