Vision-based vehicle detection 

Gorbatai Cristian (934)
Rusu Adrian	(936)

Vision-based vehicle detection 

Automated driving is a subject that has drawn many companies for obvious reasons. 
Some even already have some level of automation in certain environments, and the ability to reliably detect and classify objects from the surroundings is the cornerstone of this whole process. 
It is clear that it is impossible to avoid crashing into objects without first being able to identify them.
And in order to do that you need sensors, the receptors of machines. 
We decided to try to teach a visual sensor how to correctly detect vehicles by feeding it images of two types: images of vehicles and random images.

For the dataset we used this dataset of vehicles:
http://ai.stanford.edu/~jkrause/cars/car_dataset.html?fbclid=IwAR0cDyOJDqTOBRFPI5f39hhgNey3Pv-JJgSmHiNhEAzdhSGtjhwfGYpOg6c
It contains more than 16000 images of vehicles.

And for the non-vehicle images we just randomly generated images using Lorem Picsum:
https://picsum.photos/

Research papers:
1.A NEURAL NETWORK FOR IMAGE-BASED VEHICLE DETECTION:
In this paper vehicles are trying to be detected using a neural network as the brain which outputs a vector that is trying to simulate a Gaussian distribution of the aproximate location of a vehicle for a detection zone, with the peak roughly corresponding to the position of the center of the vehicle.

2.https://www.diva-portal.org/smash/get/diva2:1536051/FULLTEXT01.pdf
2D object detection and semantic segmentation in the Carla simulator:
In this paper the main goal is to create an application capable of object detection and semantic segmentation in CARLA, one of the best(if not the best) open source simulator for autonomous driving research. 
It should not only detect cars but also cyclists and the lanes. The paper covers object detection algorithms such as DMP, RCNN and YOLO and semantic segmentation ones FCN, Segnet, ESPnetv2; and how they can be applied for CARLA.
