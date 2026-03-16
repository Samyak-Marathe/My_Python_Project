This is a fun little project I made during my break. Here I have solve a basic second order partial differntial equation (string oscillations) using Physics Informed Neural Network.
I have also used Ranfom Fourier Features to transform the (x, t) parameters to fourier space which results in more degrees of freedom for the input parameters for predicting the true solution.
A pdf document is attached for reference. In future I plan to add more projects on solving interesting problems using ML models. However, for this one I have not written anything important here in the Read Me.

Enjoy playing with the simulation. 
RFF:
Solution of the equation is just a sinusoidal function which PINN fails to predict traditionally. Therefore, I transformed the x, t variables using random list of numbers (fresuency) w1 and w2. 
The new input varaibles are sin(w1x+w2t), cos(w1x+w2t). B = [w1 w2] is the matrix containing random frequency values. For simplicity, I have used gaussian distribution to select values of w1 and w2 about 0.

Adios!
