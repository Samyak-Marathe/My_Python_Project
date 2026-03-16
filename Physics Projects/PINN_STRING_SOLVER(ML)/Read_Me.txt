This is a fun little project I put together during my break. I wanted to solve a basic second-order partial differential equation—specifically, the 1D wave equation for string oscillations—using a Physics-Informed Neural Network (PINN).

Problem and motivation:

If you've ever tried to solve wave equations with a vanilla PINN, you probably know it usually just flatlines. The true analytical solution to this equation is highly sinusoidal, which standard PINNs traditionally fail to predict because neural networks are mathematically "lazy" and prefer to draw smooth, low-frequency lines.

To get around this, I transformed the raw $x$ and $t$ variables into Fourier space, which gives the network much more degrees of freedom to predict the true solution. Instead of feeding the network flat coordinates, I multiplied them by a matrix B = [w1, w2] containing random frequency values. For simplicity, I used a Gaussian distribution centered around 0 to select the values for w1 and w2. The new input variables fed into the network are sin(w1 * x + w2 * t) and cos(w1 * x + w2 * t). Basically, we are handing the network a buffet of pre-bent waves. It's like doing model a favor and giving it the input parameters that themselves form the basis for the true solution (x, t) may also form the basis for the solution but that would require infinite number of terms of x and t).

About the files:

I've attached a PDF document (PINN_Architecture.pdf) for reference that breaks down the 4-pillar architecture (Data, Network, PINN Engine, and Training).

The training script (PINN_W.py) uses a heavy-duty two-stage approach: Adam to bulldoze through the initial loss landscape, followed by L-BFGS to surgically snipe the exact minimum error.

Run main.py! I built a real-time pygame environment where you can actually watch the comparison of the predicted solutions and the true solution for different modes (training was done upto 5 modes for a fix value of velocity which is not a model parameter).

Additionally, animations are also included in the animation folder to see the simulation. Name of the sim files represent the type of solution. Green one is true solution while Red one is predicted using PINN+RFF. Please try to understand the name of the file and the corresponding simulation. I could not come up with a better way to explain lol.

In the future, I plan to add more projects solving interesting physical systems using ML models. However, for this one, I haven't written anything too heavy here in the Read Me.

Enjoy playing with the simulation!
