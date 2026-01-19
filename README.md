# Kilobot_swarm
An implementation of the Kilobot algorithm from "Coverage Control for Mobile Sensing Networks" in Python.

![Demo GIF](./media/swarm.gif)

# Implmentation
This is a simulation of the algorithm designed it the paper to cordiante a robotic swarm using simple rules.
![Screenshot](./media/rules.png)

The formation is mapped with a bitmap image to define a boundry

![Screenshot](./media/star_bitmap.png)
![Screenshot](./media/wrench_bitmap.png)

# Result
![Screenshot](./media/K_comparision.png)
![Screenshot](./media/Wrench_comparision.png)
![Screenshot](./media/star_comparision.png)

The end result matched the algorithm but was limited by the number of bots able to be displayed. Rewrting this simulation would in C++ would lead to better results with speed and number of bots.


# Paper Cited
J. Cortés, S. Martínez, T. Karatas, and F. Bullo, "Coverage Control for Mobile Sensing Networks,"
