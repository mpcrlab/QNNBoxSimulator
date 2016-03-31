# QNNBoxSimulator
Simulation of the "rover in a box" problem using Q learning implemented in Tensorflow.

After testing hundreds of runs with multiple different "goal" states, it has ALWAYS?! found the shortest path.

To run an individual training and analysis, run sim3.py.  When you run sim3.py (uncomment the line that prints stuff first), you'll see the network's state at each iteration, followed by two numbers at the end of the iteration.  The first number is the # of steps in the shortest path found by the network.  The 2nd number is the # of steps in the actual shortest path.  If these numbers are the same, the network has found the shortest path!

For 1000 runs and the average # of steps error, run test.py.  Two numbers will be printed out at each iteration.  The first number is the index of the iteration.  The 2nd number is the average difference between the the number of steps in the network's shortest path and the actual shortest path.  All I get when I run it is 0 :)

TODO: - add boolean parameter to turn on/off printing to console at each iteration (currently just turned off)
- add visuals
- integrate with the real rover
