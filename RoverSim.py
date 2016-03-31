import cv2
import numpy as np

class Rover:
    def __init__(self):
        self.leftTransitions = {
            "000":"001",
            "001":"002",
            "002":"102",
            "102":"103",
            "103":"203",
            "203":"200",
            "200":"210",
            "210":"310",
            "310":"320",
            "320":"020",
            "020":"030",
            "030":"000"
        }

        self.rightTransitions = {
            "001":"000",
            "002":"001",
            "102":"002",
            "103":"102",
            "203":"103",
            "200":"203",
            "210":"200",
            "310":"210",
            "320":"310",
            "020":"320",
            "030":"020",
            "000":"030"
        }

        self.state = "000"
        self.start = "000"
        self.terminal = "002"

        self.display = np.zeros((800, 800, 3), np.uint8)

    def flatten(self, string):
        total_arr = np.zeros((12))
        for i in xrange(len(string)):
            intc = int(string[i])
            total_arr[(i * 4) + intc] = 1
        return total_arr


    def frame_step(self, action, display=False):
        idx = np.argmax(action)
        if idx == 1:
            self.state = self.leftTransitions[self.state]
        elif idx == 2:
            self.state = self.rightTransitions[self.state]

        reward = 0
        terminal = False
        if self.state == self.terminal:
            reward = 100
            terminal = True
            self.state = self.start

        state_prime = self.flatten(self.state)

        if display:
            self.updateDisplay()

        return state_prime, reward, terminal

    def updateDisplay(self):
        self.display[:,:] = (0,0,0)
        self._updateDisplay(int(self.state[0]), (203, 192, 255)) #pink
        self._updateDisplay(int(self.state[1]), (0, 165, 255)) #orange
        self._updateDisplay(int(self.state[2]), (0, 255, 0)) #green

        cv2.imshow("window", self.display)
        cv2.waitKey(1)

    def _updateDisplay(self, idx, color):
        if idx == 1:
            self.display[:,0:0.333*800] = color
        elif idx == 2:
            self.display[:,0.333*800:0.666*800] = color
        elif idx == 3:
            self.display[:,0.666*800:800] = color

    def minPath(self):
        temp = self.start
        t1 = 0
        while temp != self.terminal:
            temp = self.leftTransitions[temp]
            t1 += 1
        temp = self.start
        t2 = 0
        while temp != self.terminal:
            temp = self.rightTransitions[temp]
            t2 += 1

        return min(t1, t2)