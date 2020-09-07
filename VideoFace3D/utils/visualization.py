import cv2
import random
import numpy as np

class CColor():
    def __init__(self, rgb_string):
        self.rgb_string = rgb_string

    def to_rgb(self):
        (r, g, b) = string2rgb(self.rgb_string)
        return (r, g, b)

    def to_bgr(self):
        (r, g, b) = string2rgb(self.rgb_string)
        return (b, g, r)


class ComfortableColor():
    def __init__(self):
        self.grape_fruit    = CColor("#ED5565")
        self.grape_fruit_d  = CColor("#DA4453")
        self.sun_flower     = CColor("#FFCE54")
        self.sun_flower_d   = CColor("#F6BB42")
        self.mint           = CColor("#48CFAD")
        self.mint_d         = CColor("#37BC9B")
        self.blue_jeans     = CColor("#5D9CEC")
        self.blue_jeans_d   = CColor("#4A89DC")
        self.pink_rose      = CColor("#EC89C0")
        self.pink_rose_d    = CColor("#D770AD")
        self.bitter_sweet   = CColor("#FC6E51")
        self.bitter_sweet_d = CColor("#E9573F")
        self.grass          = CColor("#A0D468")
        self.grass_d        = CColor("#8CC152")
        self.aqua           = CColor("#4FC1E9")
        self.aqua_d         = CColor("#3BAFDA")
        self.lavander       = CColor("#AC92EC")
        self.lavander_d     = CColor("#967ADC")
        self.light_gray     = CColor("#F5F7FA")
        self.light_gray_d   = CColor("#E6E9ED")
        self.medium_gray    = CColor("#CCD1D9")
        self.medium_gray_d  = CColor("#AAB2BD")
        self.dark_gray      = CColor("#656D78")
        self.dark_gray_d    = CColor("#434A54")

def string2hex(string):
    hex = 0
    for i in range(len(string)):
        if string[i] in ["{}".format(i) for i in range(10)]:
            hex += int(string[i]) * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "A":
            hex += 10 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "B":
            hex += 11 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "C":
            hex += 12 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "D":
            hex += 13 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "E":
            hex += 14 * 16 ** (len(string) - i - 1)
        elif string[i].upper() == "F":
            hex += 15 * 16 ** (len(string) - i - 1)
    return int(hex)


def string2rgb(string):
    r, g, b = string[1:3], string[3:5], string[5:7]
    return (string2hex(r), string2hex(g), string2hex(b))
    

def draw_landmarks(image, landmarks, plot_index=False, colors=None):
    image1 = image.copy()
    for i in range(len(landmarks)):
        lm_num = len(landmarks[i])
        if colors is not None:
            color = colors[i % len(colors)]
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for j in range(lm_num):
            x, y = int(landmarks[i][j][0]), int(landmarks[i][j][1])
            image1 = cv2.circle(image1, (x, y), radius=3, thickness=2, color=color)
            if plot_index:
                image1 = cv2.putText(image1, str(j), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            color,
                                            thickness=1)
    return image1


def draw_bbox(image, bboxs, colors=None):
    image1 = image.copy()
    for i in range(len(bboxs)):
        if colors is not None:
            color = colors[i % len(colors)]
        else:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image1 = cv2.rectangle(image1, (bboxs[i][0], bboxs[i][1]), (bboxs[i][2], bboxs[i][3]), color[i])
    return image1




