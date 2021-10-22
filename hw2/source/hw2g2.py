from tkinter import *
import tkinter
import random
import time
from PIL import Image, ImageTk
import threading

# the length of logs range from LENGTH_LOWER to LENGTH_UPPER

FROG_SIZE = 60
LENGTH_LOWER = 200
LENGTH_UPPER = 300
LOG_BLANK = 50
LOG_WIDTH = 10
LOG_NUM = 9
BANK_WIDTH = 60
BANK_NUM = 2
SCREEN_LENGTH = BANK_WIDTH * BANK_NUM + (LOG_NUM +
                                         1) * LOG_BLANK + LOG_NUM * LOG_WIDTH
FROG_START = SCREEN_LENGTH - BANK_WIDTH - FROG_SIZE - 5

hardness = 0
speed = [1, 2, 3, 4, 5]
is_end = 0


def create_thread(tar):
    thread = threading.Thread(target=tar)
    thread.daemon = True
    thread.start()


class frog:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Frog:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.image = (Image.open("./frog.gif"))
        self.image = self.image.resize((FROG_SIZE, FROG_SIZE), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.image)
        self.id = canvas.create_image(10, 10, anchor='nw', image=self.image)
        self.canvas.move(self.id, SCREEN_LENGTH / 2, FROG_START)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<a>', self.turn_left)
        self.canvas.bind_all('<d>', self.turn_right)
        self.canvas.bind_all('<w>', self.turn_up)
        self.canvas.bind_all('<s>', self.turn_down)

    def move(self):
        self.canvas.move(self.id, self.x, self.y)

    def draw(self):
        self.canvas.move(self.id, self.x,
                         self.y)  # offset of the current position
        self.x = 0
        self.y = 0
        self.canvas.after(100, self.draw)
        pos = self.canvas.coords(self.id)
        # if pos[0] <= 0:
        #     self.x = 0
        # elif pos[2] >= self.canvas_width:
        #     self.x = 0

    def turn_up(self, evt):
        pos = self.canvas.coords(self.id)
        if (pos[1] > BANK_WIDTH / 2):
            self.y = -LOG_BLANK - LOG_WIDTH
            self.x = 0

    def turn_down(self, evt):
        pos = self.canvas.coords(self.id)
        if (pos[1] < FROG_START):
            self.y = LOG_BLANK + LOG_WIDTH
            self.x = 0

    def turn_left(self, evt):
        self.x = -LOG_BLANK + LOG_WIDTH
        self.y = 0

    def turn_right(self, evt):
        self.x = LOG_BLANK + LOG_WIDTH
        self.y = 0


class Log:
    def __init__(self, canvas, color, startx, starty, length, left):
        self.canvas = canvas
        self.height = starty
        self.id = canvas.create_rectangle(0, 0, length, LOG_WIDTH, fill=color)
        self.canvas.move(self.id, startx, starty)
        if (left == 1):
            self.x = -speed[hardness]  # parallel speed
        else:
            self.x = speed[hardness]
        self.y = 0  # vertical speed
        self.canvas_height = self.canvas.winfo_height()  #height of the canvas
        self.canvas_width = self.canvas.winfo_width()  #width of the canvas
        self.hit_bottom = False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)
        self.canvas.after(100, self.draw)
        pos = self.canvas.coords(self.id)  #coords函数通过ID来返回当前画布上任何画好的东西的当前X和Y坐标
        # pos[1] top pos[3] bottom pos[0] left pos[2] right
        if pos[0] >= self.canvas_width:
            self.canvas.move(self.id, 0, self.height)


class Bank:
    def __init__(self, canvas, color, x, y):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,
                                          0,
                                          SCREEN_LENGTH,
                                          BANK_WIDTH,
                                          fill=color)
        self.canvas.move(self.id, x, y)


t = tkinter.Tk()
t.title("Frog Game")
t.resizable(0, 0)
t.wm_attributes("-topmost", 1)


def close_win(e):
    print("\033[H\033[2J")
    print("You quit the game!\n")
    t.destroy()


canvas = Canvas(t,
                width=SCREEN_LENGTH,
                height=SCREEN_LENGTH,
                bd=0,
                highlightthickness=0)
t.bind('<q>', lambda e: close_win(e))
canvas.pack()
t.update()

bank1 = Bank(canvas, "brown", 0, SCREEN_LENGTH - BANK_WIDTH)
bank2 = Bank(canvas, "brown", 0, 0)
log1 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 1 + LOG_WIDTH * 0.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1)
log2 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 2 + LOG_WIDTH * 1.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0)
log3 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 3 + LOG_WIDTH * 2.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1)
log4 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 4 + LOG_WIDTH * 3.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0)
log5 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 5 + LOG_WIDTH * 4.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1)
log6 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 6 + LOG_WIDTH * 5.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0)
log7 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 7 + LOG_WIDTH * 6.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1)
log8 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 8 + LOG_WIDTH * 7.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0)
log9 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 9 + LOG_WIDTH * 8.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1)

frog = Frog(canvas, "green")
log1.draw()
log2.draw()
log3.draw()
log4.draw()
log5.draw()
log6.draw()
log7.draw()
log8.draw()
log9.draw()
frog.draw()
while 1:
    # if ball.hit_bottom ==False:
    #         ball.draw()
    #         log.draw()
    # else:
    #         break
    t.update_idletasks()
    t.update()
    time.sleep(0.01)

t.mainloop()
