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
OFFSET_UPPER = 200
OFFSET_LOWER = 50
LOG_BLANK = 50
LOG_WIDTH = 10
LOG_NUM = 9
BANK_WIDTH = 60
BANK_NUM = 2
SCREEN_LENGTH = BANK_WIDTH * BANK_NUM + (LOG_NUM +
                                         1) * LOG_BLANK + LOG_NUM * LOG_WIDTH
FROG_START = SCREEN_LENGTH - BANK_WIDTH - FROG_SIZE - 5  #585

hardness = 9
speed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
running = True


def create_thread(tar, arg):
    thread = threading.Thread(target=tar, args=arg)
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
        self.y = 0  # move offset
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<a>', self.turn_left)
        self.canvas.bind_all('<d>', self.turn_right)
        self.canvas.bind_all('<w>', self.turn_up)
        self.canvas.bind_all('<s>', self.turn_down)
        self.pos = [0, 0]

    def move(self):
        self.canvas.move(self.id, self.x, self.y)

    def get_position(self):
        return self.pos

    def draw(self):
        self.move_with_log()
        self.canvas.move(self.id, self.x, self.y)  # offset of current position
        self.x = 0
        self.y = 0
        self.pos = self.canvas.coords(self.id)
        self.canvas.after(100, self.draw)

    def move_with_log(self):
        self.pos = self.canvas.coords(self.id)
        if (self.pos[1] < FROG_START and (FROG_START + 10 - self.pos[1]) %
            (LOG_BLANK + LOG_WIDTH) == 0 and self.pos[1] > 0):
            left = ((FROG_START + 10 - self.pos[1]) /
                    (LOG_BLANK + LOG_WIDTH)) % 2
            if (left == 1 and self.x == 0):
                self.x = -speed[hardness]
            elif (left == 0 and self.x == 0):
                self.x = speed[hardness]

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
    def __init__(self, canvas, color, startx, starty, length1, length2, left,
                 offset):
        self.canvas = canvas
        self.height = starty
        self.centre1 = startx
        self.centre2 = 0
        self.offset = offset
        self.length1 = length1
        self.length2 = length2
        self.id1 = canvas.create_rectangle(0,
                                           0,
                                           self.length1,
                                           LOG_WIDTH,
                                           fill=color)
        self.id2 = canvas.create_rectangle(0,
                                           0,
                                           self.length2,
                                           LOG_WIDTH,
                                           fill=color)
        self.canvas.move(self.id1, startx, self.height)
        self.pos1 = [0, 0, 0, 0]
        self.pos2 = [0, 0, 0, 0]

        if (left == 1):
            self.x = -speed[hardness]  # parallel speed
            self.centre2 = self.centre1 + offset + (self.length1 +
                                                    self.length2) * 0.6
            self.canvas.move(self.id2, self.centre2, self.height)
        else:
            self.x = speed[hardness]
            self.centre2 = self.centre1 - offset - (self.length1 +
                                                    self.length2) * 0.6
            self.canvas.move(self.id2, self.centre2, self.height)
        self.y = 0  # vertical speed
        self.canvas_height = self.canvas.winfo_height()  #height of the canvas
        self.canvas_width = self.canvas.winfo_width()  #width of the canvas
        self.hit_bottom = False

    def draw(self):
        self.canvas.move(self.id1, self.x, self.y)
        self.canvas.move(self.id2, self.x, self.y)
        self.pos1 = self.canvas.coords(self.id1)
        self.pos2 = self.canvas.coords(self.id2)
        # pos[1] top pos[3] bottom pos[0] left pos[2] right
        if self.pos1[0] >= self.canvas_width:
            self.canvas.move(self.id1, -self.canvas_width - self.length1 / 2,
                             0)
        elif self.pos1[2] <= 0:
            self.canvas.move(self.id1, self.canvas_width + self.length1 / 2, 0)
        if self.pos2[0] >= self.canvas_width:
            self.canvas.move(self.id2, -self.canvas_width - self.length1 / 2,
                             0)
        elif self.pos2[2] <= 0:
            self.canvas.move(self.id2, self.canvas_width + self.length1 / 2, 0)
        self.canvas.after(100, self.draw)


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
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log2 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 2 + LOG_WIDTH * 1.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log3 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 3 + LOG_WIDTH * 2.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log4 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 4 + LOG_WIDTH * 3.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log5 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 5 + LOG_WIDTH * 4.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log6 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 6 + LOG_WIDTH * 5.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log7 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 7 + LOG_WIDTH * 6.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log8 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 8 + LOG_WIDTH * 7.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 0,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))
log9 = Log(canvas, "green", random.randint(20, SCREEN_LENGTH / 2 - 10),
           BANK_WIDTH + LOG_BLANK * 9 + LOG_WIDTH * 8.5,
           random.randint(LENGTH_LOWER, LENGTH_UPPER),
           random.randint(LENGTH_LOWER, LENGTH_UPPER), 1,
           random.randint(OFFSET_LOWER, OFFSET_UPPER))

frog = Frog(canvas, "green")


def check_status(canvas, log1, log2, log3, log4, log5, log6, log7, log8, log9,
                 frog):
    global running
    while (1):
        frog_pos = frog.get_position()
        print(frog_pos)
        if (frog_pos[1] == -5):
            #print("\033[H\033[2J")
            print("You win the game!\n")
            running = False
        time.sleep(0.09)


# def draw_object(object):
#     object.draw()

create_thread(
    check_status,
    (canvas, log1, log2, log3, log4, log5, log6, log7, log8, log9, frog))

# create_thread(draw_object, log1)

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
while running:
    # if ball.hit_bottom ==False:
    #         ball.draw()
    #         log.draw()
    # else:
    #         break
    t.update_idletasks()
    t.update()
    time.sleep(0.01)
t.destroy()
t.mainloop()
