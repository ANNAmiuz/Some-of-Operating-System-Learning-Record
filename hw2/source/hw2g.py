from tkinter import *
import tkinter
import random
import time
from PIL import Image, ImageTk
import threading

# the length of logs range from LENGTH_LOWER to LENGTH_UPPER

FROG_SIZE = 60
LENGTH_LOWER = 150
LENGTH_UPPER = 200
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
running = True
end = 0  # 1: win 2: lose 3: quit


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
                self.x = -hardness * 1.1
            elif (left == 0 and self.x == 0):
                self.x = hardness * 1.1

    def turn_up(self, evt):
        self.pos = self.canvas.coords(self.id)
        if (self.pos[1] > BANK_WIDTH / 2):
            self.y = -LOG_BLANK - LOG_WIDTH
            self.x = 0

    def turn_down(self, evt):
        self.pos = self.canvas.coords(self.id)
        if (self.pos[1] < FROG_START):
            self.y = LOG_BLANK + LOG_WIDTH
            self.x = 0

    def turn_left(self, evt):
        self.pos = self.canvas.coords(self.id)
        left = ((FROG_START + 10 - self.pos[1]) / (LOG_BLANK + LOG_WIDTH)) % 2
        if (left == 1):
            self.x = (-LOG_BLANK - LOG_WIDTH) / 4 - hardness*1.1
        else:
            self.x = (-LOG_BLANK - LOG_WIDTH) / 4
        self.y = 0

    def turn_right(self, evt):
        self.pos = self.canvas.coords(self.id)
        left = ((FROG_START + 10 - self.pos[1]) / (LOG_BLANK + LOG_WIDTH)) % 2
        if (left == 1):
            self.x = (LOG_BLANK + LOG_WIDTH) / 4
        else:
            self.x = (LOG_BLANK + LOG_WIDTH) / 4 + hardness*1.1
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
        # self.speed = speed[hardness]
        self.left = left
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

        if (self.left == 1):
            # horizontal speed
            self.x = -hardness * 1.1
            self.centre2 = self.centre1 + offset + (self.length1 +
                                                    self.length2) * 0.6
            self.canvas.move(self.id2, self.centre2, self.height)
        else:
            self.x = hardness * 1.1
            self.centre2 = self.centre1 - offset - (self.length1 +
                                                    self.length2) * 0.6
            self.canvas.move(self.id2, self.centre2, self.height)
        self.y = 0  # vertical speed
        self.canvas_width = self.canvas.winfo_width()  #width of the canvas
        self.hit_bottom = False
        
    # horizontal end point position
    def get_position(self):
        return [self.pos1[0], self.pos1[2], self.pos2[0], self.pos2[2]]

    def draw(self):
        if (self.left == 1):
            self.x = -hardness * 1.1
        else:
            self.x = hardness * 1.1
        self.canvas.move(self.id1, self.x, self.y)
        self.canvas.move(self.id2, self.x, self.y)
        self.pos1 = self.canvas.coords(self.id1)
        self.pos2 = self.canvas.coords(self.id2)
        # pos[1] top pos[3] bottom pos[0] left pos[2] right
        if self.pos1[0] >= self.canvas_width:
            self.canvas.move(self.id1, -self.canvas_width - self.length1 / 1,
                             0)
        elif self.pos1[2] <= 0:
            self.canvas.move(self.id1, self.canvas_width + self.length1 / 1, 0)
        if self.pos2[0] >= self.canvas_width:
            self.canvas.move(self.id2, -self.canvas_width - self.length2 / 1,
                             0)
        elif self.pos2[2] <= 0:
            self.canvas.move(self.id2, self.canvas_width + self.length2 / 1, 0)
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


def quit_the_game(e):
    global running
    if (running):
        print("\033[H\033[2J")
        print("You quit the game!\n")
        t.title(
            "You quit the game! Wait 5 seconds to close the window.\n"
        )
        running = False


canvas = Canvas(t,
                width=SCREEN_LENGTH,
                height=SCREEN_LENGTH,
                bd=0,
                highlightthickness=0)
t.bind('<q>', lambda e: quit_the_game(e))
#t.bind("<Escape>", lambda x: t.destroy())
t.bind('<Escape>', lambda e: quit_the_game(e))

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

# Catch the current status of the game : win / lose
def check_status(canvas, log1, log2, log3, log4, log5, log6, log7, log8, log9,
                 frog):
    global running
    global end
    while (1):
        frog_pos = frog.get_position()
        log1_pos = log1.get_position()
        log2_pos = log2.get_position()
        log3_pos = log3.get_position()
        log4_pos = log4.get_position()
        log5_pos = log5.get_position()
        log6_pos = log6.get_position()
        log7_pos = log7.get_position()
        log8_pos = log8.get_position()
        log9_pos = log9.get_position()
        if (frog_pos[0] < -20 or frog_pos[0] > SCREEN_LENGTH - 20):
            print("\033[H\033[2J")
            print("You lose the game! Get out of the bound!\n")
            end = 2
            running = False
            break
        if (frog_pos[1] == -5):
            print("\033[H\033[2J")
            print("You win the game!\n")
            running = False
            end = 1
            break

        # at the height of the first log ~ ninth log
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) - 5):
            if not ((frog_pos[0] < log1_pos[1] - 10
                     and frog_pos[0] > log1_pos[0] - 45) or
                    (frog_pos[0] < log1_pos[3] - 10
                     and frog_pos[0] > log1_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 2 - 5):
            if not ((frog_pos[0] < log2_pos[1] - 10
                     and frog_pos[0] > log2_pos[0] - 45) or
                    (frog_pos[0] < log2_pos[3] - 10
                     and frog_pos[0] > log2_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 3 - 5):
            if not ((frog_pos[0] < log3_pos[1] - 10
                     and frog_pos[0] > log3_pos[0] - 45) or
                    (frog_pos[0] < log3_pos[3] - 10
                     and frog_pos[0] > log3_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 4 - 5):
            if not ((frog_pos[0] < log4_pos[1] - 10
                     and frog_pos[0] > log4_pos[0] - 45) or
                    (frog_pos[0] < log4_pos[3] - 10
                     and frog_pos[0] > log4_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 5 - 5):
            if not ((frog_pos[0] < log5_pos[1] - 10
                     and frog_pos[0] > log5_pos[0] - 45) or
                    (frog_pos[0] < log5_pos[3] - 10
                     and frog_pos[0] > log5_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 6 - 5):
            if not ((frog_pos[0] < log6_pos[1] - 10
                     and frog_pos[0] > log6_pos[0] - 45) or
                    (frog_pos[0] < log6_pos[3] - 10
                     and frog_pos[0] > log6_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 7 - 5):
            if not ((frog_pos[0] < log7_pos[1] - 10
                     and frog_pos[0] > log7_pos[0] - 45) or
                    (frog_pos[0] < log7_pos[3] - 10
                     and frog_pos[0] > log7_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 8 - 5):
            if not ((frog_pos[0] < log8_pos[1] - 10
                     and frog_pos[0] > log8_pos[0] - 45) or
                    (frog_pos[0] < log8_pos[3] - 10
                     and frog_pos[0] > log8_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        elif (frog_pos[1] == (LOG_BLANK + LOG_WIDTH) * 9 - 5):
            if not ((frog_pos[0] < log9_pos[1] - 10
                     and frog_pos[0] > log9_pos[0] - 45) or
                    (frog_pos[0] < log9_pos[3] - 10
                     and frog_pos[0] > log9_pos[2] - 45)):
                print("\033[H\033[2J")
                print("You lose the game! Get into the river!\n")
                running = False
                end = 2
                break
        time.sleep(0.0005)


create_thread(
    check_status,
    (canvas, log1, log2, log3, log4, log5, log6, log7, log8, log9, frog))

s = tkinter.Scale(t, label='Adjust The Speed', from_=1, to=10, orient=tkinter.HORIZONTAL, length=SCREEN_LENGTH-5, showvalue=0,tickinterval=1, resolution=0.01)
s.pack()

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
    hardness = s.get()
    t.update_idletasks()
    t.update()
    time.sleep(0.005)

if (end == 1):
    t.title(
        "You win! Wait 5 seconds to close the window.\n"
    )
elif (end == 2):
    t.title(
        "You lose! Wait 5 seconds to close the window.\n"
    )
t.update_idletasks()
t.update()
time.sleep(5)
t.destroy()
t.mainloop()
