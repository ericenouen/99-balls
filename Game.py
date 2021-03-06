import pygame
import random
import math
import numpy as np

Black = (0, 0, 0)  # Color Presets
White = (255, 255, 255)
GreenBlue = (0, 255, 200)
Purple = (102, 0, 204)
Gray = (25, 25, 25)


class Block:  # Objects to destroy

    def __init__(self, x, y, score):  # Initialize
        self.x = x
        self.y = y
        self.score = score

    def draw(self, screen):
        pygame.draw.rect(screen, Purple, (self.x, self.y, 80, 80), 0)  # Fill
        pygame.draw.rect(screen, Black, (self.x, self.y, 80, 80), 1)  # Outline

        my_font = pygame.font.SysFont('Comic Sans MS', 24)
        text_surface = my_font.render(str(self.score), False, Black)
        screen.blit(text_surface, (self.x + 30, self.y + 10))  # Block score

    def collide(self, BlockList):  # Remove a Point for each collision
        self.score -= 1
        if self.score == 0:
            self.__delete__(BlockList)

    def drop(self):  # Move block down a level
        self.y += 80

    def __delete__(self, BlockList):  # Delete
        BlockList.remove(self)


class Orb:  # Objects that give an extra ball
    def __init__(self, x, y):  # Initialize
        self.x = x + 40
        self.y = y + 40

    def draw(self, screen):
        pygame.draw.circle(screen, White, (self.x, self.y), 16, 1)  # Draw outline

    def drop(self):  # Move orb down a level
        self.y += 80

    def delete(self, OrbList):  # Delete
        OrbList.remove(self)


class Ball:  # Balls to shoot
    def __init__(self, x, y, direction):  # Initialize
        self.x = x
        self.y = y
        self.direction = direction
        self.moving = False
        self.x_ = 0
        self.y_ = 0

    def set_direction(self, direction):  # Set initial direction
        self.direction = direction

    def draw(self, screen):
        pygame.draw.circle(screen, White, (int(self.x), int(self.y)), 10)  # Draw ball

    def move(self):  # Move ball and keep it inside the frame
        self.x_ = self.x
        self.y_ = self.y
        self.y += 5 * math.sin(self.direction)
        self.x += 5 * math.cos(self.direction)
        if self.y < -5:
            self.y = 0
        if self.x > 645:
            self.x = 640
        if self.x < -5:
            self.x = 0

    def collide(self, BlockList, OrbList, Balls):  # Deal with collisions
        if self.y < 0:
            self.direction = -self.direction
            # Top of screen
        if self.x < 0 or self.x > 640:
            # Left or Right of Screen
            if self.direction < 0:
                self.direction = - math.pi - self.direction
            else:
                self.direction = math.pi - self.direction

        for block in BlockList:
            if block.x - 5 <= self.x <= block.x + 85 and block.y - 5 <= self.y <= block.y + 85:  # Inside a block
                if block.x - 5 <= self.x_ <= block.x + 85:
                    if not (block.y - 5 <= self.y_ <= block.y + 85):
                        # Hitting the bottom/top
                        self.direction = -self.direction
                elif block.y - 5 <= self.y_ <= block.y + 85:
                    # Hitting the left/right
                    if self.direction < 0:
                        self.direction = - math.pi - self.direction  # Traveling up
                    else:
                        self.direction = math.pi - self.direction  # Traveling down
                else:
                    if self.x_ > block.x and self.y_ > block.y:
                        # Bottom Right Corner
                        self.direction = math.pi/4
                    if self.x_ > block.x and self.y_ < block.y:
                        # Top Right Corner
                        self.direction = -math.pi/4
                    if self.x_ < block.x and self.y_ > block.y:
                        # Bottom Left Corner
                        self.direction = (3*math.pi)/4
                    if self.x_ < block.x and self.y_ < block.y:
                        # Top Left Corner
                        self.direction = (-3*math.pi)/4
                block.collide(BlockList)

        for o in OrbList:
            if o.x - 20 <= self.x <= (o.x + 20) and o.y - 20 <= self.y <= (o.y + 20):  # Orb collision
                Balls.append(Ball(320, 810, 0))
                o.delete(OrbList)


class Run:
    def __init__(self):
        self.state = np.zeros([8, 8, 3])
        self.level = 1
        self.blockCount = 0

        self.BlockList = []
        self.OrbList = []  # Initialize lists of objects
        self.Balls = [Ball(320, 790, 0)]

        self.start_x = 320
        self.start_y = 790

        import os
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (500, 100)
        # Set initial location of the form

        self.pygame = pygame  # Set up pygame
        self.pygame.init()

        self.screen = self.pygame.display.set_mode((640, 900))
        self.pygame.display.set_caption("Balls")
        icon = self.pygame.image.load("icon.png")
        self.pygame.display.set_icon(icon)
        # Set up the form

        self.blocks_destroyed = 0
        self.Update()

    def Update(self):
        self.blockCount = len(self.BlockList)
        self.state = np.zeros([8, 8, 3])

        OrbPos = int(random.randrange(8))
        self.OrbList.append(Orb(OrbPos * 80, 0))
        # Place an orb on each level
        for i in range(8):
            if random.random() < .5 and i != OrbPos:  # Spawn blocks
                if random.random() < .25:
                    self.BlockList.append(Block((i * 80), 0, self.level * 2))
                    # Twice the value
                else:
                    self.BlockList.append(Block((i * 80), 0, self.level))
                    # Regular value

                self.BlockList[self.blockCount].draw(self.screen)
                self.blockCount += 1

        for b in self.BlockList:  # Move everything down a row
            b.drop()
            if b.y >= 720:  # Game over
                return True
            self.state[(b.x // 80), (8-(b.y // 80)), 0] = b.score / self.level
            self.state[(b.x // 80), (8-(b.y // 80)), 1] = (b.x / 80) / 8
            self.state[(b.x // 80), (8-(b.y // 80)), 2] = (b.y / 80) / 8
            # The state is the locations of all of the blocks
        for o in self.OrbList:
            o.drop()
            if o.y > 720:
                o.delete(self.OrbList)
            else:
                self.state[(o.x // 80), (8-(o.y // 80)), 0] = (0 - 4) / 8  # No clue what number should represent an orb
                self.state[(o.x // 80), (8-(o.y // 80)), 1] = (o.x / 80) / 8
                self.state[(o.x // 80), (8-(o.y // 80)), 2] = (o.y / 80) / 8
                # The state is also the locations of all of the orbs
        self.level += 1
        return False

    def SetTheta(self, direction, draw):  # Set theta
        direction = abs(direction)
        direction = direction % math.pi  # Puts direction into the range needed to shoot
        if direction < .01:
            direction = -.1
        if direction > math.pi-.01:
            direction = -(math.pi-.1)
        direction = -direction

        self.blocks_destroyed = len(self.BlockList)
        for b in self.Balls:
            b.set_direction(direction)
            # Set initial direction
        self.start_x = self.shoot(draw)  # Shoot all the balls
        self.blocks_destroyed -= len(self.BlockList)
        terminal = self.Update()

        state = np.reshape(self.state, (1, 192))
        state = np.append(state, self.start_x)  # Add position of ball to state
        state = np.append(state, len(self.Balls))  # Add number of balls to state
        state = state.astype('float32').reshape((-1, 194))
        return state, terminal

    def shoot(self, draw):
        count = 0
        for b in self.Balls:  # Start all balls
            b.moving = True
        shot = True
        max_b = None
        while shot:
            for i in range(len(self.Balls)):
                if self.Balls[i].y > 790:  # Stop balls if they reach the end
                    self.Balls[i].moving = False

                if count > i * 5:  # Move balls spaced away from each other
                    if self.Balls[i].moving:
                        self.Balls[i].move()
                        self.Balls[i].collide(self.BlockList, self.OrbList, self.Balls)

                if self.Balls[i].moving and max_b is None and 790 < self.Balls[i].y < 810:
                    # Get the first ball that lands to set its position for the next starting point
                    max_b = self.Balls[i]
            if draw:
                self.redraw()

            count += 1
            if 790 < min(ball.y for ball in self.Balls if ball.y != 810):  # If the last ball stops
                shot = False
                for b in self.Balls:  # Reset balls
                    b.y = 790
                    b.x = max_b.x

                    if b.x < 5:  # Keep inside frame
                        b.x = 5
                    elif b.x > 635:
                        b.x = 635
        return (max_b.x / 80) / 8

    def redraw(self):
        self.screen.fill(Gray)
        # Blocks
        for b in self.BlockList:
            b.draw(self.screen)

        # Orbs
        for o in self.OrbList:
            o.draw(self.screen)

        # Floor
        pygame.draw.rect(self.screen, GreenBlue, (0, 800, 640, 25))

        # Balls
        for b in self.Balls:
            b.draw(self.screen)

        # Level
        my_font = pygame.font.SysFont('Comic Sans MS', 24)
        text_surface = my_font.render("Level " + str(self.level), False, White)
        self.screen.blit(text_surface, (280, 860))

        # Update
        pygame.display.update()