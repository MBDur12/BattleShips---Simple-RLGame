import pygame
import os
from random import randint
pygame.font.init()
"""
DEFINE CONSTANTS and SETUP WINDOW
"""
WIDTH, HEIGHT = 450, 500
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
FPS = 60
SCORE_FONT = pygame.font.SysFont("comicsans", 20)
GAMEOVER_FONT = pygame.font.SysFont("comicsans", 80)
# Player Paddle
PADDLE_WIDTH, PADDLE_HEIGHT = 60, 10
PADDLE_VELOCITY = 4
# Ball to keep up
BALL_WIDTH, BALL_HEIGHT = 20, 40
# Define here a random starting point? Maybe do later then as I also want the ball velocity to change over time
pygame.display.set_caption("Paddle Ball")
# Define custom events to call
INCREMENT_SCORE = pygame.USEREVENT + 1 # event to call every second to update the score count
INCREASE_BALL_SPEED = pygame.USEREVENT + 2
# Define colours
BLACK = (0, 0 ,0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)


def handle_paddle_movement(keys_pressed, paddle):
    if keys_pressed[pygame.K_LEFT] and paddle.x - PADDLE_VELOCITY > 0:
        paddle.x -= PADDLE_VELOCITY
    if keys_pressed[pygame.K_RIGHT] and paddle.x + PADDLE_VELOCITY + PADDLE_WIDTH < WIDTH:
        paddle.x += PADDLE_VELOCITY
    

def handle_ball_movement(ball, speed):
    if ball.x <= 0 or ball.x >= WIDTH:
        speed[0] = -speed[0]
    
    if ball.y <= 0:
        speed[1] = -speed[1]


def game_over():
    game_over_text = GAMEOVER_FONT.render("Game Over", 1, WHITE)
    WIN.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - game_over_text.get_height()//2))
    pygame.display.update()
    pygame.time.delay(1500)


def handle_collision(ball, paddle, ball_speed):
    if (ball.x >= paddle.x and ball.x <= paddle.x + PADDLE_WIDTH) and ball.y >= HEIGHT - 20 - PADDLE_HEIGHT:
        print(f"Collision - Ball: {ball.x}, {ball.y}; Paddle {paddle.x}. {paddle.y}")
        ball_speed[1] = -ball_speed[1]
    

def draw_window(paddle, ball, score):
    WIN.fill(BLACK)
    score_text = SCORE_FONT.render(f"Score: {score}", 1, WHITE)
    WIN.blit(score_text, (WIDTH//2 - score_text.get_width()//2, 20))

    pygame.draw.circle(WIN, RED, (ball.x, ball.y), 15)
    pygame.draw.rect(WIN, WHITE, paddle)
    pygame.display.update()

def check_game_over(ball):
    if ball.y >= HEIGHT:
        return True
    return False


def main():
    lost_game = False
    paddle = pygame.Rect(WIDTH//2 - PADDLE_WIDTH//2, HEIGHT - 20, PADDLE_WIDTH, PADDLE_HEIGHT)
    # randomize starting position of the ball (in the upper part of the screen)
    ball_starting_x, ball_starting_y= randint(0, WIDTH), randint(0, HEIGHT//4)
    # representation of ball to adjust movement on screen
    ball = pygame.Rect(ball_starting_x, ball_starting_y, BALL_WIDTH, BALL_HEIGHT)
    ball_speed = [5, 5]

    score = 0
    
    # Increase timer every second and increase ball speed every 5 seconds
    pygame.time.set_timer(INCREMENT_SCORE, 1000)
    pygame.time.set_timer(INCREASE_BALL_SPEED, 5000)
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
            
            if event.type == INCREMENT_SCORE:
                score += 10

            if event.type == INCREASE_BALL_SPEED:
                ball_speed = [1.1*val for val in ball_speed]

            
         

        
        handle_ball_movement(ball, ball_speed)
        handle_collision(ball, paddle, ball_speed)
        keys_pressed = pygame.key.get_pressed()
        handle_paddle_movement(keys_pressed, paddle)
        if check_game_over(ball):
            game_over()
            lost_game = True
            break

        ball = ball.move(ball_speed)
        
        
        

        draw_window(paddle, ball, score)
    
    if lost_game:
        main()
    else:
        pygame.quit()
   

    




if __name__ == "__main__":
    main()