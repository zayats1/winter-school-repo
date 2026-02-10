import pygame
import random
import sys


pygame.init()

# Grid settings
TILE_SIZE = 40
GRID_WIDTH = 15
GRID_HEIGHT = 10

WIDTH = GRID_WIDTH * TILE_SIZE
HEIGHT = GRID_HEIGHT * TILE_SIZE

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grid Coin Collector")

# Colors
WHITE = (240, 240, 240)
BLUE = (50, 100, 255)
GOLD = (255, 215, 0)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)

font = pygame.font.SysFont(None, 36)
big_font = pygame.font.SysFont(None, 64)

clock = pygame.time.Clock()

# Map legend:
# 0 = empty
# 1 = wall
LEVEL_MAP = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,0,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
    [1,1,0,1,1,1,0,1,1,0,1,0,0,0,1],
    [1,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
    [1,0,1,1,1,0,1,0,1,1,1,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,0,0,1,1,1,0,1,1,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

# Player position (grid coordinates)
player_x, player_y = 1, 1

# Coin
def spawn_coin():
    while True:
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        if LEVEL_MAP[y][x] == 0 and (x, y) != (player_x, player_y):
            return x, y

coin_x, coin_y = spawn_coin()
coins_collected = 0
moves_count = 0 

running = True
while running:
    clock.tick(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            dx, dy = 0, 0
            if event.key == pygame.K_LEFT:
                dx = -1
            if event.key == pygame.K_RIGHT:
                dx = 1
            if event.key == pygame.K_UP:
                dy = -1
            if event.key == pygame.K_DOWN:
                dy = 1

            new_x = player_x + dx
            new_y = player_y + dy

            if LEVEL_MAP[new_y][new_x] == 0:
                player_x, player_y = new_x, new_y
                moves_count += 1 

    # Coin collection
    if player_x == coin_x and player_y == coin_y:
        coins_collected += 1
        coin_x, coin_y = spawn_coin()

    # Draw
    screen.fill(WHITE)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE, TILE_SIZE, TILE_SIZE)

            if LEVEL_MAP[y][x] == 1:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)

    # Draw coin
    pygame.draw.circle(
        screen,
        GOLD,
        (coin_x*TILE_SIZE + TILE_SIZE//2, coin_y*TILE_SIZE + TILE_SIZE//2),
        TILE_SIZE//4
    )

    # Draw player
    pygame.draw.rect(
        screen,
        BLUE,
        (player_x*TILE_SIZE, player_y*TILE_SIZE, TILE_SIZE, TILE_SIZE)
    )

    # UI: Draw Coins (Top Left)
    score_text = font.render(f"Coins: {coins_collected}", True, BLACK)
    screen.blit(score_text, (10, 10))

    # UI: Draw Moves (Top Right)
    moves_text = font.render(f"Moves: {moves_count}", True, BLACK)
    # To align right: Screen Width - Text Width - Margin
    moves_x_pos = WIDTH - moves_text.get_width() - 10 
    screen.blit(moves_text, (moves_x_pos, 10))

    if coins_collected >= 10:
        win_text = big_font.render("YOU WIN!", True, BLACK)
        result_text = font.render(f"Finished in {moves_count} moves", True, BLACK)
        
        # Center the text
        win_rect = win_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 20))
        result_rect = result_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 30))

        screen.blit(win_text, win_rect)
        screen.blit(result_text, result_rect)
        
        pygame.display.flip()
        pygame.time.delay(3000)
        running = False

    pygame.display.flip()

pygame.quit()