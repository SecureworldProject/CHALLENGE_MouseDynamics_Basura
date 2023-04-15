import pygame
import random

# Inicializar Pygame
pygame.init()

# Definir colores
white = (255, 255, 255)
black = (0, 0, 0)

# Definir dimensiones de la pantalla
screen_width = 800
screen_height = 600

# Definir dimensiones de la meta y la velocidad del jugador
target_size = 50
speed = 5

# Definir posición inicial aleatoria del jugador y la meta
player_pos = [random.randint(0, screen_width-target_size), random.randint(0, screen_height-target_size)]
target_pos = [random.randint(0, screen_width-target_size), random.randint(0, screen_height-target_size)]

# Crear ventana
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Laberinto")

# Función para dibujar el jugador y la meta
def draw_player_and_target():
    pygame.draw.rect(screen, black, (player_pos[0], player_pos[1], target_size, target_size))
    pygame.draw.rect(screen, black, (target_pos[0], target_pos[1], target_size, target_size))

# Ciclo principal del juego
game_over = False
while not game_over:
    # Obtener eventos del teclado y del mouse
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            target_pos = list(event.pos)

    # Mover al jugador hacia la meta
    x_diff = target_pos[0] - player_pos[0]
    y_diff = target_pos[1] - player_pos[1]
    distance = (x_diff**2 + y_diff**2)**0.5
    if distance != 0:
        x_move = x_diff / distance * speed
        y_move = y_diff / distance * speed
        player_pos[0] += x_move
        player_pos[1] += y_move

    # Verificar colisiones
    if pygame.Rect(player_pos[0], player_pos[1], target_size, target_size).colliderect(pygame.Rect(target_pos[0], target_pos[1], target_size, target_size)):
        print("¡Ganaste!")
        target_pos = [random.randint(0, screen_width-target_size), random.randint(0, screen_height-target_size)]

    # Dibujar elementos en pantalla
    screen.fill(white)
    draw_player_and_target()
    pygame.display.update()

# Cerrar Pygame
pygame.quit()
