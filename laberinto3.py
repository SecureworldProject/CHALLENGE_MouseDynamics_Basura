import pygame

# Inicializar pygame
pygame.init()

# Establecer dimensiones de la pantalla
width = 640
height = 480
screen = pygame.display.set_mode((width, height))

# Definir colores
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)

# Definir tamaño y posición del personaje y del final
player_size = 15
player_pos = [0, height - player_size]
end_size = 30
end_pos = [width - end_size, 0]

# Definir variables de movimiento
speed = 5

# Definir el laberinto
maze = [[1, 1, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 0, 0, 1]]

columnas=width/len(maze[0])
filas=height/len(maze)
# Dibujar el laberinto
def draw_maze():
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                pygame.draw.rect(screen, black, (j*columnas, i*filas, columnas, filas))

# Dibujar el personaje y el final
def draw_player_and_end():
    pygame.draw.rect(screen, red, (end_pos[0], end_pos[1], end_size, end_size))
    pygame.draw.circle(screen, blue, (player_pos[0]+player_size//2, player_pos[1]+player_size//2), player_size//2)

# Verificar colisiones
def check_collision():
    if player_pos[0] > width - player_size or player_pos[0] < 0 or player_pos[1] > height - player_size or player_pos[1] < 0:
        return True
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                wall_rect = pygame.Rect(j*40, i*40, 40, 40)
                if wall_rect.collidepoint(player_pos):
                    return True
    if pygame.Rect(end_pos[0], end_pos[1], end_size, end_size).collidepoint(player_pos):
        return "WIN"

# Loop principal del juego
game_over = False
while not game_over:
    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            print(mouse_pos)
            target_pos = [mouse_pos[0]-player_size//2, mouse_pos[1]-player_size//2]

            # Actualizar posición del personaje
                # Actualizar posición del personaje
            if target_pos is not None:
                x_diff = target_pos[0] - player_pos[0]
                y_diff = target_pos[1] - player_pos[1]
                distance = (x_diff**2 + y_diff**2)**0.5
                if distance != 0:
                    x_move = x_diff / distance * speed
                    y_move = y_diff / distance * speed
                    player_pos[0] += x_move
                    player_pos[1] += y_move

            # Verificar colisiones
            if check_collision() == True:
                print("¡Perdiste!")
                game_over = True
            elif check_collision() == "WIN":
                print("¡Ganaste!")
                game_over = True

            # Dibujar elementos en pantalla
            screen.fill(white)
            draw_maze()
            draw_player_and_end()
            pygame.display.update()

# Cerrar Pygame
pygame.quit()

