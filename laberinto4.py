import pygame

# Inicializar pygame
pygame.init()

# Establecer dimensiones de la pantalla
width = 640
height = 640
screen = pygame.display.set_mode((width, height))

# Definir colores
black = (0, 0, 0)
white = (255, 255, 255)
blue = (0, 0, 255)
red = (255, 0, 0)
green=(0,255,0)


# Definir variables de movimiento
speed = 5

# Definir el laberinto
maze = [[1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,1,0,2],
        [1,0,1,1,1,0,1,0,1,0,0,1],
        [1,0,0,0,0,0,1,0,1,0,0,1],
        [1,0,1,1,1,1,1,0,1,0,0,1],
        [1,0,0,0,1,0,0,0,1,0,0,1],
        [1,1,1,0,1,0,1,1,1,0,0,1],
        [3,0,1,0,1,0,0,0,1,0,0,1],
        [1,0,1,0,0,1,1,0,1,0,1,1],
        [1,0,1,0,1,1,0,0,1,0,0,1],
        [1,0,0,0,1,0,0,1,1,1,0,1],
        [1,1,1,1,1,0,1,0,0,0,0,1],
        [1,1,1,1,0,0,1,0,1,0,0,1],
        [1,0,0,0,0,1,0,0,1,0,0,1],
        [1,0,1,1,1,1,0,1,0,0,0,1],
        [1,0,0,0,0,0,0,1,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1]]
columnas=width/len(maze[0])
filas=height/len(maze)

# Definir tamaño y posición del personaje y del final




player_size = 15
player_pos = [0*columnas, 7*filas+filas/2]
end_size = 30
end_pos = [width - end_size, 0]

# Dibujar el laberinto
def draw_maze():
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                pygame.draw.rect(screen, black, (j*columnas, i*filas, columnas, filas))
            if maze[i][j] == 2:
                pygame.draw.rect(screen, red, (j*columnas, i*filas, columnas, filas))
            if maze[i][j] == 3:
                pygame.draw.rect(screen, green, (j*columnas, i*filas, columnas, filas))
# Dibujar el personaje y el final
def draw_player_and_end():
    pygame.draw.circle(screen, blue, (player_pos[0]+player_size//2, player_pos[1]+player_size//2), player_size//2)

# Verificar colisiones
def check_collision():
    if player_pos[0] > width - player_size or player_pos[0] < 0 or player_pos[1] > height - player_size or player_pos[1] < 0:
        return True
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                
                wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)
                if wall_rect.collidepoint(player_pos):
                    return True
            if maze[i][j] == 2:
                wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)
                if wall_rect.collidepoint(player_pos):
                     return "WIN"
                
# Loop principal del juego
game_over = False
mover=False
screen.fill(white)
draw_maze()
draw_player_and_end()
pygame.display.update()
while not game_over:
    # Manejo de eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if mover==False:
            mouse_pos = pygame.mouse.get_pos()
            for i in range(len(maze)):
                for j in range(len(maze[i])):
                    if maze[i][j] == 3:
                        wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)     
                        mouse_pos = pygame.mouse.get_pos()
                        if wall_rect.collidepoint(mouse_pos):
                            mover=True
        else:
            mouse_pos = pygame.mouse.get_pos()
            print(mouse_pos)
            target_pos = [mouse_pos[0]-player_size//2, mouse_pos[1]-player_size//2]

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

