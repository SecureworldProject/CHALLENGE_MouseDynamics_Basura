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
#tiempo de captura de datos
#------------------------------------------------------------------------------------
intervalo=100



# Definir variables de movimiento
speed = 5

# Definir el laberinto
maze = [[1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,1,0,0,0,0,0,1,0,2],
        [1,0,1,1,1,0,1,0,1,1,1,0,1],
        [1,0,0,0,0,0,1,0,1,0,0,0,1],
        [1,0,1,1,1,1,1,0,1,0,1,1,1],
        [1,0,0,0,1,0,0,0,1,0,1,0,1],
        [1,1,1,0,1,0,1,1,1,0,1,0,1],
        [3,0,1,0,1,0,0,0,1,0,1,0,1],
        [1,0,1,0,0,1,1,0,1,0,1,0,1],
        [1,0,1,0,1,1,0,0,1,0,0,0,1],
        [1,0,0,0,1,0,0,0,1,1,1,0,1],
        [1,1,1,1,1,0,1,0,0,0,1,0,1],
        [1,1,1,1,0,0,1,1,1,1,1,0,1],
        [1,0,0,0,0,1,1,0,0,0,0,0,1],
        [1,0,1,1,1,1,0,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1]]
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
            #si en el array hay un 1 es un muro
            if maze[i][j] == 1:
                pygame.draw.rect(screen, black, (j*columnas, i*filas, columnas+1, filas+1))
            #si en el array hay un 2 es la salida del laberinto
            if maze[i][j] == 2:
                pygame.draw.rect(screen, red, (j*columnas, i*filas, columnas+1, filas+1))
            #si en el array hay un 3 es la entrada del laberinto
            if maze[i][j] == 3:
                pygame.draw.rect(screen, green, (j*columnas, i*filas, columnas+1, filas+1))
# Dibujar el personaje 
def draw_player_and_end():
    pygame.draw.circle(screen, blue, (player_pos[0]+player_size//2, player_pos[1]+player_size//2), player_size//2)

# Verificar colisiones
def check_collision():
    #recorremos el array para generar rectangulos y comprobar si colisiona el personaje con los rectangulos
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                #creamos un rectangulo en la posicion del muro para comprobar la colision
                wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)
                #creamos un cuadrado en la posicion del personaje
                circulo_rect = pygame.Rect(player_pos[0], player_pos[1], player_size, player_size)
                #comprobamos si chocan el rectangulo con el cuadrado
                if wall_rect.colliderect(circulo_rect):
                    return True
                #comprobamos si nos chocamos con el cuadrado final para que acabe el bucle
            if maze[i][j] == 2:
                wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)
                if wall_rect.collidepoint(player_pos):
                     return "WIN"
                
# Loop principal del juego
game_over = False
#creamos una variable que nos impida movernos hasta que lleguemos al cuadrado de la salida
mover=False
#pintamos todos los elementos 
screen.fill(white)
draw_maze()
draw_player_and_end()
pygame.display.update()
while not game_over:
    # Manejo de eventos
    for event in pygame.event.get():
        #acaba el programa si cerramos la ventana
        if event.type == pygame.QUIT:
            game_over = True
        #va a entrar en el bucle hasta que el raton del ordenador llegue al cuadrado de entrada al laberinto
        if mover==False:
            #coge la posicion del raton 
            mouse_pos = pygame.mouse.get_pos()
            #recorremos el array para comprobar si "colisionamos con el cuadrado de entrada al laberinto"
            for i in range(len(maze)):
                for j in range(len(maze[i])):
                    if maze[i][j] == 3:
                        wall_rect = pygame.Rect(j*columnas, i*filas, columnas, filas)
                        #comprobamos si la posicion del raton colisiona con el cuadrado de salida
                        if wall_rect.collidepoint(mouse_pos):
                            #-------------------------------------------------------------------------------------------------
                            #guardamos el tiempo especifico para guardar nuestros datos
                            tiempo_intervalo=pygame.time.get_ticks()
                            #-------------------------------------------------------------------------------------------------
                            #cambiamos la variable mover para que se pueda mover el personaje
                            mover=True
        else:
            #coge la posicion del raton 
            mouse_pos = pygame.mouse.get_pos()
            #-----------------------------------------------------------------------------------------------------------------
            #si el tiempo actual - el tiempo de la ultima vez que se ha optenido un dato es igual o mayor que el intervalo captura
            #el dato y reinicia el tiempo 
            if pygame.time.get_ticks()-tiempo_intervalo>=intervalo:
                tiempo_intervalo=pygame.time.get_ticks()
                print(mouse_pos)
            #-----------------------------------------------------------------------------------------------------------------

            target_pos = [mouse_pos[0]-player_size//2, mouse_pos[1]-player_size//2]

            # Actualizar posición del personaje
            if target_pos is not None:
                #guardamos la posicion del personaje en una variable auxiliar 
                aux=[player_pos[0],player_pos[1]]
                
                #movimientos mas lentos 
                x_diff = target_pos[0] - player_pos[0]
                y_diff = target_pos[1] - player_pos[1]
                distance = (x_diff**2 + y_diff**2)**0.5
                if distance != 0:
                    x_move = x_diff / distance * speed
                    y_move = y_diff / distance * speed
                    player_pos[0] += x_move
                    player_pos[1] += y_move

                #comprobamos si hay colision, si hay colision con un muro no se actualiza la posicion del personaje
                #si colisiona con el cuadrado de salida del laberinto acaba el programa
                #y si no colisiona actualiza la posicion del personaje
                if check_collision() == True:
                    player_pos[0]=aux[0]
                    player_pos[1]=aux[1]
                elif check_collision() == "WIN":
                    print("¡Ganaste!")
                    game_over = True
                
                    
                screen.fill(white)
        
                draw_maze()
                draw_player_and_end()


            # Dibujar elementos en pantalla
        
        pygame.display.update()

# Cerrar Pygame
pygame.quit()

