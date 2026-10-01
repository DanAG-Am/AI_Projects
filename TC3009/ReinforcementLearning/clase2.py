'''
RL - RNN y aproximación con reg log (grad descendiente) - Sistema de recomendación de películas
Tendremos una lista de 3 géneros, 3 subgéneros por cada género y 20 películas por subgénero.
Pasos:
1) Mostreamos al usuario género y subgénero antes de recomendarle películas.
2) Preguntamos si le gusta o no la película recomendada: 
    si no le gusta, mostramos las 20 películas del subgénero, 
    y si le gusta, seguimos recomendando películas del mismo subgénero.
3) Actualizamos nuestro modelo de recomendación basado en la retroalimentación del usuario (parametros)
primera iteración muestrear
'''

# Importar librerias
import numpy as np
from scipy.special import softmax

# usuario inventado de prueba
user = {
    'genero_favorito': 'Accion',
    'subgenero_favorito': 'Aventura',
    'peliculas_vistas': []
}

# Definir géneros
genres = ['Accion', 'Comedia', 'Guerra']
# Definir subgéneros por cada género, mapeando a genero
subgenres = {'Accion': ['Aventura', 'Superhéroes', 'Artes Marciales'], 'Comedia': ['Romántica', 'Slapstick', 'Satírica'], 'Guerra': ['Histórica', 'Moderna', 'Futurista']}
# Definir películas por cada subgénero, mapeando a subgénero
peliculas = {
    'Aventura': [
        'Indiana Jones y los cazadores del arca perdida',
        'Piratas del Caribe: La maldición del Perla Negra',
        'La momia',
        'Jurassic Park',
        'El señor de los anillos: La comunidad del anillo',
        'King Kong',
        'Jumanji',
        'Uncharted',
        'Tomb Raider',
        'La leyenda del tesoro perdido',
        'Avatar',
        'Viaje al centro de la Tierra',
        'Las crónicas de Narnia',
        'El Hobbit: Un viaje inesperado',
        'La isla del tesoro',
        'Stardust',
        'Prince of Persia: Las arenas del tiempo',
        'Godzilla',
        'Apocalypto',
        'Mad Max: Fury Road'
    ],

    'Superhéroes': [
        'Iron Man',
        'Spider-Man',
        'The Avengers',
        'Black Panther',
        'Thor',
        'Doctor Strange',
        'Capitán América: El primer vengador',
        'Guardianes de la Galaxia',
        'Deadpool',
        'Wonder Woman',
        'Batman Begins',
        'The Dark Knight',
        'Superman',
        'Aquaman',
        'Shang-Chi y la leyenda de los Diez Anillos',
        'Black Widow',
        'The Batman',
        'Spider-Man: No Way Home',
        'X-Men',
        'Logan'
    ],

    'Artes Marciales': [
        'The Karate Kid',
        'Karate Kid II',
        'Karate Kid III',
        'Ip Man',
        'Ip Man 2',
        'Ip Man 3',
        'El maestro borracho',
        'Fist of Legend',
        'Hero',
        'Tigre y dragón',
        'The Raid: Redemption',
        'The Raid 2',
        'Ong-Bak',
        'Tom-Yum-Goong',
        'Ninja Assassin',
        'Enter the Dragon',
        'Police Story',
        'Drunken Master II',
        'Kung Fu Hustle',
        'Fearless'
    ],

    'Romántica': [
        'The Notebook',
        'Pride and Prejudice',
        'La La Land',
        'Titanic',
        'A Walk to Remember',
        'Crazy Rich Asians',
        'Me Before You',
        'Love Actually',
        'Notting Hill',
        '500 Days of Summer',
        'The Fault in Our Stars',
        'A Star is Born',
        'Call Me by Your Name',
        'Brooklyn',
        'The Vow',
        'Before Sunrise',
        'Before Sunset',
        'Before Midnight',
        'Eternal Sunshine of the Spotless Mind',
        'Amélie'
    ],

    'Slapstick': [
        'La quimera del oro',
        'Tiempos modernos',
        'El gran dictador',
        'La general',
        'Los tres chiflados',
        'Una noche en la ópera',
        'El maquinista de La General',
        'El mundo está loco, loco, loco',
        'La fiesta inolvidable',
        'La pantera rosa',
        'Aterriza como puedas',
        'Top Secret!',
        '¿Y dónde está el piloto?',
        'Loca academia de policía',
        'Dos tontos muy tontos',
        'Ace Ventura: Detective de mascotas',
        'Mr. Bean: La película',
        'La máscara',
        'Johnny English',
        'Scary Movie'
    ],

    'Satírica': [
        'Dr. Strangelove',
        'Network',
        'American Psycho',
        'Borat',
        'El dictador',
        'Gracias por fumar',
        'Wag the Dog',
        'Jojo Rabbit',
        'No mires arriba',
        '¿Teléfono rojo? Volamos hacia Moscú',
        'La muerte de Stalin',
        'The Truman Show',
        'Idiocracia',
        'La gran apuesta',
        'Parásitos',
        'El show de Truman',
        'South Park: Más grande, más largo y sin cortes',
        'El gran Lebowski',
        'Los productores',
        'El club de la pelea'
    ],

    'Histórica': [
        'Gladiador',
        'Troya',
        'Braveheart',
        'El último samurái',
        '300',
        'El patriota',
        'Espartaco',
        'Lawrence de Arabia',
        'Ben-Hur',
        'La lista de Schindler',
        'Dunkerque',
        'Lincoln',
        'Napoleón',
        'Juana de Arco',
        'El reino de los cielos',
        'Apocalypto',
        'El discurso del rey',
        'Pearl Harbor',
        'Éxodo: Dioses y reyes',
        '12 años de esclavitud'
    ],

    'Moderna': [
        'Black Hawk Down',
        'Salvar al soldado Ryan',
        'En tierra hostil',
        'American Sniper',
        'Lone Survivor',
        '13 Horas: Los soldados secretos de Bengasi',
        'Zero Dark Thirty',
        'La caída del halcón negro',
        'Jarhead',
        'Green Zone',
        'Tres Reyes',
        'El único superviviente',
        'Actos de valor',
        'Tears of the Sun',
        'We Were Soldiers',
        'Fury',
        '1917',
        'Hasta el último hombre',
        'Munich',
        'Operación Valquiria'
    ],

    'Futurista': [
        'Star Wars: Episodio IV - Una nueva esperanza',
        'Star Wars: El imperio contraataca',
        'Star Trek',
        'Dune',
        'Dune: Parte Dos',
        'Blade Runner',
        'Blade Runner 2049',
        'Matrix',
        'Terminator',
        'Terminator 2: El juicio final',
        'Aliens',
        'Avatar',
        'El quinto elemento',
        'Elysium',
        'Oblivion',
        'Al filo del mañana',
        'Guerra Mundial Z',
        'Distrito 9',
        'Minority Report',
        'Yo, robot'
    ]
}

lr = 0.1

# parámetros (logits) del policy gradient: uno para géneros, uno por subgéneros de cada género
# y uno por películas de cada subgénero
logits_genero = np.zeros(len(genres))
logits_subgenero = {g: np.zeros(len(subgenres[g])) for g in genres}
logits_pelicula = {sg: np.zeros(len(peliculas[sg])) for sg in peliculas}


def elegir_indice(logits):
    probs = softmax(logits)
    return int(np.random.choice(len(logits), p=probs))


def actualizar_logits(logits, idx, recompensa):
    onehot = np.eye(len(logits))[idx]
    logits += lr * recompensa * (onehot - softmax(logits))


def pedir_feedback(pelicula):
    # menú para el usuario: la recompensa se asigna según si le gusta o no
    while True:
        respuesta = input(f"¿Te gustó '{pelicula}'? (s/n): ").strip().lower()
        if respuesta in ('s', 'si', 'sí'):
            return 1.0
        if respuesta in ('n', 'no'):
            return -1.0
        print("Respuesta inválida, escribe 's' o 'n'.")


def main():
    print("--- Sistema de recomendación de películas ---")
    while True:
        genero_idx = elegir_indice(logits_genero)
        genero = genres[genero_idx]

        subgenero_idx = elegir_indice(logits_subgenero[genero])
        subgenero = subgenres[genero][subgenero_idx]

        pelicula_idx = elegir_indice(logits_pelicula[subgenero])
        pelicula = peliculas[subgenero][pelicula_idx]

        print(f"\nGénero: {genero} | Subgénero: {subgenero}")
        print(f"Te recomendamos: {pelicula}")

        recompensa = pedir_feedback(pelicula)

        actualizar_logits(logits_genero, genero_idx, recompensa)
        actualizar_logits(logits_subgenero[genero], subgenero_idx, recompensa)
        actualizar_logits(logits_pelicula[subgenero], pelicula_idx, recompensa)

        while True:
            continuar = input("\n¿Quieres otra recomendación? (s/n): ").strip().lower()
            if continuar in ('s', 'si', 'sí', 'n', 'no'):
                break
            print("Respuesta inválida, escribe 's' o 'n'.")
        if continuar in ('n', 'no'):
            break

    print("Adiós")
    print("\n--- Probabilidades aprendidas ---")
    print("Géneros:", dict(zip(genres, softmax(logits_genero).round(3))))
    for g in genres:
        print(f"Subgéneros de {g}:", dict(zip(subgenres[g], softmax(logits_subgenero[g]).round(3))))

if __name__ == '__main__':
    main()




