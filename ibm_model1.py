from re import S
import numpy as np
import pandas as pd

##############################
######### DATASETS ###########
##############################

# French dataset for Lecture 7 Assignment
input = {
    "the house": "la maison", "the flower": "la fleur", 
    "the blue house": "la maison bleu", "the blue flower": "la fleur bleu", 
    "blue apple": "pomme bleu"
}

# Spanish dataset used in class
# input = {
#     "white house": "casa blanca",
#     "my house": "mi casa"
# }

# Spanish dataset with 54 sentance pairs for extra credit
# input = {
#     "the red car": "el auto rojo", "that car is new": "ese auto es nuevo",
#     "soccer is the best sport": "fútbol es el mejor deporte", "ball for soccer": "balón de fútbol",
#     "he runs fast": "él corre rápido", "the child runs fast": "el niño corre rápido",
#     "the blue car goes fast": "el auto azúl va rápido", "blue is his favorite color": "azúl es su color favorito",
#     "the white car is big": "el auto blanca es grande", "his clothing is white and blue": "su ropa es blanca y azúl",
#     "white snow": "nieve blanca", "the water is pretty": "el agua es linda",
#     "the tomato tastes yummy": "el tomate sabe rico", "we bought food": "nosotros compramos comida",
#     "read a book": "leer un libro", "check the weather": "fijar el clima",
#     "spend time with his friend": "pasar tiempo con su amigo", "I love you": "yo te amo",
#     "I gave you a red car": "yo te di un auto rojo", "I gave you a kiss": "yo te di un beso",
#     "drive a car": "manejar un auto", "play a sport": "jugar un deporte",
#     "cook in the room": "cocinar en el cuarto", "cook food": "cocinar comida",
#     "apples and bananas": "manzanas y bananas", "rotten apples": "manzanas podridas",
#     "rotten grapes": "uvas podridas", 
#     "she jumps up and down": "ella salta arriba y abajo", "the dog jumps high": "el perro salta alto",
#     "he is funny": "él es gracioso", "they are funny": "ellos son graciosos",
#     "eat fruits and vegetables": "comer frutas y verduras",
#     "vegetables are healthy": "verduras son saludables", "fruits are sweet": "frutas son dulces",
#     "my people": "mi gente", "I have my very big family": "yo tengo mi familia muy grande",
#     "my family plays soccer": "mi familia juega fútbol", "that dog is very small": "ese perro es muy chico",
#     "this place is awesome": "este lugar es genial", "this book is long": "este libro es largo",
#     "my house is close": "mi casa es cerca", "my place is far": "mi lugar es lejos",
#     "they are far from my family": "ellos son lejos de mi familia",
#     "this sport is fun": "este deporte es divertido", "soccer is a sport": "fútbol es un deporte",
#     "I love my friend": "yo amo mi amigo", "the friend is very close": "el amigo es muy cerca",
#     "I ate three bananas": "yo comí tres bananas", "I ate vegetables": "yo comí verduras",
#     "white house": "casa blanca", "my house": "mi casa"
# }

source_set = set()
target_set = set()

for sentance in list(input):
    source = sentance.split()
    target = input[sentance].split()
    for word in source:
        source_set.add(word) 
    for word in target:
        target_set.add(word)

dictionary_init = np.ones(shape=(len(source_set), len(target_set)))
dictionary = pd.DataFrame(dictionary_init, columns=source_set, index=target_set)

num_iterations = 0

while True:
    prob_prev = np.argmax(dictionary.to_numpy(), axis=1)
    count_init = np.zeros(shape=(len(source_set), len(target_set)))
    count = pd.DataFrame(count_init, columns=source_set, index=target_set)
    s_total = {}
    total_init = np.zeros(shape=len(target_set))
    total = pd.DataFrame(total_init, columns=[0], index=target_set)
    for sentance in list(input):
        for source in sentance.split():
            s_total[source] = 0
            for target in input[sentance].split():
                s_total[source] += dictionary[source][target]

        for source in sentance.split():
            for target in input[sentance].split():
                count[source][target] += dictionary[source][target] / s_total[source]
                total[0][target] += dictionary[source][target] / s_total[source]

    for target in target_set:
        for source in source_set:
            dictionary[source][target] = count[source][target] / total[0][target]

    prob_curr = np.argmax(dictionary.to_numpy(), axis=1)
    
    num_iterations += 1
    print(dictionary)
    if (prob_prev == prob_curr).all():
        break

print("Number of iterations until convergence: {}\n".format(num_iterations))
col = dictionary.columns
row = dictionary.index
num = 0
for index in prob_curr:
    print("{} = {}".format(row[num], col[index]))
    num += 1




