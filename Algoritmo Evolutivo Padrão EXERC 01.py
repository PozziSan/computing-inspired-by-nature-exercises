import matplotlib.pyplot as plt
import random
import time

bitstring = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]
tam_populacao = 100


#inicializa os indivíduos da populução
def inicializa_pe():
    pe = []
    for i in range(tam_populacao):
        individuo = []
        for j in range(12):
            individuo.append(random.randint(0, 1))
            #print(pe, individuo, i, j)
        pe.append(individuo)
    #print(pe)
    return pe

#faz a comparação com dos indivíduos com o bitstring
def comparar(pe):
    pontuacaoindividuo = []
    for i in range(len(pe)):
        #print('pe[i]', pe[i])
        contador = 0
        for j in range(len(pe[i])):
            if pe[i][j] == bitstring[j]:
                contador += 1
        pontuacaoindividuo.append(contador)
        #print(contador, pe[i])

    return pontuacaoindividuo


#seleciona os individuos para cruzamento com base na pontuação de acertos dos bits
def selecionar(pe, pontuacaoindividuo):
    cruzamentos = []
    acerto_individual = []

    for i in range(len(pontuacaoindividuo)):
        if sum(pontuacaoindividuo) == 0:   #se a soma for 0 será passado para frente
            return pe
        else:
            acerto_individual.append(pontuacaoindividuo[i] / sum(pontuacaoindividuo))
    selecao_roleta = []
    for i in range(len(pontuacaoindividuo)):
        if i == 0:
            selecao_roleta.append(acerto_individual[i])
        else:
            selecao_roleta.append(acerto_individual[i] + selecao_roleta[-1])
    for index in range(tam_populacao):
        sorteio = random.random()
        for i in range(len(selecao_roleta)):
            if sorteio < selecao_roleta[i]:
                cruzamentos.append(pe[i])
                break
    #print('cruzamento', cruzamentos)
    #cruzamentos.append(crossover(cruzamentos))
    cruzamentos = crossover(cruzamentos)
    return cruzamentos


#realiza o crossover pegando as metades dos individuos e gerando dois novos indivíduos
def crossover(escolhidos):
    for i in range(0, len(escolhidos), 2):
        #print('antes ', escolhidos[i], escolhidos[i+1])
        auxiliar = escolhidos[i][:6]
        escolhidos[i][:6] = escolhidos[i + 1][:6]
        escolhidos[i + 1][:6] = auxiliar
        #print('depois', escolhidos[i], escolhidos[i+1])
    return escolhidos


#Realiza o calculo da possível mutação e a mutação
def mutacao(pop):
    for i in range(len(pop)):
        mutacionar = random.random()
        #print('mutacionar', mutacionar)
        if mutacionar <= 0.001:
            valor_aleatorio = random.randint(0, 11)
            #print(pop[i])
            #print('VA', valor_aleatorio)
            if pop[i][valor_aleatorio] == 1:
                pop[i][valor_aleatorio] = 0
            else:
                pop[i][valor_aleatorio] = 1
            #print('novo individuo com mutação', pop[i])

    return pop


if __name__ == '__main__':
    start = time.time()
    populacao = inicializa_pe()
    #print('POP', populacao)
    variaval2 = comparar(populacao)
    #print('vairvel 2', variaval2)
    geracao = 0
    while 12 not in variaval2:
        populacao = selecionar(populacao, variaval2)
        #print('popula', populacao)
        populacao = mutacao(populacao)
        variaval2 = comparar(populacao)
        geracao += 1
    print('n de gerações', geracao)
    print('população', populacao)
    end = time.time()

    print(end - start)


