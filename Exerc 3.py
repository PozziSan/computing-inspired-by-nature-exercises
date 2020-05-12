import random
import math
import numpy as np
import matplotlib.pyplot as plt

tam_populacao = 1000


def inicializa_pe():
    pe = []
    for i in range(tam_populacao):
        for j in range(1):# gera X
            individuoX = round(random.uniform(-5, 5), 7)
        for k in range(1):# gera Y
            individuoY = round(random.uniform(-5, 5), 7)
        pe.append(individuoX)
        pe.append(individuoY)
    #print('pe', pe)
    return pe


def funcao(pe):
    lista_xy = []
    for i in range(0, len(pe), 2):
        x = pe[i]
        y = pe[i+1]
        #print(x)
        #print(y)
        fxy = round((1 - x)**2 + 100 * (y - x**2)**2, 7)
        #print('resultado:', fxy)
        lista_xy.append(fxy)
    #print('listaXY', lista_xy)
    return lista_xy


def comparar(lista_xy):
    pontuacaoindividuo = []
    #for i in range(len(pe)):
    for j in range(len(lista_xy)):
        f_aptidao = round(1 / (lista_xy[j] + 1), 7)
        #print('função de aptidão:', f_aptidao)
        pontuacaoindividuo.append(f_aptidao)
    #print('pontuação do individuo', pontuacaoindividuo)
    return pontuacaoindividuo


def selecionar(pe, pontuacaoindividuo):
    cruzamentos = []
    acerto_individual = []
    for i in range(len(pontuacaoindividuo)):
        acerto_individual.append(pontuacaoindividuo[i] / sum(pontuacaoindividuo))
    selecao_roleta = []
    for i in range(len(pontuacaoindividuo)):
        if i == 0:
            selecao_roleta.append(acerto_individual[i])
        else:
            selecao_roleta.append(acerto_individual[i] + selecao_roleta[-1])
    for index in range(len(pe)):
        sorteio = random.random()
        for i in range(len(selecao_roleta)):
            if sorteio < selecao_roleta[i]:
                cruzamentos.append(pe[i])
                break
    #print('sorteio', sorteio)
    #print('acerto individual', acerto_individual)
    #print('cruzamentos', cruzamentos)

    return cruzamentos


def mutacao(pop):

    #print('pop',pop)
    for i in range(0, len(pop), 2):
        #print('pop', pop)
        mutacionar = random.random()
        #print('mutacionar', mutacionar)
        if mutacionar == mutacionar:
            valor_aleatorio = random.randint(0, 1)
            if valor_aleatorio == 0:
                truncador = math.trunc(pop[i + valor_aleatorio])
                if truncador == -5:
                    truncador += 1
                elif truncador == 5:
                    truncador -= 1
                else:
                    sobe_desce = random.randint(0, 1)
                    if sobe_desce == 0:
                        truncador += 1
                    else:
                        truncador -= 1
                #print('trunc', truncador)
                #print('pop[i] antes', pop[i + valor_aleatorio])
                pop[i] = pop[i] - math.trunc(pop[i + valor_aleatorio]) + truncador
                if pop[i] > 5:
                    pop[i] = 5
                elif pop[i] < -5:
                    pop[i] = -5
                #print('pop[i] depois', pop[i] + valor_aleatorio)

            elif valor_aleatorio == 1:
                modulador = round(pop[i] % 1, 7)
                #print('modulador', modulador)
                escolha = random.randint(0, 6)
                modulador = round(modulador + (10**(-1 + (escolha * -1))), 7)
                #print('depois', modulador)
                if pop[i] > 5:
                    pop[i] = 5
                elif pop[i] < -5:
                    pop[i] = -5

    return pop


if __name__ == '__main__':
    contador = 0
    populacao = inicializa_pe()                                  # gera a população
    medios = []
    melhores_minimos = []
    melhores_maximos = []
    medios2 = []
    melhores_minimos2 = []
    melhores_maximos2 = []
    while contador <= 50:
        var_funcao = funcao(populacao)                           # calcula a função com a população gerada
        var_comparar = comparar(var_funcao)                      # compara a população
        melhores_minimos.append(min(var_funcao))                 # compara o mínimo da população
        melhores_maximos.append(max(var_funcao))                 # compara o máximo da população
        medios.append(np.mean(var_funcao))                       # compara a médio da população
        melhores_minimos2.append(min(var_comparar))              # compara a aptidão mínima da população
        melhores_maximos2.append(max(var_comparar))              # compara a aptidão máxima da população
        medios2.append(np.mean(var_comparar))                    # compara a aptidão média da população
        populacao = selecionar(populacao, var_comparar)          # seleciona os melhores
        populacao = mutacao(populacao)                           # muta as populações
        contador += 1
        #print('var sel', populacao)
    print('minimos',melhores_minimos)
    print('medios', medios)
    print('maximos', melhores_maximos2)
    print('minimos2', melhores_minimos2)
    print('medios2', medios2)
    print('maximos2', melhores_maximos2)
    plt.plot(melhores_maximos, color='blue')
    plt.plot(medios, color='green')
    plt.plot(melhores_minimos, color='red')
    plt.ylabel('Aptidão')
    plt.xlabel('Gerações')
    #plt.axis([0, 50, min(melhores_minimos), max(melhores_maximos)])
    plt.grid(True)
    plt.show()

    plt.plot(melhores_maximos2, color='blue')
    plt.plot(medios2, color='green')
    plt.plot(melhores_minimos2, color='red')
    plt.ylabel('Aptidão')
    plt.xlabel('Gerações')
    #plt.axis([0, 50, min(melhores_minimos2), max(melhores_maximos2)])
    plt.grid(True)
    plt.show()