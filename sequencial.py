import numpy as np
import time
import matplotlib.pyplot as plt 


TAMANHO = 200 
ITERACOES = 1000

def inicializar_grade(n):
    
    grade = np.zeros((n, n))
    
    grade[n//2-10:n//2+10, n//2-10:n//2+10] = 100.0
    return grade

def atualizar_sequencial(grade, n):
    nova_grade = grade.copy()
    
    for i in range(1, n-1):
        for j in range(1, n-1):
            
            nova_grade[i][j] = 0.25 * (grade[i-1][j] + grade[i+1][j] + grade[i][j-1] + grade[i][j+1])
    return nova_grade

if __name__ == "__main__":
    grade = inicializar_grade(TAMANHO)
    
    inicio = time.time()
    for _ in range(ITERACOES):
        grade = atualizar_sequencial(grade, TAMANHO)
    fim = time.time()
    
    print(f"Sequencial - Tempo: {fim - inicio:.4f} segundos")
    plt.imshow(grade, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Mapa de Calor Final")
    plt.savefig('visualizacao_calor.png')
    print("Imagem do calor salva como visualizacao_calor.png")
   