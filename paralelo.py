import numpy as np
import time
import threading


TAMANHO = 200
ITERACOES = 1000
NUM_THREADS = 8

def inicializar_grade(n):
    grade = np.zeros((n, n))
    grade[n//2-10:n//2+10, n//2-10:n//2+10] = 100.0
    return grade

def worker(id_thread, grade_antiga, grade_nova, n, barreira):
    # Divide o trabalho: cada thread pega um pedaço das linhas
    linhas_por_thread = (n - 2) // NUM_THREADS
    inicio = 1 + id_thread * linhas_por_thread
    fim = inicio + linhas_por_thread
    
    # A última thread pega o resto (se a divisão não for exata)
    if id_thread == NUM_THREADS - 1:
        fim = n - 1

    for _ in range(ITERACOES):
        # Fase de Cálculo
        for i in range(inicio, fim):
            for j in range(1, n-1):
                grade_nova[i][j] = 0.25 * (grade_antiga[i-1][j] + grade_antiga[i+1][j] + grade_antiga[i][j-1] + grade_antiga[i][j+1])
        
        # Espera todas as threads terminarem o cálculo
        barreira.wait()
        
        # Fase de Atualização (apenas uma thread precisa copiar ou trocamos referências)
        # Para simplificar, a thread 0 copia o array para o antigo para a proxima iteração
        if id_thread == 0:
            np.copyto(grade_antiga, grade_nova)
        
        # Espera a cópia terminar
        barreira.wait()

if __name__ == "__main__":
    grade_antiga = inicializar_grade(TAMANHO)
    grade_nova = grade_antiga.copy()
    
    barreira = threading.Barrier(NUM_THREADS)
    threads = []

    inicio_tempo = time.time()

    for i in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(i, grade_antiga, grade_nova, TAMANHO, barreira))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    fim_tempo = time.time()
    print(f"Paralelo ({NUM_THREADS} threads) - Tempo: {fim_tempo - inicio_tempo:.4f} segundos")