import socket
import pickle
import numpy as np
import time


HOST = '127.0.0.1'
PORTA = 65432
NUM_CLIENTES = 2 
TAMANHO = 200
ITERACOES = 100 

def inicializar_grade(n):
    grade = np.zeros((n, n))
    grade[n//2-10:n//2+10, n//2-10:n//2+10] = 100.0
    return grade

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORTA))
    server.listen(NUM_CLIENTES)
    print(f"Servidor aguardando {NUM_CLIENTES} clientes...")

    conexoes = []
    for _ in range(NUM_CLIENTES):
        conn, addr = server.accept()
        conexoes.append(conn)
        print(f"Conectado por {addr}")

    grade = inicializar_grade(TAMANHO)
    
    inicio_tempo = time.time()

    
    linhas_por_cliente = (TAMANHO - 2) // NUM_CLIENTES

    for it in range(ITERACOES):
        
        for i, conn in enumerate(conexoes):
            inicio = 1 + i * linhas_por_cliente
            fim = inicio + linhas_por_cliente
            if i == NUM_CLIENTES - 1: fim = TAMANHO - 1
            
           
            dados = {
                'subgrade': grade[inicio-1:fim+1, :], 
                'range': (inicio, fim)
            }
            conn.sendall(pickle.dumps(dados)) 
            conn.send(b'END') 

        # 2. Receber resultados e montar
        for i, conn in enumerate(conexoes):
            buffer = b""
            while True:
                part = conn.recv(4096)
                buffer += part
                if b'END' in part:
                    buffer = buffer.replace(b'END', b'')
                    break
            
            resultado = pickle.loads(buffer)
            linhas_calculadas = resultado['processado']
            inicio, fim = resultado['range']
            
            
            grade[inicio:fim, :] = linhas_calculadas[1:-1, :]

    fim_tempo = time.time()
    print(f"Distribuído - Tempo: {fim_tempo - inicio_tempo:.4f} segundos")
    
   
    for conn in conexoes:
        conn.close()

if __name__ == "__main__":
    main()