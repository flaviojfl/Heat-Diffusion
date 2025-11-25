import socket
import pickle
import numpy as np

HOST = '127.0.0.1'
PORTA = 65432

def calcular(subgrade):
    
    h, w = subgrade.shape
    nova_subgrade = subgrade.copy()
    for i in range(1, h-1):
        for j in range(1, w-1):
            nova_subgrade[i][j] = 0.25 * (subgrade[i-1][j] + subgrade[i+1][j] + subgrade[i][j-1] + subgrade[i][j+1])
    return nova_subgrade

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORTA))
    print("Cliente conectado e calculando...")

    while True:
        try:
            buffer = b""
            while True:
                part = client.recv(4096)
                if not part: break
                buffer += part
                if b'END' in part:
                    buffer = buffer.replace(b'END', b'')
                    break
            
            if not buffer: break

            dados = pickle.loads(buffer)
            subgrade = dados['subgrade']
            range_info = dados['range']

           
            processado = calcular(subgrade)

           
            resposta = {'processado': processado, 'range': range_info}
            client.sendall(pickle.dumps(resposta))
            client.send(b'END')
            
        except Exception as e:
            print("Conexão encerrada ou erro.")
            break

    client.close()

if __name__ == "__main__":
    main()