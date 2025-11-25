import matplotlib.pyplot as plt
metodos = ['Sequencial', 'Paralelo (2 threads)', 'Paralelo (4 threads)', 'Distribuído (Estimado)']
tempos = [46.0, 42.52, 45.73, 23.76] 

plt.figure(figsize=(10, 6))
barras = plt.bar(metodos, tempos, color=['blue', 'orange', 'red', 'green'])

plt.xlabel('Método de Execução')
plt.ylabel('Tempo (segundos)')
plt.title('Comparativo de Tempo: Simulação de Calor (1000 iterações)')
plt.grid(axis='y', linestyle='--', alpha=0.7)


for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2., altura,
             f'{altura:.2f}s',
             ha='center', va='bottom')

plt.savefig('grafico_comparativo.png') 
print("Gráfico salvo como grafico_comparativo.png")