import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import openpyxl
from openpyxl.drawing.image import Image
import tkinter as tk
from tkinter import filedialog

# Função para carregar os dados
def load_data():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    data = pd.read_excel(file_path)
    return data, file_path

# Função para salvar os resultados
def save_results():
    root = tk.Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    return save_path

def main():
    data, file_path = load_data()

    # Listando variáveis e selecionando características e variável alvo
    print("Variáveis disponíveis:")
    for i, col in enumerate(data.columns):
        print(f"{i}: {col}")

    feature_indices = input("Digite os números das variáveis de entrada separados por espaço: ")
    target_index = int(input("Digite o número da variável alvo: "))
    standardize_choice = int(input("Deseja padronizar os dados? (1 para Sim, 2 para Não): "))
    qqplot_choice = int(input("Deseja que os QQ plot sejam gerados com os dados padronizados? (1 para Sim, 2 para Não): "))

    feature_columns = [data.columns[int(i)] for i in feature_indices.split()]
    target_column = data.columns[target_index]

    print(f"Características selecionadas: {feature_columns}")
    print(f"Variável alvo: {target_column}")

    # Separando as variáveis de entrada e alvo
    X = data[feature_columns]
    y = data[target_column]

    # Normalização das variáveis de entrada, se selecionado
    if standardize_choice == 1:
        X = (X - X.mean()) / X.std()

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinando o modelo KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Avaliação do modelo
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions)

    # Criando um arquivo Excel para os resultados
    save_path = save_results()
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        # Salvando os dados originais com as previsões
        data['Predictions'] = model.predict(X)
        data.to_excel(writer, sheet_name='Dados e Previsões', index=False)

        # Salvando as métricas de avaliação
        metrics_df = pd.DataFrame(report).transpose()
        metrics_df.loc['accuracy'] = accuracy
        metrics_df.to_excel(writer, sheet_name='Métricas de Avaliação')

        # Gráficos de normalidade (QQ plot) e Matriz de Confusão
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sm.qqplot(predictions, line='s', ax=axes[0])
        axes[0].set_title('QQ Plot das Previsões')
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Matriz de Confusão')
        plt.suptitle(f'Métricas de Avaliação (Acurácia: {accuracy:.2f})', y=0.95)
        plt.savefig('avaliacao_knn.png')
        plt.close(fig)

        # Inserindo a imagem no Excel
        wb = writer.book
        ws = wb['Métricas de Avaliação']
        img = Image('avaliacao_knn.png')
        ws.add_image(img, 'H2')

        # Estatísticas descritivas das variáveis selecionadas
        desc_stats = data[feature_columns].describe()
        desc_stats.to_excel(writer, sheet_name='Estatísticas Descritivas')

        # Gráficos QQ plot para cada variável selecionada
        if qqplot_choice == 1:
            data_for_qqplot = (data[feature_columns] - data[feature_columns].mean()) / data[feature_columns].std()
        else:
            data_for_qqplot = data[feature_columns]

        for col in data_for_qqplot.columns:
            fig = sm.qqplot(data_for_qqplot[col], line='s')
            plt.title(f'QQ Plot - {col}')
            shapiro_test = stats.shapiro(data_for_qqplot[col])
            plt.text(-2, 2, f'p-valor: {shapiro_test.pvalue:.4f}', fontsize=12, verticalalignment='top')
            plt.savefig(f'qqplot_{col}.png')
            plt.close(fig)
            img = Image(f'qqplot_{col}.png')
            ws.add_image(img, f'A{ws.max_row + 2}')

    print(f"Resultados salvos em {save_path}")

if __name__ == "__main__":
    main()
