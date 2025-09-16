# Aplicativo de Validação Facial para Android 🤖

Um projeto de exemplo para Android que demonstra um sistema de reconhecimento facial básico, utilizando o **ML Kit** da Google para detecção de rostos e o **TensorFlow Lite** para gerar e comparar embeddings faciais.

## 📜 Descrição

Este aplicativo permite que o usuário realize duas operações principais:
1.  **Cadastro**: Captura o rosto de uma pessoa pela câmera, processa-o com um modelo de Machine Learning para criar uma "assinatura" numérica única (embedding) e a armazena.
2.  **Validação**: Captura um novo rosto e o compara com a assinatura armazenada, determinando se pertence à mesma pessoa com base em um limiar de similaridade.

## ✨ Funcionalidades

* 📸 **Preview da Câmera em Tempo Real** com a API CameraX.
* 🧐 **Detecção de Rosto** utilizando o Google ML Kit para localizar o rosto na imagem.
* 👤 **Cadastro de Rosto** com um único clique para armazenar o embedding facial.
* ✔️ **Validação Facial** comparando o rosto atual com o rosto cadastrado através da Similaridade de Cosseno.
* 📊 **Exibição do Status** em tempo real (sucesso, falha, rosto não detectado).

## 🛠️ Tecnologias Utilizadas

* **Linguagem**: [Kotlin](https://kotlinlang.org/)
* **Câmera**: [Android CameraX](https://developer.android.com/training/camerax) - Para uma API de câmera moderna e simplificada.
* **Detecção Facial**: [Google ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection) - Para encontrar a localização e os contornos do rosto na imagem.
* **Reconhecimento Facial**: [TensorFlow Lite](https://www.tensorflow.org/lite) - Para executar o modelo de Machine Learning (`MobileFaceNet`) no dispositivo.
* **UI**: [View Binding](https://developer.android.com/topic/libraries/view-binding) - Para interagir com os componentes visuais de forma segura.

## 🧠 Como Funciona o Fluxo de Reconhecimento?

O processo de validação segue os seguintes passos:

1.  **Captura**: O `CameraX` tira uma foto a partir da câmera frontal.
2.  **Detecção**: O `ML Kit Face Detector` processa a imagem e retorna as coordenadas da caixa delimitadora (`boundingBox`) do rosto encontrado.
3.  **Pré-processamento**:
    * A imagem original é recortada para conter apenas o rosto.
    * A imagem do rosto é redimensionada para **160x160 pixels**, que é o tamanho de entrada esperado pelo modelo.
    * Os pixels da imagem são normalizados para um intervalo de `[-1, 1]`.
4.  **Geração do Embedding**:
    * O rosto pré-processado é passado para o interpretador do **TensorFlow Lite**, que carrega o modelo `mobilefacenet.tflite`.
    * O modelo transforma a imagem em um vetor de **512 números de ponto flutuante** (o "embedding").
5.  **Comparação**:
    * No modo de **Validação**, o embedding recém-gerado é comparado com o embedding armazenado durante o **Cadastro**.
    * A comparação é feita usando a métrica de **Similaridade de Cosseno**. Se o resultado for superior a um limiar pré-definido (neste código, `0.8f`), os rostos são considerados da mesma pessoa.

## 🚀 Como Executar o Projeto

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
    ```
2.  **Abra no Android Studio:**
    * Abra o Android Studio.
    * Selecione `Open an Existing Project` e navegue até a pasta do projeto clonado.
3.  **Compile e Execute:**
    * Aguarde o Gradle sincronizar as dependências.
    * Execute o aplicativo em um emulador ou dispositivo físico Android (API 21+).
