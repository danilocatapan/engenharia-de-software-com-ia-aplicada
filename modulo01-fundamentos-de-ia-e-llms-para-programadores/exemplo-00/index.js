import tf from '@tensorflow/tfjs-node';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // Primeira cada da rede:
    // entrada de 7 posições (idade normalizada + 3 cores + 3 localizações)

    // 80 neuronios = aqui coloqueo tudo isso, pq tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode
    // e consequentemente, mais processamento ela vai usar

    // A ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neuronio é positiva, passa pra frente!
    // se for zzero o negaiva, pode jogar fora, não vai servir para nada.
    model.add(tf.layers.dense({
        inputShape: [7], // 7 características de entrada
        units: 80, // número de neurônios na camada oculta
        activation: 'relu' // função de ativação
    }))

    // Saída: 3 neurônios (premium, medium, basic) com softmax para classificação
    model.add(tf.layers.dense({
        units: 3, // número de classes de saída
        activation: 'softmax' // função de normalização para classificação
    }))

    // Compilando o modelo
    // optimizer Adan ( Adaptive Moment Estimation) é um otimizador que ajusta os pesos da rede de forma eficiente durante o treinamento.
    // é um treinador pessoal moderno para redes neurais
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy 
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com a respota certa
    // a categoria premuium será sempre [1, 0, 0], medium [0, 1, 0] e basic [0, 0, 1]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // Exemplo classico: classificação de imagens, recomendação, categorização de usuários, etc.
    // qualquer coisa em quuue a resposta certa é "apenas uma entre várias possíveis"

    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']})

    // Treinando o modelo
    await model.fit(inputXs, outputYs, {
        verbose: 0, // 0 para não mostrar o progresso do treinamento
        epochs: 100, // número de vezes que o modelo vai passar por todo o dataset
        shuffle: true, // embaralha os dados a cada época para melhorar o aprendizado
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch: ${epoch + 1}: loss = ${logs.loss.toFixed(4)}`)
            }
        }
    })

    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor (tfjs)
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição (output será um vetor de 3 probabilidades, uma para cada categoria)
    const prediction = model.predict(tfInput)
    const predArray = await prediction.array() // converte o tensor de volta para array js
    return predArray[0].map((prob, index) => ({ prob, index }))
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// quanto mais dados melhor!
// assim o algoritmo consegue entender melhor os padrões e fazer previsões mais precisas
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "zé", idade: 28, cor: "verde", localizacao: "Curitiba" }
// Normaliando a idade da nova pessoa usando o mesmo padrão do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 -25) / (40 - 25) = 0.2

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1,   // azul
        0,   // vermelho
        0,   // verde
        1,   // São Paulo
        0,   // Rio
        0    // Curitiba
    ]
]

const predicitions = await predict(model, pessoaTensorNormalizado)
const results = predicitions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
    
console.log(results)