let modelo;
let entrenado = false;

async function entrenarModelo() {
  const xs = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const ys = tf.tensor2d(xs.dataSync().map(x => 2 * x + 6), [9, 1]);

  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  modelo.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

  document.getElementById('resultado').innerText = "Entrenando modelo...";

  await modelo.fit(xs, ys, {
    epochs: 350,
    callbacks: {
      onTrainEnd: () => {
        entrenado = true;
        document.getElementById('resultado').innerText = "¡Modelo entrenado! Ya puedes predecir.";
      }
    }
  });
}

async function predecir() {
  const x = parseFloat(document.getElementById('inputX').value);
  if (!entrenado) {
    document.getElementById('resultado').innerText = "Primero entrena el modelo.";
    return;
  }

  const input = tf.tensor2d([x], [1, 1]);
  const output = modelo.predict(input);
  const y = (await output.data())[0];

  document.getElementById('resultado').innerText = `El resultado de predecir para ${x} es: ${y}`;
}
