import { Network } from './network/Network.js';
import { CanvasRenderer } from './ui/CanvasRenderer.js';
import { UIController } from './ui/UIController.js';

// Instanciar el núcleo matemático
const network = new Network('mse', 0.03);
window.network = network;

// Construir una arquitectura inicial
network.buildArchitecture(1, [1], 1);

// Instanciar el motor de renderizado
const canvasElement = document.getElementById('neural-canvas');
const renderer = new CanvasRenderer(canvasElement);

// Instanciar el controlador central de eventos e interfaz
const ui = new UIController(network, renderer);

// Pintar la red por primera vez
renderer.render(network);