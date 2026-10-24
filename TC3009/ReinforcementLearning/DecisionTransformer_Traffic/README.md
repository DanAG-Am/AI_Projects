# Simulación de tráfico por zonas

## Ejecutar el agente y generar los datos

Desde esta carpeta, ejecuta:

```bash
python3 decision_transformer_zones.py
```

El script entrena el Decision Transformer, evalúa las políticas, genera `simulation_data.json` y abre la animación de Matplotlib.

## Abrir la visualización web

Con los datos generados, inicia un servidor local desde esta carpeta:

```bash
python3 -m http.server 8000
```

Abre <http://localhost:8000/simulation.html>. La página crea una escena WebGL con Three.js: una cuadrícula urbana ampliada con edificios de distintas alturas, corredores, zonas de demanda y vehículos 3D. Veinticuatro autos de tráfico circulan continuamente por tres avenidas, además de los vehículos de la flota controlada por el agente. Los semáforos tienen ciclos rojo/amarillo/verde y los pasajeros indican solicitudes cubiertas y pendientes junto a cada zona.

Cada paso del episodio representa una hora, de 07:00 a 19:00. La vista indica la franja del día, las solicitudes por zona, los autos disponibles, los viajes atendidos y la demanda que queda esperando. La congestión de hora pico también reduce la velocidad de circulación. Puedes arrastrar para rotar, usar la rueda o +/- para zoom, WASD para mover la cámara y las flechas para rotarla. Los controles permiten pausar, reiniciar o seleccionar una hora.

## Utilidad para operaciones de movilidad

Una simulación urbana permite explorar decisiones operativas en un entorno controlado antes de probarlas en una ciudad real. Puede ayudar a equipos de movilidad a:

- Comparar estrategias para reubicar vehículos hacia zonas donde se espera mayor demanda.
- Identificar desequilibrios entre solicitudes y disponibilidad, y visualizar dónde podría quedar demanda sin atender.
- Analizar cómo la hora del día, la congestión y las señales de tránsito pueden afectar la circulación y la cobertura.
- Evaluar indicadores como viajes atendidos, solicitudes pendientes y costo de reubicación al comparar políticas.
- Comunicar escenarios de planificación de flota a equipos de operaciones y producto mediante una visualización 3D compartida.

Esta versión es un prototipo educativo con demanda y tráfico sintéticos. Sirve para formular hipótesis y demostrar conceptos, pero no sustituye la validación con datos representativos, restricciones operativas reales ni modelos calibrados antes de tomar decisiones de negocio. La cobertura por zona es agregada; no se despachan ni se siguen viajes individuales.

La página carga Three.js desde jsDelivr, por lo que necesita conexión a internet además del servidor local que sirve el episodio JSON.

## Interpretación del movimiento

El agente toma decisiones de **reubicación entre zonas**, no de conducción en cada instante. Por eso, “sin movimiento” significa que el auto conserva su zona asignada; no significa que el vehículo esté detenido. Los autos de la flota recorren continuamente las calles de su zona y respetan semáforos; cuando el agente cambia su asignación, el vehículo atraviesa la avenida hasta la zona destino. La demanda y las solicitudes pendientes aproximan la operación de una plataforma de movilidad, mientras que la recompensa actual de RL sigue midiendo viajes atendidos menos el costo de reubicación. No representa todavía despacho de cada viaje individual ni datos reales de una empresa.