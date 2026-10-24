let THREE;

const sceneShell = document.querySelector("#scene-shell");
const playButton = document.querySelector("#play-button");
const resetButton = document.querySelector("#reset-button");
const slider = document.querySelector("#step-slider");
const stepOutput = document.querySelector("#step-output");
const agentValue = document.querySelector("#agent-value");
const actionValue = document.querySelector("#action-value");
const rewardValue = document.querySelector("#reward-value");
const totalValue = document.querySelector("#total-value");
const message = document.querySelector("#message");
const timeValue = document.querySelector("#time-value");
const signalValue = document.querySelector("#signal-value");

const ZONE_X = [-12, 0, 12];
const CITY_STREET_X = [-30, -20, -10, 10, 20, 30];
const TRAFFIC_CORRIDORS = [0, -12, 18];
const CAR_COLORS = [0x2474d2, 0xe45748, 0x35a56c, 0xf1ad32, 0x8658c7, 0x27a5a8];
const START_HOUR = 7;
const SIGNAL_CYCLE_SECONDS = 12;
const SLOT_OFFSETS = [
  [-2.7, 3.9],
  [0, 3.9],
  [2.7, 3.9],
  [-2.7, 6.4],
  [0, 6.4],
  [2.7, 6.4],
];
const CRUISE_SPEED = 3.2;

let episode;
let cars = [];
let ambientCars = [];
let trafficSignals = [];
let passengerMarkers = [];
let zoneLabels = [];
let currentStep = 0;
let timer;
let cameraRadius = 76;
let cameraTheta = 0.68;
let cameraPhi = 0.9;
let dragStart;
let renderer;
let scene;
let camera;
let cameraTarget;
let sunlight;
let simulationStartedAt = performance.now();

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
}

function addBox(parent, position, size, material, castShadow = true) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(size[0], size[1], size[2]),
    material,
  );
  mesh.position.set(position[0], position[1], position[2]);
  mesh.castShadow = castShadow;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function createTextSprite(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.fillStyle = "rgba(255, 255, 255, 0.9)";
  context.beginPath();
  context.roundRect(8, 8, 496, 112, 24);
  context.fill();
  context.fillStyle = "#172033";
  context.font = "bold 36px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(text, 256, 64, 460);

  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true }),
  );
  sprite.scale.set(5.9, 1.48, 1);
  return sprite;
}

function updateZoneLabel(zone, demand, vehicles, served, waiting) {
  const label = zoneLabels[zone];
  const nextSprite = createTextSprite(
    `Z${zone + 1} · DEM ${demand} · OK ${served} · ESP ${waiting}`,
  );
  nextSprite.position.copy(label.position);
  scene.remove(label);
  label.material.map.dispose();
  label.material.dispose();
  zoneLabels[zone] = nextSprite;
  scene.add(nextSprite);
}

function createTrafficSignal(x, z, axis, coordinate, laneCoordinate, phaseOffset) {
  const signal = new THREE.Group();
  const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x33404a, roughness: 0.75 });
  const housingMaterial = new THREE.MeshStandardMaterial({ color: 0x202a32, roughness: 0.5 });
  const bulbs = [
    new THREE.MeshStandardMaterial({ color: 0x661e1c, emissive: 0x100000 }),
    new THREE.MeshStandardMaterial({ color: 0x66531a, emissive: 0x100000 }),
    new THREE.MeshStandardMaterial({ color: 0x1d5b30, emissive: 0x100000 }),
  ];

  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, 3.1, 8), poleMaterial);
  pole.position.y = 1.55;
  pole.castShadow = true;
  signal.add(pole);

  const housing = new THREE.Mesh(new THREE.BoxGeometry(0.42, 1.05, 0.38), housingMaterial);
  housing.position.set(0, 3.35, 0);
  housing.castShadow = true;
  signal.add(housing);

  const lamps = bulbs.map((material, index) => {
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 10), material);
    lamp.position.set(0, 3.67 - index * 0.32, 0.21);
    signal.add(lamp);
    return lamp;
  });

  signal.position.set(x, 0, z);
  scene.add(signal);
  trafficSignals.push({
    axis,
    coordinate,
    laneCoordinate,
    phaseOffset,
    bulbs,
    lamps,
  });
}

function createCar(id, colorIndex = id - 1) {
  const car = new THREE.Group();
  const paint = new THREE.MeshStandardMaterial({
    color: CAR_COLORS[colorIndex % CAR_COLORS.length],
    roughness: 0.34,
    metalness: 0.15,
  });
  const glass = new THREE.MeshStandardMaterial({
    color: 0xa9e0f4,
    roughness: 0.18,
    metalness: 0.1,
  });
  const dark = new THREE.MeshStandardMaterial({ color: 0x202936, roughness: 0.8 });
  const lights = new THREE.MeshStandardMaterial({
    color: 0xfff0b2,
    emissive: 0xffca56,
    emissiveIntensity: 0.35,
  });

  addBox(car, [0, 0.42, 0], [1.55, 0.48, 0.88], paint);
  addBox(car, [-0.08, 0.77, 0], [0.83, 0.42, 0.72], glass);
  addBox(car, [0.76, 0.48, -0.27], [0.13, 0.16, 0.12], lights, false);
  addBox(car, [0.76, 0.48, 0.27], [0.13, 0.16, 0.12], lights, false);

  for (const x of [-0.48, 0.48]) {
    for (const z of [-0.46, 0.46]) {
      const wheel = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22, 0.22, 0.16, 14),
        dark,
      );
      wheel.rotation.x = Math.PI / 2;
      wheel.position.set(x, 0.23, z);
      wheel.castShadow = true;
      car.add(wheel);
    }
  }

  car.position.y = 0.08;
  car.userData.id = id;
  car.userData.zone = 0;
  car.userData.slot = 0;
  car.userData.motion = null;
  car.userData.cruiseRoute = null;
  car.userData.cruiseRouteLength = 0;
  car.userData.cruiseStartedAt = performance.now() + id * 250;
  car.userData.routeDistance = 0;
  car.userData.lastUpdate = performance.now();
  scene.add(car);
  return car;
}

function createPassenger(zone, index) {
  const person = new THREE.Group();
  const personMaterial = new THREE.MeshStandardMaterial({
    color: 0xf08a45,
    roughness: 0.8,
    emissive: 0x351300,
  });
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.18, 0.48, 4, 8),
    personMaterial,
  );
  body.position.y = 0.43;
  body.castShadow = true;
  person.add(body);

  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.17, 12, 10),
    new THREE.MeshStandardMaterial({ color: 0xf1c8a4, roughness: 0.9 }),
  );
  head.position.y = 0.82;
  head.castShadow = true;
  person.add(head);
  person.position.set(ZONE_X[zone] - 2.6 + index * 0.65, 0.14, 6.4);
  person.userData.personMaterial = personMaterial;
  scene.add(person);
  return person;
}

function createTree(x, z, scale = 1) {
  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(0.14 * scale, 0.2 * scale, 1.25 * scale, 8),
    new THREE.MeshStandardMaterial({ color: 0x805637, roughness: 1 }),
  );
  trunk.position.set(x, 0.62 * scale, z);
  trunk.castShadow = true;
  scene.add(trunk);

  const leaves = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.78 * scale, 1),
    new THREE.MeshStandardMaterial({ color: 0x43875a, roughness: 1 }),
  );
  leaves.position.set(x, 1.55 * scale, z);
  leaves.castShadow = true;
  scene.add(leaves);
}

function createBuilding(x, z, width, depth, height, color) {
  const building = new THREE.Group();
  const wall = new THREE.MeshStandardMaterial({ color, roughness: 0.88 });
  const roof = new THREE.MeshStandardMaterial({ color: 0x6c7884, roughness: 0.9 });
  addBox(building, [0, height / 2, 0], [width, height, depth], wall);
  addBox(building, [0, height + 0.12, 0], [width + 0.25, 0.24, depth + 0.25], roof);

  const windowMaterial = new THREE.MeshStandardMaterial({
    color: 0x8fc9de,
    emissive: 0x203c48,
    roughness: 0.35,
  });
  const floors = Math.max(1, Math.floor(height / 1.5));
  for (let floor = 0; floor < floors; floor += 1) {
    for (const side of [-1, 1]) {
      addBox(
        building,
        [side * width * 0.22, 0.9 + floor * 1.5, depth / 2 + 0.012],
        [0.55, 0.68, 0.03],
        windowMaterial,
        false,
      );
    }
  }

  building.position.set(x, 0, z);
  scene.add(building);
}

function createUrbanRoadGrid(asphalt, curb, white) {
  for (const z of [-20, -12, 18, 26]) {
    addBox(scene, [0, 0.015, z], [72, 0.12, 3.5], asphalt, false);
    addBox(scene, [0, 0.045, z - 1.9], [72, 0.12, 0.3], curb, false);
    addBox(scene, [0, 0.045, z + 1.9], [72, 0.12, 0.3], curb, false);
  }

  for (const x of CITY_STREET_X) {
    addBox(scene, [x, 0.015, 0], [3.5, 0.12, 58], asphalt, false);
    addBox(scene, [x - 1.9, 0.045, 0], [0.3, 0.12, 58], curb, false);
    addBox(scene, [x + 1.9, 0.045, 0], [0.3, 0.12, 58], curb, false);
  }

  for (const z of [-12, 18]) {
    for (const x of [-20, -10, 10, 20]) {
      createTrafficSignal(x, z, "x", x, z, (x + z + 100) / 4);
    }
  }

  for (let x = -34; x <= 34; x += 4) {
    for (const z of [-20, -12, 18, 26]) {
      addBox(scene, [x, 0.09, z], [1.9, 0.025, 0.12], white, false);
    }
  }

  for (const x of CITY_STREET_X) {
    for (let z = -27; z <= 27; z += 4) {
      addBox(scene, [x, 0.09, z], [0.12, 0.025, 1.9], white, false);
    }
  }

  const blockCentersX = [-35, -25, -15, -5, 5, 15, 25, 35];
  const blockCentersZ = [-25, -16, 22];
  const buildingColors = [0xc2c9cf, 0xd1c4b1, 0xd5d8d2, 0xc8d4da, 0xe1cfaa];
  blockCentersX.forEach((x, column) => {
    blockCentersZ.forEach((z, row) => {
      const height = 4 + ((column * 3 + row * 2) % 6) * 1.35;
      createBuilding(
        x,
        z,
        4.8 + (column % 2) * 0.7,
        4.2,
        height,
        buildingColors[(column + row * 2) % buildingColors.length],
      );
      if ((column + row) % 3 === 0) {
        createTree(x + 3.2, z + 2.7, 0.75);
      }
    });
  });
}

function createStreet() {
  const grass = new THREE.MeshStandardMaterial({ color: 0x93bb84, roughness: 1 });
  const asphalt = new THREE.MeshStandardMaterial({ color: 0x424b55, roughness: 0.96 });
  const curb = new THREE.MeshStandardMaterial({ color: 0xc8c9c1, roughness: 1 });
  const white = new THREE.MeshStandardMaterial({ color: 0xf4f0d5, roughness: 0.9 });
  const plotMats = [
    new THREE.MeshStandardMaterial({ color: 0xb5d19d, roughness: 1 }),
    new THREE.MeshStandardMaterial({ color: 0xa9cda9, roughness: 1 }),
    new THREE.MeshStandardMaterial({ color: 0xd5c49a, roughness: 1 }),
  ];

  addBox(scene, [0, -0.3, 0], [78, 0.5, 62], grass, false);
  createUrbanRoadGrid(asphalt, curb, white);
  addBox(scene, [0, -0.03, 0], [72, 0.12, 5], asphalt, false);
  addBox(scene, [0, 0.04, 2.7], [72, 0.16, 0.35], curb, false);
  addBox(scene, [0, 0.04, -2.7], [72, 0.16, 0.35], curb, false);

  for (let x = -35; x <= 35; x += 4) {
    addBox(scene, [x, 0.04, 0], [2.1, 0.025, 0.13], white, false);
  }

  for (let zone = 0; zone < ZONE_X.length; zone += 1) {
    const centerX = ZONE_X[zone];
    addBox(scene, [centerX, 0.015, 7.5], [9.4, 0.12, 10.4], plotMats[zone], false);
    addBox(scene, [centerX, 0.09, 4.1], [9.4, 0.03, 5.4], curb, false);
    addBox(scene, [centerX, 0.12, 4.1], [9.1, 0.035, 5.0], asphalt, false);
    for (const roadX of [centerX - 3.9, centerX + 3.9]) {
      addBox(scene, [roadX, 0.09, 0.8], [1.3, 0.03, 6.4], curb, false);
      addBox(scene, [roadX, 0.12, 0.8], [1.05, 0.035, 6.1], asphalt, false);
      createTrafficSignal(
        roadX + 0.85,
        0.2,
        "z",
        0,
        roadX,
        zone * 2 + (roadX > centerX ? 1 : 0),
      );
    }
    createTrafficSignal(centerX, 3.15, "x", centerX, 0, zone * 2 + 0.5);

    for (const xOffset of [-3.5, 3.5]) {
      createBuilding(
        centerX + xOffset,
        9.7,
        2.1,
        2.8,
        2.8 + ((zone + (xOffset > 0 ? 1 : 0)) % 3) * 0.8,
        [0xd5d8d2, 0xe1cfaa, 0xc8d4da][(zone + (xOffset > 0 ? 1 : 0)) % 3],
      );
    }

    const label = createTextSprite(`ZONA ${zone + 1}`);
    label.position.set(centerX, 3.1, 11.8);
    zoneLabels.push(label);
    scene.add(label);

    createTree(centerX - 4.3, 12.8, 0.8);
    createTree(centerX + 4.3, 12.8, 0.8);
    createTree(centerX - 4.5, -4.5, 0.75);
  }

  for (let x = -23; x <= 23; x += 2.5) {
    for (const z of [-2.25, 2.25]) {
      addBox(scene, [x, 0.045, z], [0.13, 0.025, 0.7], white, false);
    }
  }

  for (const z of [-13, -8, -4]) {
    createBuilding(-20 + ((Math.abs(z) % 3) * 2), z, 4.3, 3.2, 3 + Math.abs(z) * 0.22, 0xc2c9cf);
    createBuilding(19 - ((Math.abs(z) % 3) * 2), z, 4.1, 3.1, 4 + Math.abs(z) * 0.18, 0xd1c4b1);
  }
}

function createAmbientTraffic() {
  ambientCars = TRAFFIC_CORRIDORS.flatMap((corridor, corridorIndex) => {
    const eastLane = corridor - 1.2;
    const westLane = corridor + 1.2;
    const route = [
      new THREE.Vector3(-34, 0.08, eastLane),
      new THREE.Vector3(34, 0.08, eastLane),
      new THREE.Vector3(34, 0.08, westLane),
      new THREE.Vector3(-34, 0.08, westLane),
    ];
    route.push(route[0].clone());
    const routeLength = route.slice(1).reduce(
      (sum, point, segment) => sum + point.distanceTo(route[segment]),
      0,
    );

    return Array.from({ length: 8 }, (_, index) => {
      const car = createCar(cars.length + corridorIndex * 8 + index + 1, index + corridorIndex);
      car.scale.setScalar(0.82);
      car.userData.ambient = true;
      car.userData.cruiseRoute = route;
      car.userData.cruiseRouteLength = routeLength;
      car.userData.routeCrossings = routeCrossings(route);
      car.userData.routeDistance = (index * routeLength) / 8;
      car.userData.cruiseSpeed = 3.5 + ((index + corridorIndex) % 3) * 0.35;
      car.userData.lastUpdate = performance.now();
      moveAlongRoute(car, route, car.userData.routeDistance);
      return car;
    });
  });
}

function createPassengerMarkers() {
  passengerMarkers = ZONE_X.map((_, zone) =>
    Array.from({ length: 5 }, (_, index) => {
      const passenger = createPassenger(zone, index);
      passenger.visible = false;
      return passenger;
    }),
  );
}

function initializeScene() {
  cameraTarget = new THREE.Vector3(0, 0, 0);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xbcd7e8);
  scene.fog = new THREE.Fog(0xbcd7e8, 52, 105);

  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 160);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  sceneShell.prepend(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0xeaf5ff, 0x62714f, 2.1));
  sunlight = new THREE.DirectionalLight(0xfff2d8, 3.2);
  sunlight.position.set(-14, 26, 12);
  sunlight.castShadow = true;
  sunlight.shadow.mapSize.set(2048, 2048);
  sunlight.shadow.camera.left = -32;
  sunlight.shadow.camera.right = 32;
  sunlight.shadow.camera.top = 32;
  sunlight.shadow.camera.bottom = -32;
  scene.add(sunlight);

  createStreet();
  cars = episode.initial_cars.flatMap((count, zone) =>
    Array.from({ length: count }, (_, index) => createCar(
      episode.initial_cars.slice(0, zone).reduce((sum, value) => sum + value, 0) + index + 1,
    )),
  );
  placeCars(episode.initial_cars);
  createPassengerMarkers();
  createAmbientTraffic();
  renderStep(0);
  resizeRenderer();
  updateCamera();

  window.addEventListener("resize", resizeRenderer);
  sceneShell.addEventListener("pointerdown", onPointerDown);
  sceneShell.addEventListener("pointermove", onPointerMove);
  sceneShell.addEventListener("pointerup", onPointerUp);
  sceneShell.addEventListener("pointercancel", onPointerUp);
  sceneShell.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  renderer.setAnimationLoop(animateScene);
}

function resizeRenderer() {
  if (!renderer || !camera) return;
  const width = Math.max(1, sceneShell.clientWidth);
  const height = Math.max(1, sceneShell.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function updateCamera() {
  const sinPhi = Math.sin(cameraPhi);
  camera.position.set(
    cameraTarget.x + cameraRadius * sinPhi * Math.sin(cameraTheta),
    cameraTarget.y + cameraRadius * Math.cos(cameraPhi),
    cameraTarget.z + cameraRadius * sinPhi * Math.cos(cameraTheta),
  );
  camera.lookAt(cameraTarget);
}

function onPointerDown(event) {
  dragStart = { x: event.clientX, y: event.clientY, theta: cameraTheta, phi: cameraPhi };
  sceneShell.setPointerCapture(event.pointerId);
}

function onPointerMove(event) {
  if (!dragStart) return;
  cameraTheta = dragStart.theta - (event.clientX - dragStart.x) * 0.008;
  cameraPhi = THREE.MathUtils.clamp(
    dragStart.phi + (event.clientY - dragStart.y) * 0.006,
    0.3,
    1.42,
  );
  updateCamera();
}

function onPointerUp() {
  dragStart = undefined;
}

function onWheel(event) {
  event.preventDefault();
  cameraRadius = THREE.MathUtils.clamp(cameraRadius * (1 + event.deltaY * 0.001), 24, 100);
  updateCamera();
}

function onKeyDown(event) {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement) return;
  const move = 1.8;
  if (event.key === "ArrowLeft") cameraTheta -= 0.08;
  else if (event.key === "ArrowRight") cameraTheta += 0.08;
  else if (event.key === "ArrowUp") cameraPhi = Math.max(0.3, cameraPhi - 0.06);
  else if (event.key === "ArrowDown") cameraPhi = Math.min(1.42, cameraPhi + 0.06);
  else if (event.key.toLowerCase() === "w") cameraTarget.z -= move;
  else if (event.key.toLowerCase() === "s") cameraTarget.z += move;
  else if (event.key.toLowerCase() === "a") cameraTarget.x -= move;
  else if (event.key.toLowerCase() === "d") cameraTarget.x += move;
  else if (event.key === "+" || event.key === "=") cameraRadius = Math.max(24, cameraRadius - 2);
  else if (event.key === "-" || event.key === "_") cameraRadius = Math.min(100, cameraRadius + 2);
  else return;
  event.preventDefault();
  updateCamera();
}

function parkingPosition(zone, slot) {
  const [offsetX, z] = SLOT_OFFSETS[slot];
  return new THREE.Vector3(ZONE_X[zone] + offsetX, 0.08, z);
}

function placeCars(counts) {
  cars.forEach((car) => { car.userData.motion = null; });
  let carIndex = 0;
  counts.forEach((count, zone) => {
    for (let slot = 0; slot < count; slot += 1) {
      const car = cars[carIndex];
      car.userData.zone = zone;
      car.userData.slot = slot;
      car.position.copy(parkingPosition(zone, slot));
      car.rotation.y = 0;
      setCruiseRoute(car, zone, slot, performance.now() + car.userData.id * 250);
      carIndex += 1;
    }
  });
}

function getZoneRoute(zone, slot) {
  const centerX = ZONE_X[zone];
  const parking = parkingPosition(zone, slot);
  const route = [
    parking,
    new THREE.Vector3(centerX + 3.9, 0.08, 4.1),
    new THREE.Vector3(centerX + 3.9, 0.08, -2.1),
    new THREE.Vector3(centerX - 3.9, 0.08, -2.1),
    new THREE.Vector3(centerX - 3.9, 0.08, 4.1),
    parking,
  ];
  route.push(route[0].clone());
  return route;
}

function routeCrossings(route) {
  const crossings = [];
  let distance = 0;
  for (let segment = 0; segment < route.length - 1; segment += 1) {
    const from = route[segment];
    const to = route[segment + 1];
    const length = from.distanceTo(to);
    for (const signal of trafficSignals) {
      const fromCoordinate = signal.axis === "x" ? from.x : from.z;
      const toCoordinate = signal.axis === "x" ? to.x : to.z;
      const fromLane = signal.axis === "x" ? from.z : from.x;
      const toLane = signal.axis === "x" ? to.z : to.x;
      const crosses = (fromCoordinate < signal.coordinate && toCoordinate >= signal.coordinate)
        || (fromCoordinate > signal.coordinate && toCoordinate <= signal.coordinate);
      if (!crosses || fromCoordinate === toCoordinate) continue;
      const progress = (signal.coordinate - fromCoordinate) / (toCoordinate - fromCoordinate);
      const laneAtCrossing = fromLane + (toLane - fromLane) * progress;
      if (Math.abs(laneAtCrossing - signal.laneCoordinate) > 2.5) continue;
      crossings.push({
        distance: distance + length * progress,
        signal,
      });
    }
    distance += length;
  }
  crossings.sort((left, right) => left.distance - right.distance);
  return crossings;
}

function setCruiseRoute(car, zone, slot, startedAt) {
  const route = getZoneRoute(zone, slot);
  car.userData.zone = zone;
  car.userData.slot = slot;
  car.userData.cruiseRoute = route;
  car.userData.cruiseRouteLength = route.slice(1).reduce(
    (sum, point, index) => sum + point.distanceTo(route[index]),
    0,
  );
  car.userData.routeCrossings = routeCrossings(route);
  car.userData.routeDistance = 0;
  car.userData.cruiseStartedAt = startedAt;
  car.userData.lastUpdate = performance.now();
}

function signalState(signal, now) {
  const phase = ((now / 1000 + signal.phaseOffset) % SIGNAL_CYCLE_SECONDS);
  if (phase < 5) return "green";
  if (phase < 7) return "yellow";
  return "red";
}

function updateTrafficSignals(now) {
  let redSignals = 0;
  for (const signal of trafficSignals) {
    const state = signalState(signal, now);
    const activeIndex = state === "red" ? 0 : state === "yellow" ? 1 : 2;
    if (state !== "green") redSignals += 1;
    signal.bulbs.forEach((material, index) => {
      const color = index === 0
        ? (index === activeIndex ? 0xf04438 : 0x661e1c)
        : index === 1
          ? (index === activeIndex ? 0xffc247 : 0x66531a)
          : (index === activeIndex ? 0x43d17b : 0x1d5b30);
      material.color.setHex(color);
      material.emissive.setHex(index === activeIndex ? color : 0x100000);
      material.emissiveIntensity = index === activeIndex ? 1.3 : 0.12;
    });
  }
  return redSignals;
}

function createCarMotion(car, source, destination, destinationSlot) {
  const start = car.position.clone();
  const points = [
    start,
    new THREE.Vector3(ZONE_X[source], 0.08, -2.1),
    new THREE.Vector3(ZONE_X[destination], 0.08, -2.1),
    parkingPosition(destination, destinationSlot),
  ];
  const lengths = points.slice(1).map((point, index) => point.distanceTo(points[index]));
  const totalLength = lengths.reduce((sum, length) => sum + length, 0);
  car.userData.motion = {
    points,
    lengths,
    totalLength,
    crossings: routeCrossings(points),
    routeDistance: 0,
    destinationSlot,
  };
  car.userData.lastUpdate = performance.now();
  car.userData.zone = destination;
}

function moveAlongRoute(car, route, distance) {
  let segment = 0;
  let remaining = distance;
  while (segment < route.length - 1) {
    const length = route[segment].distanceTo(route[segment + 1]);
    if (remaining <= length) {
      const progress = length === 0 ? 1 : remaining / length;
      car.position.lerpVectors(route[segment], route[segment + 1], progress);
      const direction = route[segment + 1].clone().sub(route[segment]);
      car.rotation.y = -Math.atan2(direction.z, direction.x);
      return;
    }
    remaining -= length;
    segment += 1;
  }

  car.position.copy(route[route.length - 1]);
}

function congestionFactor(hour) {
  if ((hour >= 7 && hour <= 9) || (hour >= 16 && hour <= 18)) return 0.58;
  if (hour >= 11 && hour <= 14) return 0.78;
  return 1;
}

function advanceOnRoute(car, route, routeLength, crossings, speed, now, distance, loop) {
  const delta = Math.min(Math.max((now - car.userData.lastUpdate) / 1000, 0), 0.1);
  const hour = START_HOUR + currentStep;
  let nextDistance = distance + speed * congestionFactor(hour) * delta;

  for (const crossing of crossings) {
    let crossingDistance = crossing.distance;
    if (loop && crossingDistance <= distance + 0.02) {
      crossingDistance += routeLength;
    }
    const distanceToCrossing = crossingDistance - distance;
    if (distanceToCrossing > speed * delta + 1.1 || distanceToCrossing < 0) continue;
    const state = signalState(crossing.signal, now);
    if (state === "green") continue;

    const stopLine = crossingDistance - 0.9;
    if (distance < stopLine) {
      nextDistance = Math.min(nextDistance, stopLine);
    } else {
      nextDistance = distance;
    }
    break;
  }

  if (loop) nextDistance %= routeLength;
  else nextDistance = Math.min(nextDistance, routeLength);
  car.userData.lastUpdate = now;
  moveAlongRoute(car, route, nextDistance);
  return nextDistance;
}

function advanceCruisingCar(car, now) {
  const speed = car.userData.ambient ? car.userData.cruiseSpeed : CRUISE_SPEED;
  car.userData.routeDistance = advanceOnRoute(
    car,
    car.userData.cruiseRoute,
    car.userData.cruiseRouteLength,
    car.userData.routeCrossings,
    speed,
    now,
    car.userData.routeDistance,
    true,
  );
}

function updateCarMotion(now) {
  for (const car of [...cars, ...ambientCars]) {
    const motion = car.userData.motion;
    if (motion) {
      motion.routeDistance = advanceOnRoute(
        car,
        motion.points,
        motion.totalLength,
        motion.crossings,
        5.2,
        now,
        motion.routeDistance,
        false,
      );
      if (motion.routeDistance >= motion.totalLength) {
        car.position.copy(motion.points[motion.points.length - 1]);
        setCruiseRoute(
          car,
          car.userData.zone,
          motion.destinationSlot,
          now,
        );
        car.userData.motion = null;
      }
      continue;
    }

    advanceCruisingCar(car, now);
  }
}

function countsAt(step) {
  return step === 0 ? episode.initial_cars : episode.history.cars[step - 1];
}

function dayPeriod(hour) {
  if (hour < 10) return "Hora pico AM";
  if (hour < 16) return "Día";
  if (hour < 19) return "Hora pico PM";
  return "Atardecer";
}

function updateDayLighting(hour) {
  if (!sunlight) return;
  const progress = THREE.MathUtils.clamp((hour - START_HOUR) / 12, 0, 1);
  const morning = new THREE.Color(0xbcd7e8);
  const evening = new THREE.Color(0xe7a278);
  scene.background.copy(morning).lerp(evening, progress);
  scene.fog.color.copy(scene.background);
  sunlight.color.set(0xfff1d8).lerp(new THREE.Color(0xffa56e), progress);
  sunlight.intensity = 3.2 - progress * 1.65;
}

function renderStep(step) {
  currentStep = step;
  const frame = Math.max(0, step - 1);
  const counts = countsAt(step);
  const demands = episode.history.demand[frame];

  counts.forEach((count, zone) => {
    const served = Math.min(count, demands[zone]);
    const waiting = Math.max(0, demands[zone] - served);
    updateZoneLabel(zone, demands[zone], count, served, waiting);
    passengerMarkers[zone].forEach((passenger, index) => {
      passenger.visible = index < demands[zone];
      passenger.userData.personMaterial.color.setHex(index < served ? 0x39b86c : 0xf08a45);
      passenger.userData.personMaterial.emissive.setHex(index < served ? 0x073519 : 0x351300);
    });
  });
  const hour = START_HOUR + step;
  timeValue.textContent = `${String(hour).padStart(2, "0")}:00 · ${dayPeriod(hour)}`;
  updateDayLighting(hour);
  stepOutput.value = `${step} / ${episode.history.cars.length}`;
  slider.value = String(step);
  agentValue.textContent = `${episode.agent} · semilla ${episode.seed}`;

  if (step === 0) {
    actionValue.textContent = "Listo para iniciar";
    rewardValue.textContent = "-";
    totalValue.textContent = "0.00";
  } else {
    const action = episode.history.actions[frame];
    const move = episode.action_moves[action];
    actionValue.textContent = move
      ? `Zona ${move[0] + 1} → Zona ${move[1] + 1}`
      : "Sin movimiento";
    rewardValue.textContent = Number(episode.history.rewards[frame]).toFixed(2);
    totalValue.textContent = episode.history.rewards
      .slice(0, step)
      .reduce((sum, value) => sum + value, 0)
      .toFixed(2);
  }

  playButton.textContent = timer
    ? "⏸ Pausar"
    : step === episode.history.cars.length
      ? "▶ Reproducir de nuevo"
      : "▶ Reproducir";
}

function advance() {
  if (currentStep >= episode.history.cars.length) {
    pause();
    return;
  }

  const frame = currentStep;
  const action = episode.history.actions[frame];
  const move = episode.action_moves[action];
  const nextCounts = episode.history.cars[frame];

  if (move) {
    const [source, destination] = move;
    const counts = episode.initial_cars.map((_, zone) =>
      cars.filter((car) => car.userData.zone === zone).length,
    );
    const expectedCounts = counts.slice();
    expectedCounts[source] -= 1;
    expectedCounts[destination] += 1;

    if (expectedCounts.every((count, zone) => count === nextCounts[zone])) {
      const car = cars.find((candidate) => candidate.userData.zone === source);
      const destinationSlot = cars
        .filter((candidate) => candidate.userData.zone === destination)
        .length;
      if (car && destinationSlot < SLOT_OFFSETS.length) {
        createCarMotion(car, source, destination, destinationSlot);
      } else {
        placeCars(nextCounts);
      }
    } else {
      placeCars(nextCounts);
    }
  } else {
    placeCars(nextCounts);
  }

  currentStep += 1;
  renderStep(currentStep);
  setMessage(`Reproduciendo episodio · paso ${currentStep} de ${episode.history.cars.length}`);
  if (currentStep >= episode.history.cars.length) pause();
}

function pause() {
  window.clearInterval(timer);
  timer = undefined;
  if (episode) {
    playButton.textContent = currentStep >= episode.history.cars.length
      ? "▶ Reproducir de nuevo"
      : "▶ Reproducir";
  }
}

function reset() {
  pause();
  placeCars(episode.initial_cars);
  renderStep(0);
  setMessage("Episodio listo.");
}

function validateEpisode(data) {
  const history = data?.history;
  const length = history?.cars?.length;
  return Array.isArray(data?.initial_cars)
    && data.initial_cars.length === ZONE_X.length
    && Array.isArray(data?.action_moves)
    && Number.isInteger(length)
    && length > 0
    && history.demand?.length === length
    && history.actions?.length === length
    && history.rewards?.length === length
    && history.cars.every((row) => row.length === data.initial_cars.length)
    && history.demand.every((row) => row.length === data.initial_cars.length)
    && history.actions.every((action) => Number.isInteger(action) && data.action_moves[action] !== undefined);
}

function animateScene(now) {
  updateCarMotion(now);
  const redSignals = updateTrafficSignals(now);
  signalValue.textContent = `${redSignals} en rojo/precaución`;
  renderer.render(scene, camera);
}

playButton.addEventListener("click", () => {
  if (currentStep >= episode.history.cars.length) reset();
  if (timer) {
    pause();
    setMessage("Reproducción pausada.");
  } else {
    setMessage("Reproduciendo episodio.");
    timer = window.setInterval(advance, 1800);
    playButton.textContent = "⏸ Pausar";
  }
});

resetButton.addEventListener("click", reset);
slider.addEventListener("input", () => {
  pause();
  placeCars(countsAt(Number(slider.value)));
  renderStep(Number(slider.value));
  setMessage("Episodio listo.");
});

try {
  THREE = await import("https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js");
  const response = await fetch("./simulation_data.json");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  episode = await response.json();
  if (!validateEpisode(episode)) {
    throw new Error("El archivo no tiene el formato esperado.");
  }
  initializeScene();
  slider.max = String(episode.history.cars.length);
  slider.disabled = false;
  playButton.disabled = false;
  resetButton.disabled = false;
  renderStep(0);
  setMessage("Escena WebGL lista. Reproduce el episodio o explora la ciudad.");
} catch (error) {
  agentValue.textContent = "No disponible";
  setMessage(
    `No se pudo cargar la visualización (${error.message}). Revisa la conexión a internet para descargar Three.js y ejecuta la página desde un servidor local.`,
    true,
  );
}
