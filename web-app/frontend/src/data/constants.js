export const GENERALIST_CAT = {
  id: 'generalist',
  title: 'Generalist',
  label: 'All categories',
  desc: 'Multi-class detection across all object types.',
  detections: [
    { id: 'TRK-001', cls: 'VEHICLE',   conf: 0.94, x: 0.12, y: 0.28, w: 0.14, h: 0.18 },
    { id: 'TRK-002', cls: 'PERSONNEL', conf: 0.87, x: 0.52, y: 0.44, w: 0.06, h: 0.12 },
    { id: 'TRK-003', cls: 'VEHICLE',   conf: 0.91, x: 0.68, y: 0.22, w: 0.16, h: 0.20 },
    { id: 'TRK-004', cls: 'PERSONNEL', conf: 0.79, x: 0.35, y: 0.55, w: 0.05, h: 0.13 },
    { id: 'TRK-005', cls: 'AIRCRAFT',  conf: 0.83, x: 0.74, y: 0.10, w: 0.10, h: 0.08 },
  ],
  src: 'UA-GEN-2024-04811',
  location: 'Donetsk Oblast',
  coords: '48.1° N, 37.8° E',
}

export const ML_CATEGORIES = [
  {
    id: 'aircraft',
    title: 'Aircraft',
    label: 'Drone, Missle and Aircraft detection',
    modelInfo: 'YOLOv8m · 32K imgs · mAP50 0.92',
    desc: 'nc=1 specialist for fixed-wing, rotary-wing, UAV/drone, and missile detection. Trained on ~32K images from 3 Kaggle datasets.',
    videoSrc: '/aircraft.mp4',
    detections: [
      { id: 'AC-001', cls: 'UAV',        conf: 0.97, x: 0.55, y: 0.18, w: 0.08, h: 0.07 },
      { id: 'AC-002', cls: 'HELICOPTER', conf: 0.88, x: 0.22, y: 0.32, w: 0.12, h: 0.10 },
      { id: 'AC-003', cls: 'UAV',        conf: 0.93, x: 0.73, y: 0.42, w: 0.07, h: 0.06 },
      { id: 'AC-004', cls: 'FIXED-WING', conf: 0.76, x: 0.40, y: 0.12, w: 0.14, h: 0.09 },
    ],
    src: 'UA-AIR-2024-02219',
    location: 'Zaporizhzhia Region',
    coords: '47.8° N, 35.1° E',
  },
  {
    id: 'personnel',
    title: 'Personnel',
    label: 'Human Combatant detection',
    modelInfo: 'YOLOv8m · 26K imgs · GDINO labels',
    desc: 'nc=1 specialist for soldiers, fighters, and RPG/ATGM operators. Trained on kiit-mita and piterfm datasets with GDINO-assisted labeling.',
    videoSrc: '/personnel.mp4',
    detections: [
      { id: 'P-001', cls: 'SOLDIER',  conf: 0.96, x: 0.20, y: 0.38, w: 0.05, h: 0.14 },
      { id: 'P-002', cls: 'SOLDIER',  conf: 0.92, x: 0.28, y: 0.40, w: 0.05, h: 0.14 },
      { id: 'P-003', cls: 'SOLDIER',  conf: 0.88, x: 0.36, y: 0.36, w: 0.05, h: 0.15 },
      { id: 'P-004', cls: 'SOLDIER',  conf: 0.74, x: 0.62, y: 0.50, w: 0.04, h: 0.12 },
      { id: 'P-005', cls: 'SOLDIER',  conf: 0.90, x: 0.70, y: 0.35, w: 0.05, h: 0.14 },
      { id: 'P-006', cls: 'SOLDIER',  conf: 0.85, x: 0.78, y: 0.42, w: 0.05, h: 0.13 },
    ],
    src: 'UA-PER-2024-07733',
    location: 'Kharkiv Oblast',
    coords: '49.9° N, 36.2° E',
  },
  {
    id: 'vehicles',
    title: 'Vehicles',
    label: 'Armor & transport',
    modelInfo: 'YOLOv8m · 45K imgs · 4 datasets',
    desc: 'nc=1 specialist for tanks, APCs, IFVs, self-propelled artillery, and logistics. Trained on ~45K images across 4 Kaggle datasets including 26K GDINO-labeled piterfm images.',
    videoSrc: '/vehicle.mp4',
    detections: [
      { id: 'V-001', cls: 'MBT T-72', conf: 0.95, x: 0.10, y: 0.30, w: 0.18, h: 0.22 },
      { id: 'V-002', cls: 'APC BMP',  conf: 0.91, x: 0.38, y: 0.35, w: 0.15, h: 0.18 },
      { id: 'V-003', cls: 'MBT T-72', conf: 0.89, x: 0.65, y: 0.25, w: 0.18, h: 0.22 },
      { id: 'V-004', cls: 'SPG',      conf: 0.82, x: 0.22, y: 0.58, w: 0.13, h: 0.16 },
      { id: 'V-005', cls: 'TRUCK',    conf: 0.77, x: 0.55, y: 0.60, w: 0.12, h: 0.14 },
    ],
    src: 'UA-VEH-2024-05542',
    location: 'Lyman, Donetsk',
    coords: '48.9° N, 37.8° E',
  },
]

export const CAT_COLORS = {
  generalist: 'oklch(0.65 0.18 55deg)',
  aircraft:   'oklch(0.62 0.16 220deg)',
  personnel:  'oklch(0.60 0.18 145deg)',
  vehicles:   'oklch(0.60 0.20 25deg)',
}

export const CAT_RGB = {
  generalist: '194, 120,  40',
  aircraft:   ' 56, 152, 210',
  personnel:  ' 60, 180,  90',
  vehicles:   '210,  80,  50',
}

export const FOOTAGE_DATA = [
  { id: 1, title: 'Close Encounters of the FPV Kind',            date: '2026-05-02', duration: '00:00:51', detClass: 'AIRCRAFT', source: 'Funker530',    tag: 'annotated', src: '50E23154', videoUrl: '/media/annotated/50e23154_funker_aircraft.mp4' },
  { id: 2, title: 'Russian FPV Near-Miss — Ukrainian Vehicle',   date: '2026-05-02', duration: '00:00:42', detClass: 'AIRCRAFT', source: 'Funker530',    tag: 'annotated', src: '5488898D', videoUrl: '/media/annotated/5488898d_funker_aircraft.mp4' },
  { id: 3, title: 'Mi-28 Hit by Ukrainian FPV Drone',            date: '2026-05-02', duration: '00:01:25', detClass: 'AIRCRAFT', source: 'GeoConfirmed', tag: 'annotated', src: '5B0D33CB', videoUrl: '/media/annotated/5b0d33cb_aircraft_annotated.mp4' },
  { id: 4, title: 'Russian Shahed/Geran Drone Hits Building',    date: '2026-05-02', duration: '00:00:12', detClass: 'AIRCRAFT', source: 'GeoConfirmed', tag: 'annotated', src: 'DB04C53B', videoUrl: '/media/annotated/db04c53b_aircraft_annotated.mp4' },
]

export const DET_CLASSES = ['All', 'Aircraft', 'Vehicle', 'Personnel']
export const SOURCES     = ['All sources', 'Funker530', 'GeoConfirmed']

export const TICKER_ITEMS = [
  'ARCHIVE ONLINE — ACTIVE COLLECTION',
  'FUNKER530 — MONITORING ACTIVE',
  'GEOCONFIRMED — MONITORING ACTIVE',
  'AIRCRAFT MODEL — mAP50 0.911',
  'VEHICLE MODEL — TRAINING',
  'PERSONNEL MODEL — QUEUED',
  'GDINO LABELING — 26K IMAGES COMPLETE',
  'LAST SCRAPE: AUTO',
  'COLLECTION PERIOD: FEB 2022 — PRESENT',
  '3 DETECTION CLASSES: AIRCRAFT · VEHICLE · PERSONNEL',
]
