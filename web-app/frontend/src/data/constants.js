// Canvas color palette — keyed by category id, used by useMLCanvas.js
export const CAT_RGB = {
  generalist: '194, 120,  40',
  aircraft:   ' 56, 152, 210',
  personnel:  ' 60, 180,  90',
  vehicles:   '210,  80,  50',
}

// Hero section demo card — purely visual, canvas animation only
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
}

// ML detection section cards — live stats and video injected at runtime from /api/stats and /api/annotated-clips
export const ML_CATEGORIES = [
  {
    id: 'aircraft',
    title: 'Aircraft',
    label: 'Drone, Missile and Aircraft detection',
    desc: 'nc=1 specialist for fixed-wing, rotary-wing, UAV/drone, and missile detection. Trained on ~115K images across 6 Kaggle datasets.',
    detections: [
      { id: 'AC-001', cls: 'UAV',        conf: 0.97, x: 0.55, y: 0.18, w: 0.08, h: 0.07 },
      { id: 'AC-002', cls: 'HELICOPTER', conf: 0.88, x: 0.22, y: 0.32, w: 0.12, h: 0.10 },
      { id: 'AC-003', cls: 'UAV',        conf: 0.93, x: 0.73, y: 0.42, w: 0.07, h: 0.06 },
      { id: 'AC-004', cls: 'FIXED-WING', conf: 0.76, x: 0.40, y: 0.12, w: 0.14, h: 0.09 },
    ],
  },
  {
    id: 'personnel',
    title: 'Personnel',
    label: 'Human Combatant detection',
    desc: 'nc=1 specialist for soldiers, fighters, and RPG/ATGM operators. Trained on ~25K images with GDINO-assisted labeling.',
    detections: [
      { id: 'P-001', cls: 'SOLDIER', conf: 0.96, x: 0.20, y: 0.38, w: 0.05, h: 0.14 },
      { id: 'P-002', cls: 'SOLDIER', conf: 0.92, x: 0.28, y: 0.40, w: 0.05, h: 0.14 },
      { id: 'P-003', cls: 'SOLDIER', conf: 0.88, x: 0.36, y: 0.36, w: 0.05, h: 0.15 },
      { id: 'P-004', cls: 'SOLDIER', conf: 0.74, x: 0.62, y: 0.50, w: 0.04, h: 0.12 },
      { id: 'P-005', cls: 'SOLDIER', conf: 0.90, x: 0.70, y: 0.35, w: 0.05, h: 0.14 },
      { id: 'P-006', cls: 'SOLDIER', conf: 0.85, x: 0.78, y: 0.42, w: 0.05, h: 0.13 },
    ],
  },
  {
    id: 'vehicles',
    title: 'Vehicles',
    label: 'Armor & transport',
    desc: 'nc=1 specialist for tanks, APCs, IFVs, self-propelled artillery, and logistics vehicles. Trained on ~87K images across 5 datasets.',
    detections: [
      { id: 'V-001', cls: 'MBT T-72', conf: 0.95, x: 0.10, y: 0.30, w: 0.11, h: 0.13 },
      { id: 'V-002', cls: 'APC BMP',  conf: 0.91, x: 0.38, y: 0.35, w: 0.09, h: 0.11 },
      { id: 'V-003', cls: 'MBT T-72', conf: 0.89, x: 0.65, y: 0.25, w: 0.11, h: 0.13 },
      { id: 'V-004', cls: 'SPG',      conf: 0.82, x: 0.22, y: 0.58, w: 0.08, h: 0.10 },
      { id: 'V-005', cls: 'TRUCK',    conf: 0.77, x: 0.55, y: 0.60, w: 0.08, h: 0.09 },
    ],
  },
]
