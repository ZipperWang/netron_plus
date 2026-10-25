// JSON shapes returned by the Rust core (see crates/core/src/{summary,api,view}.rs).

export type Dim = number | string | null;
export type Shape = Dim[] | null;
export type ShapeSource = 'unknown' | 'declared' | 'inferred' | 'tensor';

export interface KeyValue {
  key: string;
  value: string;
}

export interface Opset {
  domain: string;
  version: number;
}

export interface GraphInfo {
  id: number;
  name: string;
  label: string;
  kind: 'main' | 'subgraph' | 'function' | 'training';
  parent: number | null;
  parentNode: number | null;
  attribute: string | null;
  nodes: number;
  depth: number;
}

export interface IoInfo {
  value: number;
  name: string;
  dtype: string;
  shape: Shape;
  declared: Shape;
  display: string;
  bytes: number | null;
  doc: string;
}

export interface Stats {
  nodes: number;
  nodesTotal: number;
  opTypes: number;
  initializers: number;
  values: number;
  graphs: number;
  subgraphs: number;
  functions: number;
  unknownShapes: number;
  intermediateValues: number;
  customOps: number;
  hiddenConstants: number;
}

export interface DtypeRow {
  dtype: string;
  tensors: number;
  elements: number;
  bytes: number;
}

export interface OpRow {
  op: string;
  domain: string;
  category: string;
  count: number;
  params: number;
  flops: number;
  macs: number;
  costed: number;
}

export interface CategoryRow {
  category: string;
  count: number;
  flops: number;
}

export interface TensorRow {
  graph: number;
  value: number;
  name: string;
  dtype: string;
  shape: string;
  elements: number;
  bytes: number;
}

export interface SymbolInfo {
  name: string;
  value: number | null;
  inputs: string[];
}

export interface MemoryStats {
  weightsBytes: number;
  activationBytes: number;
  peakBytes: number;
  peakNode: number | null;
  inputBytes: number;
  outputBytes: number;
  knownValues: number;
  unknownValues: number;
}

export interface Totals {
  params: number;
  paramBytes: number;
  macs: number;
  flops: number;
  costedNodes: number;
  uncostedNodes: number;
  uncostedOps: string[];
  assumed: string[];
}

export interface ModelSummary {
  id: number;
  fileName: string;
  filePath: string | null;
  fileSize: number;
  irVersion: number;
  producerName: string;
  producerVersion: string;
  domain: string;
  modelVersion: number;
  doc: string;
  opsets: Opset[];
  metadata: KeyValue[];
  graphs: GraphInfo[];
  inputs: IoInfo[];
  outputs: IoInfo[];
  stats: Stats;
  dtypes: DtypeRow[];
  ops: OpRow[];
  categories: CategoryRow[];
  largest: TensorRow[];
  symbols: SymbolInfo[];
  overrides: { params: Record<string, number>; inputs: Record<string, (number | null)[]> };
  memory: MemoryStats;
  totals: Totals;
  issues: { error: number; warning: number; info: number };
  externalData: boolean;
}

export interface ViewLine {
  k: string;
  v: string;
  value: number;
}

export type ViewNodeKind = 'op' | 'input' | 'output' | 'outer' | 'missing';

export interface ViewNode {
  kind: ViewNodeKind;
  node: number;
  value: number;
  title: string;
  name: string;
  cat: string;
  lines: ViewLine[];
  module: string;
  sub: boolean;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ViewEdge {
  from: number;
  to: number;
  value: number;
  name: string;
  label: string;
  points: number[];
  implicit: boolean;
  detached: boolean;
  back: boolean;
}

export interface GraphView {
  graph: number;
  nodes: ViewNode[];
  edges: ViewEdge[];
  width: number;
  height: number;
  hiddenConstants: number;
  foldedNodes: number;
}

export interface NodeRef {
  graph: number;
  id: number;
  name: string;
  op: string;
  slot: number | null;
}

export interface ValueBrief {
  id: number;
  name: string;
  kind: string;
  dtype: string;
  shape: Shape;
  source: ShapeSource;
  producer: NodeRef | null;
  consumers: number;
  isInput: boolean;
  isOutput: boolean;
  tensor: number | null;
  preview: string | null;
}

export interface SlotInfo {
  slot: number;
  label: string;
  value: ValueBrief | null;
}

export interface AttrInfo {
  name: string;
  typeName: string;
  value: string;
  doc: string;
  refAttrName: string;
  graphs: number[];
  tensor: number | null;
}

export interface NodeCost {
  params: number;
  paramBytes: number;
  macs: number | null;
  flops: number | null;
  readBytes: number | null;
  writeBytes: number | null;
}

export interface NodeDetail {
  graph: number;
  id: number;
  order: number;
  name: string;
  op: string;
  domain: string;
  category: string;
  opset: number | null;
  doc: string;
  docUrl: string | null;
  inputs: SlotInfo[];
  outputs: SlotInfo[];
  implicitInputs: ValueBrief[];
  attributes: AttrInfo[];
  subgraphs: { id: number; label: string; nodes: number }[];
  cost: NodeCost;
  metadata: KeyValue[];
  module: string;
}

export interface ExternalInfo {
  location: string;
  offset: number;
  length: number | null;
}

export interface TensorMeta {
  index: number;
  dtype: string;
  dims: number[];
  elements: number;
  bytes: number | null;
  storage: 'raw' | 'typed' | 'external';
  external: ExternalInfo | null;
  doc: string;
  preview: string[];
  numeric: boolean;
}

export interface TypeInfo {
  kind: string;
  dtype: number;
  shape: Shape;
  denotation: string;
  display: string;
}

export interface ValueDetail {
  graph: number;
  id: number;
  name: string;
  kind: string;
  dtype: string;
  shape: Shape;
  declared: TypeInfo | null;
  source: ShapeSource;
  isInput: boolean;
  isOutput: boolean;
  doc: string;
  elements: number | null;
  bytes: number | null;
  producer: NodeRef | null;
  consumers: NodeRef[];
  tensor: TensorMeta | null;
  contents: Dim[] | null;
  outer: [number, number] | null;
  quantization: KeyValue[];
}

export interface Histogram {
  min: number;
  max: number;
  counts: number[];
  integerBins: boolean;
}

export interface TensorStats {
  dtype: string;
  dims: number[];
  count: number;
  finite: number;
  nan: number;
  posInf: number;
  negInf: number;
  zeros: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  absMean: number | null;
  absMax: number | null;
  l2Norm: number | null;
  fp16Overflow: number;
  fp16Underflow: number;
  subnormal: number;
  integer: boolean;
  histogram: Histogram | null;
  percentiles: [number, number][];
  sampled: boolean;
}

export interface ValuesPage {
  total: number;
  offset: number;
  values: string[];
}

export interface NodeRow {
  id: number;
  order: number;
  name: string;
  op: string;
  domain: string;
  cat: string;
  inputs: string;
  outputs: string;
  params: number;
  macs: number | null;
  flops: number | null;
  outBytes: number | null;
  readBytes: number | null;
  module: string;
}

export interface ValueRow {
  id: number;
  name: string;
  kind: string;
  dtype: string;
  shape: string;
  elements: number | null;
  bytes: number | null;
  producer: number | null;
  producerOp: string;
  consumers: number;
  isInput: boolean;
  isOutput: boolean;
  tensor: number | null;
  source: ShapeSource;
  external: boolean;
}

export interface WeightRow {
  graph: number;
  value: number;
  tensor: number;
  name: string;
  dtype: string;
  shape: string;
  elements: number;
  bytes: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  absMean: number | null;
  zeros: number;
  nan: number;
  inf: number;
  fp16Overflow: number;
  consumers: string;
  error: string | null;
}

export type Severity = 'error' | 'warning' | 'info';

export interface Issue {
  severity: Severity;
  code: string;
  message: string;
  graph: number;
  node: number | null;
  value: number | null;
  count: number;
  items: string[];
  params: Record<string, string>;
}

export interface ModuleNode {
  name: string;
  path: string;
  nodes: number;
  params: number;
  flops: number;
  children: ModuleNode[];
}

export interface SearchHit {
  kind: 'node' | 'value';
  graph: number;
  id: number;
  name: string;
  op: string;
  detail: string;
}
