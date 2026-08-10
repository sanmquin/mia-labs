import { useState, useEffect, useRef } from 'react';
import {
  Play, Pause, RotateCcw, ChevronRight, Compass, BarChart2, BookOpen, AlertCircle, CheckCircle2, XCircle, Award, HelpCircle
} from 'lucide-react';

const GRID_SIZE = 10;

// ── TypeScript Types ──
interface MazeResult {
  name: string;
  grid: number[][];
  start: [number, number];
  end: [number, number];
  bfs_path: [number, number][];
  tess_path: [number, number][];
  tess_reached: boolean;
  tess_limit: boolean;
  ce_path: [number, number][];
  ce_reached: boolean;
  ce_limit: boolean;
}

interface RunInfo {
  version: string;
  name: string;
  epochs: number;
  train_mazes: number;
  test_mazes: number;
  success_rate: number;
  optimal_rate: number;
  efficiency: number;
  loss_type: string;
  notes: string;
}

interface TrainingHistory {
  runs: RunInfo[];
  stress_test: {
    tesseract_success: number;
    ce_baseline_success: number;
    seeds: number[];
    num_mazes: number;
  };
  per_cell_val_accuracy_v4: Record<string, number>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'adversarial' | 'testing' | 'progress'>('adversarial');

  // Data states
  const [adversarialMazes, setAdversarialMazes] = useState<MazeResult[]>([]);
  const [testMazes, setTestMazes] = useState<MazeResult[]>([]);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Selector states
  const [selectedAdvIdx, setSelectedAdvIdx] = useState(0);
  const [selectedTestIdx, setSelectedTestIdx] = useState(0);

  // Playback states (for active simulator)
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playSpeed, setPlaySpeed] = useState(300); // ms
  const [selectedModel, setSelectedModel] = useState<'tess' | 'ce' | 'bfs'>('tess');

  // Load static data from public/data
  useEffect(() => {
    Promise.all([
      fetch('/data/adversarial_results.json').then(res => res.json()),
      fetch('/data/test_dataset_results.json').then(res => res.json()),
      fetch('/data/training_history.json').then(res => res.json())
    ])
      .then(([advData, testData, histData]) => {
        setAdversarialMazes(advData);
        setTestMazes(testData);
        setHistory(histData);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load visualization data", err);
        setError("Could not load visualization data from public folder. Please check if export_viz_data.py has been run.");
        setLoading(false);
      });
  }, []);

  // Playback interval logic
  const timerRef = useRef<number | null>(null);

  const activeMaze: MazeResult | undefined =
    activeTab === 'adversarial' ? adversarialMazes[selectedAdvIdx] :
    activeTab === 'testing' ? testMazes[selectedTestIdx] : undefined;

  const activePath = activeMaze ? (
    selectedModel === 'tess' ? activeMaze.tess_path :
    selectedModel === 'ce' ? activeMaze.ce_path :
    activeMaze.bfs_path
  ) : [];

  const maxSteps = activePath.length;

  useEffect(() => {
    // Reset step whenever maze or model changes
    setCurrentStep(0);
    setIsPlaying(false);
  }, [selectedAdvIdx, selectedTestIdx, selectedModel, activeTab]);

  useEffect(() => {
    if (isPlaying) {
      timerRef.current = window.setInterval(() => {
        setCurrentStep(prev => {
          if (prev < maxSteps - 1) {
            return prev + 1;
          } else {
            setIsPlaying(false);
            return prev;
          }
        });
      }, playSpeed);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPlaying, maxSteps, playSpeed]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center justify-center p-6">
        <div className="w-16 h-16 border-4 border-teal-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p className="text-slate-400 font-medium">Assembling Tesseract Spatial Analyzer...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center justify-center p-6">
        <AlertCircle className="w-16 h-16 text-red-500 mb-4 animate-bounce" />
        <h1 className="text-2xl font-bold text-red-400 mb-2">Initialization Error</h1>
        <p className="text-slate-400 max-w-md text-center">{error}</p>
      </div>
    );
  }

  if ((activeTab === 'adversarial' || activeTab === 'testing') && !activeMaze) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center justify-center p-6">
        <AlertCircle className="w-16 h-16 text-red-500 mb-4 animate-bounce" />
        <h1 className="text-2xl font-bold text-red-400 mb-2">Initialization Error</h1>
        <p className="text-slate-400 max-w-md text-center">No active maze selected.</p>
      </div>
    );
  }

  // Helper to render grid cell with active path overlays
  const renderGridCell = (r: number, c: number) => {
    if (!activeMaze) return null;
    const cellVal = activeMaze.grid[r][c];
    const isStart = activeMaze.start[0] === r && activeMaze.start[1] === c;
    const isEnd = activeMaze.end[0] === r && activeMaze.end[1] === c;

    // Check if cell is in current active path slice
    const visiblePathSlice = activePath.slice(0, currentStep + 1);
    const pathIdx = visiblePathSlice.findIndex(p => p[0] === r && p[1] === c);
    const isCellOnPath = pathIdx !== -1;
    const isCurrentAgentPos = visiblePathSlice.length > 0 &&
      visiblePathSlice[visiblePathSlice.length - 1][0] === r &&
      visiblePathSlice[visiblePathSlice.length - 1][1] === c;

    // Determine background color
    let bgClass = "bg-slate-900 border-slate-800";
    if (cellVal >= 3 && cellVal <= 9) {
      bgClass = "bg-slate-800 border-slate-700 shadow-inner"; // Wall
    }

    return (
      <div
        key={`${r}-${c}`}
        className={`relative w-full aspect-square border flex items-center justify-center font-bold text-sm transition-all duration-200 ${bgClass}`}
      >
        {/* Wall overlay decoration */}
        {cellVal >= 3 && cellVal <= 9 && (
          <div className="absolute inset-1 bg-slate-850 rounded opacity-60 flex items-center justify-center text-[10px] text-slate-500 font-mono">
            {cellVal === 9 ? 'W9' : `W${cellVal}`}
          </div>
        )}

        {/* Path marker overlays */}
        {isCellOnPath && cellVal < 3 && (
          <div className={`absolute inset-[3px] rounded-full transition-all duration-300 ${
            selectedModel === 'tess' ? 'bg-teal-500/20 border border-teal-500/40' :
            selectedModel === 'ce' ? 'bg-rose-500/20 border border-rose-500/40' :
            'bg-violet-500/25 border border-violet-500/45'
          }`} />
        )}

        {/* Path numbering */}
        {isCellOnPath && cellVal < 3 && !isCurrentAgentPos && (
          <span className={`absolute bottom-0.5 right-1 text-[8px] font-mono leading-none ${
            selectedModel === 'tess' ? 'text-teal-400' :
            selectedModel === 'ce' ? 'text-rose-400' :
            'text-violet-400'
          }`}>
            {pathIdx}
          </span>
        )}

        {/* Current Agent Location */}
        {isCurrentAgentPos && cellVal < 3 && (
          <div className={`absolute inset-[6px] rounded-md flex items-center justify-center shadow-lg transform scale-110 animate-pulse transition-transform ${
            selectedModel === 'tess' ? 'bg-teal-400 text-slate-950' :
            selectedModel === 'ce' ? 'bg-rose-500 text-white' :
            'bg-violet-400 text-slate-950'
          }`}>
            <span className="text-[10px]">●</span>
          </div>
        )}

        {/* Start / Goal overlays */}
        {isStart && (
          <span className="absolute text-emerald-400 text-xs font-black drop-shadow-md">S</span>
        )}
        {isEnd && (
          <span className="absolute text-amber-400 text-xs font-black drop-shadow-md">G</span>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-teal-500 selection:text-slate-900">

      {/* ── Top Header Bar ── */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center space-x-3">
          <div className="h-9 w-9 bg-gradient-to-tr from-teal-500 to-cyan-500 rounded-lg flex items-center justify-center shadow-md shadow-teal-500/20">
            <Compass className="w-5 h-5 text-slate-950 font-black" />
          </div>
          <div className="text-left">
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-teal-400 to-cyan-300 bg-clip-text text-transparent">
              Tesseract Analyzer
            </h1>
            <p className="text-[10px] text-slate-400 font-mono">Structured-Loss Spatial Transformer Solver</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab('adversarial')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'adversarial' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="w-3.5 h-3.5 text-teal-400" />
            <span>Adversarial Mazes</span>
          </button>
          <button
            onClick={() => setActiveTab('testing')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'testing' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span>Test Set</span>
          </button>
          <button
            onClick={() => setActiveTab('progress')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-md text-xs font-semibold transition-all cursor-pointer ${
              activeTab === 'progress' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5 text-violet-400" />
            <span>v1-v5 History</span>
          </button>
        </nav>
      </header>

      {/* ── Main Layout Body ── */}
      <main className="flex-1 flex flex-col md:flex-row p-6 gap-6 max-w-7xl mx-auto w-full">

        {/* TAB 1 & 2: Maze visualizer dashboard */}
        {(activeTab === 'adversarial' || activeTab === 'testing') && activeMaze && (
          <>
            {/* Left Sidebar: Maze selector & parameters */}
            <div className="w-full md:w-80 shrink-0 flex flex-col gap-4">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 text-left">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    {activeTab === 'adversarial' ? 'Adversarial Collection' : 'Procedural Test Dataset'}
                  </h3>
                  <span className="bg-slate-850 text-slate-300 text-[10px] font-mono px-2 py-0.5 rounded border border-slate-700">
                    {activeTab === 'adversarial' ? `${selectedAdvIdx+1} / ${adversarialMazes.length}` : `${selectedTestIdx+1} / ${testMazes.length}`}
                  </span>
                </div>

                {/* List of mazes */}
                <div className="flex flex-col gap-1 max-h-56 overflow-y-auto pr-1">
                  {activeTab === 'adversarial' ? (
                    adversarialMazes.map((maze, idx) => (
                      <button
                        key={idx}
                        onClick={() => { setSelectedAdvIdx(idx); setCurrentStep(0); }}
                        className={`flex items-center justify-between p-2.5 rounded-lg text-left text-xs font-semibold border cursor-pointer transition-all ${
                          selectedAdvIdx === idx
                            ? 'bg-teal-500/10 border-teal-500/30 text-teal-300'
                            : 'bg-slate-950 border-transparent hover:bg-slate-850 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <div className={`h-2 w-2 rounded-full ${maze.tess_reached ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                          <span>{maze.name}</span>
                        </div>
                        <ChevronRight className="w-3 h-3 text-slate-500" />
                      </button>
                    ))
                  ) : (
                    testMazes.map((maze, idx) => (
                      <button
                        key={idx}
                        onClick={() => { setSelectedTestIdx(idx); setCurrentStep(0); }}
                        className={`flex items-center justify-between p-2.5 rounded-lg text-left text-xs font-semibold border cursor-pointer transition-all ${
                          selectedTestIdx === idx
                            ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                            : 'bg-slate-950 border-transparent hover:bg-slate-850 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center space-x-2">
                          <div className={`h-2 w-2 rounded-full ${maze.tess_reached ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                          <span>{maze.name}</span>
                        </div>
                        <ChevronRight className="w-3 h-3 text-slate-500" />
                      </button>
                    ))
                  )}
                </div>
              </div>

              {/* Loss Engine Commentary */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-3 flex-1 text-left">
                <div className="flex items-center space-x-2 text-teal-400 border-b border-slate-800 pb-2">
                  <HelpCircle className="w-4 h-4" />
                  <h4 className="text-xs font-bold uppercase tracking-wider">Analysis & Design</h4>
                </div>
                {activeTab === 'adversarial' ? (
                  <div className="text-xs text-slate-400 leading-relaxed flex flex-col gap-2.5">
                    {selectedAdvIdx === 0 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 1: The Detour</p>
                        <p>Requires moving <strong className="text-amber-400 font-semibold">DOWN and AWAY</strong> from the goal before crossing. Tesseract succeeds perfectly (1.0x optimal), while CE baseline gets completely lured into dead ends.</p>
                      </>
                    )}
                    {selectedAdvIdx === 1 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 2: The Spiral</p>
                        <p>Clockwise inward spiral corridor (59 steps). Both models fail due to sequence context limit and lack of extremely long corridors in standard training distribution.</p>
                      </>
                    )}
                    {selectedAdvIdx === 2 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 3: The False Shortcut</p>
                        <p>Diagonal corridor dead-ends. BFS path is monotonic toward goal. Both Tesseract and CE solve it, but CE is more optimal here.</p>
                      </>
                    )}
                    {selectedAdvIdx === 3 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 4: The Zigzag</p>
                        <p>Boustrophedon rows. Tests systematic traversal. Both models find the perfect path (46 cells) successfully.</p>
                      </>
                    )}
                    {selectedAdvIdx === 4 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 5: The Bottleneck</p>
                        <p>Two open regions joined by a single chokepoint. Extremely hard out-of-distribution topology. Both fail.</p>
                      </>
                    )}
                    {selectedAdvIdx === 5 && (
                      <>
                        <p className="font-semibold text-slate-200">Maze 6: The U-Turn</p>
                        <p>Perimeter traversal. Tesseract completes the full perimeter successfully (1.5x optimal) despite moving completely away from the goal, while CE baseline crashes.</p>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="text-xs text-slate-400 leading-relaxed">
                    <p className="mb-2">Procedural test set generated from completely unseen seeds.</p>
                    <p className="mb-2">These mazes verify the overall generalized robustness of the learned topological policies under standard DFS maze conditions with loop injection.</p>
                    <p>Observe how <span className="text-teal-400 font-semibold">Tesseract</span> consistently achieves a near-perfect path, outclassing the error-prone trajectories of <span className="text-rose-400 font-semibold">Cross Entropy</span>.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Right Main Panel: 10x10 visualizer and playback */}
            <div className="flex-1 flex flex-col lg:flex-row gap-6">

              {/* Visualizer Simulator Grid */}
              <div className="flex-1 bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4 items-center justify-center">
                <div className="w-full flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="text-left">
                    <h2 className="text-sm font-bold text-slate-200">{activeMaze.name}</h2>
                    <p className="text-[10px] text-slate-400">Autoregressive pathfinding visualizer (Step {currentStep} / {maxSteps - 1})</p>
                  </div>
                  <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-850">
                    <button
                      onClick={() => setSelectedModel('tess')}
                      className={`px-3 py-1 rounded-md text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer ${
                        selectedModel === 'tess' ? 'bg-teal-500 text-slate-950 shadow-md' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      Tesseract
                    </button>
                    <button
                      onClick={() => setSelectedModel('ce')}
                      className={`px-3 py-1 rounded-md text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer ${
                        selectedModel === 'ce' ? 'bg-rose-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      CE Baseline
                    </button>
                    <button
                      onClick={() => setSelectedModel('bfs')}
                      className={`px-3 py-1 rounded-md text-[10px] font-bold tracking-wider uppercase transition-all cursor-pointer ${
                        selectedModel === 'bfs' ? 'bg-violet-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      BFS (Optimal)
                    </button>
                  </div>
                </div>

                {/* The 10x10 Grid */}
                <div className="w-full max-w-sm aspect-square bg-slate-950 border border-slate-850 rounded-lg p-2 grid grid-cols-10 gap-[2px]">
                  {Array.from({ length: GRID_SIZE }).map((_, r) =>
                    Array.from({ length: GRID_SIZE }).map((_, c) => renderGridCell(r, c))
                  )}
                </div>

                {/* Simulation Control Interface */}
                <div className="w-full max-w-sm flex flex-col gap-3">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Playback Speed</span>
                    <div className="flex space-x-2">
                      <button onClick={() => setPlaySpeed(600)} className={`px-2 py-0.5 rounded font-mono text-[10px] cursor-pointer ${playSpeed === 600 ? 'bg-slate-800 text-white font-bold' : 'bg-transparent text-slate-500'}`}>0.5x</button>
                      <button onClick={() => setPlaySpeed(300)} className={`px-2 py-0.5 rounded font-mono text-[10px] cursor-pointer ${playSpeed === 300 ? 'bg-slate-800 text-white font-bold' : 'bg-transparent text-slate-500'}`}>1x</button>
                      <button onClick={() => setPlaySpeed(100)} className={`px-2 py-0.5 rounded font-mono text-[10px] cursor-pointer ${playSpeed === 100 ? 'bg-slate-800 text-white font-bold' : 'bg-transparent text-slate-500'}`}>3x</button>
                    </div>
                  </div>

                  {/* Slider & Buttons */}
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => setIsPlaying(!isPlaying)}
                      className="p-3 bg-teal-500 hover:bg-teal-400 text-slate-950 rounded-lg cursor-pointer transition-all shadow-md shadow-teal-500/15"
                    >
                      {isPlaying ? <Pause className="w-4 h-4 fill-slate-950" /> : <Play className="w-4 h-4 fill-slate-950" />}
                    </button>
                    <button
                      onClick={() => { setCurrentStep(0); setIsPlaying(false); }}
                      className="p-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg cursor-pointer transition-all"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>

                    <input
                      type="range"
                      min={0}
                      max={maxSteps - 1}
                      value={currentStep}
                      onChange={e => { setCurrentStep(Number(e.target.value)); setIsPlaying(false); }}
                      className="flex-1 accent-teal-500 cursor-pointer h-1 bg-slate-800 rounded-lg"
                    />
                  </div>
                </div>
              </div>

              {/* Simulation Metrics & Stats Card */}
              <div className="w-full lg:w-72 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col gap-4 text-left">
                <div className="flex items-center space-x-2 text-teal-400 border-b border-slate-800 pb-2">
                  <Award className="w-4 h-4" />
                  <h3 className="text-xs font-bold uppercase tracking-wider">Navigation Performance</h3>
                </div>

                <div className="flex flex-col gap-3 text-xs">
                  {/* BFS Length */}
                  <div className="flex justify-between items-center bg-slate-950 px-3 py-2.5 rounded-lg">
                    <span className="text-slate-400 font-semibold">BFS Optimal Path:</span>
                    <span className="font-bold font-mono text-violet-400">{activeMaze.bfs_path.length} cells</span>
                  </div>

                  {/* Selected Model Stats */}
                  <div className="bg-slate-950 p-3 rounded-lg flex flex-col gap-2">
                    <div className="flex justify-between items-center border-b border-slate-800/60 pb-2">
                      <span className="text-slate-400 font-semibold">Trajectory Taken:</span>
                      <span className="font-bold font-mono">{activePath.length} cells</span>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-slate-400 font-semibold">Success Status:</span>
                      {selectedModel === 'tess' ? (
                        activeMaze.tess_reached ? (
                          <span className="text-emerald-400 font-bold flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5" /> REACHED</span>
                        ) : (
                          <span className="text-rose-400 font-bold flex items-center gap-1"><XCircle className="w-3.5 h-3.5" /> {activeMaze.tess_limit ? 'TIMEOUT' : 'STUCK'}</span>
                        )
                      ) : selectedModel === 'ce' ? (
                        activeMaze.ce_reached ? (
                          <span className="text-emerald-400 font-bold flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5" /> REACHED</span>
                        ) : (
                          <span className="text-rose-400 font-bold flex items-center gap-1"><XCircle className="w-3.5 h-3.5" /> {activeMaze.ce_limit ? 'TIMEOUT' : 'STUCK'}</span>
                        )
                      ) : (
                        <span className="text-violet-400 font-bold flex items-center gap-1"><CheckCircle2 className="w-3.5 h-3.5" /> BFS DEFINITIVE</span>
                      )}
                    </div>

                    <div className="flex justify-between items-center pt-2 border-t border-slate-800/60">
                      <span className="text-slate-400 font-semibold">Optimality Ratio:</span>
                      <span className={`font-mono font-bold ${
                        selectedModel === 'bfs' ? 'text-violet-400' :
                        (selectedModel === 'tess' && activeMaze.tess_reached) ? 'text-teal-400' :
                        (selectedModel === 'ce' && activeMaze.ce_reached) ? 'text-rose-400' : 'text-slate-500'
                      }`}>
                        {selectedModel === 'bfs' ? '1.00x' :
                         selectedModel === 'tess' ? (activeMaze.tess_reached ? `${(activeMaze.tess_path.length / activeMaze.bfs_path.length).toFixed(2)}x` : 'N/A') :
                         (activeMaze.ce_reached ? `${(activeMaze.ce_path.length / activeMaze.bfs_path.length).toFixed(2)}x` : 'N/A')}
                      </span>
                    </div>
                  </div>

                  {/* General summary points */}
                  <div className="p-3 bg-slate-950/40 rounded-lg text-slate-400 leading-relaxed text-[11px] flex flex-col gap-1.5 border border-slate-850">
                    <div className="flex items-center space-x-1.5 text-slate-200 font-semibold mb-1">
                      <AlertCircle className="w-3.5 h-3.5 text-teal-400" />
                      <span>Heuristic vs. Reasoner</span>
                    </div>
                    <p>Heuristic solvers (like CE baseline) navigate by greedily matching Euclidean proximity, leading to crash failure on detours.</p>
                    <p>Tesseract models acquire structural invariants that allow backtracking and counter-intuitive planning steps.</p>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* TAB 3: Chronological History and Per-cell accuracies */}
        {activeTab === 'progress' && history && (
          <div className="flex-1 flex flex-col gap-6 w-full animate-fadeIn">

            {/* Top Cards: v1-v5 progression timeline */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4 text-left">
              <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
                <BarChart2 className="w-5 h-5 text-teal-400" />
                <h2 className="text-sm font-bold text-slate-200">The Tesseract Paradigm Progression (v1 - v5)</h2>
              </div>

              {/* The Timeline Row */}
              <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                {history.runs.map((run, idx) => (
                  <div key={idx} className="bg-slate-950 border border-slate-850 hover:border-teal-500/25 rounded-xl p-4 flex flex-col gap-2.5 transition-all text-left">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-mono uppercase bg-teal-500/10 border border-teal-500/20 px-1.5 py-0.5 rounded text-teal-400 font-bold">{run.version}</span>
                      <span className="text-[10px] text-slate-500 font-mono">{run.epochs} epochs</span>
                    </div>
                    <h3 className="text-xs font-bold text-slate-200 leading-tight">{run.name}</h3>
                    <div className="flex flex-col gap-0.5">
                      <div className="flex justify-between items-center text-[11px]">
                        <span className="text-slate-400">Success rate:</span>
                        <span className="font-bold font-mono text-teal-300">{run.success_rate.toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between items-center text-[11px]">
                        <span className="text-slate-400">Optimal rate:</span>
                        <span className="font-bold font-mono text-cyan-300">{run.optimal_rate.toFixed(1)}%</span>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 leading-relaxed border-t border-slate-900 pt-1.5">{run.notes}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Bottom Row: Stress test comparatives + Per-cell Heatmap */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">

              {/* Stress Test Comparison */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4 text-left">
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
                  <Award className="w-5 h-5 text-cyan-400" />
                  <h3 className="text-sm font-bold text-slate-200">Generalization Stress Test (500 Unseen Mazes)</h3>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed">
                  Evaluated across five entirely out-of-distribution seeds (100 mazes each).
                  Same transformer encoder architecture, same training data scope, only the loss function differs.
                </p>

                {/* SVG Visual bar chart */}
                <div className="flex flex-col gap-4 pt-2">
                  {/* Tesseract */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between items-center text-xs font-semibold">
                      <span className="text-teal-300 font-semibold">Tesseract Structured Loss (v5)</span>
                      <span className="font-mono text-teal-400 text-sm font-bold">{history.stress_test.tesseract_success}% (495/500)</span>
                    </div>
                    <div className="w-full bg-slate-950 h-5.5 rounded-lg overflow-hidden border border-slate-800/80 p-1 flex">
                      <div
                        className="bg-gradient-to-r from-teal-500 to-cyan-400 h-full rounded-md shadow-md"
                        style={{ width: `${history.stress_test.tesseract_success}%` }}
                      />
                    </div>
                  </div>

                  {/* CE Baseline */}
                  <div className="flex flex-col gap-1.5">
                    <div className="flex justify-between items-center text-xs font-semibold">
                      <span className="text-rose-400 font-semibold">Cross-Entropy Loss Baseline</span>
                      <span className="font-mono text-rose-500 text-sm font-bold">{history.stress_test.ce_baseline_success}% (109/500)</span>
                    </div>
                    <div className="w-full bg-slate-950 h-5.5 rounded-lg overflow-hidden border border-slate-800/80 p-1 flex">
                      <div
                        className="bg-gradient-to-r from-rose-600 to-pink-500 h-full rounded-md shadow-md"
                        style={{ width: `${history.stress_test.ce_baseline_success}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-850 text-[11px] text-slate-400 leading-relaxed mt-auto">
                  <span className="font-bold text-slate-200 block mb-1">Empirical Conclusion:</span>
                  Standard CE model optimization forces curve-fitting on easy distributions, resulting in terrible out-of-distribution failure.
                  Tesseract's structured loss prevents gradient dilution, achieving an ultra-robust, invariant policy.
                </div>
              </div>

              {/* Per-Cell Accuracy Heatmap (Implicit Curriculum visualizer) */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col gap-4 text-left">
                <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
                  <Compass className="w-5 h-5 text-violet-400" />
                  <h3 className="text-sm font-bold text-slate-200">Per-Cell Validation Accuracy Heatmap (v4)</h3>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed">
                  Validation accuracies decomposed across the 6 Tesseract problem cells.
                  Observe how endgame cells are mastered first, and early-game forks remain the most challenging bottlenecks.
                </p>

                {/* Grid Heatmap representation */}
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div className="flex flex-col gap-2">
                    <h4 className="text-[10px] uppercase font-bold tracking-wider text-slate-400 text-center">Corridors (≤ 2 Neighbors)</h4>

                    {/* Endgame Corridor */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Endgame</span>
                      <div className="h-10 w-10 bg-teal-500/80 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["endgame-corridor"]}%
                      </div>
                    </div>

                    {/* Mid Corridor */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Mid-Game</span>
                      <div className="h-10 w-10 bg-teal-500/60 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["mid-corridor"]}%
                      </div>
                    </div>

                    {/* Early Corridor */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Early-Game</span>
                      <div className="h-10 w-10 bg-teal-500/50 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["early-corridor"]}%
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-col gap-2">
                    <h4 className="text-[10px] uppercase font-bold tracking-wider text-slate-400 text-center">Forks (&gt; 2 Neighbors)</h4>

                    {/* Endgame Fork */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Endgame</span>
                      <div className="h-10 w-10 bg-teal-500/70 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["endgame-fork"]}%
                      </div>
                    </div>

                    {/* Mid Fork */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Mid-Game</span>
                      <div className="h-10 w-10 bg-teal-500/75 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["mid-fork"]}%
                      </div>
                    </div>

                    {/* Early Fork */}
                    <div className="bg-slate-950 border border-slate-850 p-3 rounded-lg flex flex-col items-center justify-center gap-1.5">
                      <span className="text-[10px] text-slate-500 uppercase font-semibold">Early-Game</span>
                      <div className="h-10 w-10 bg-teal-500/55 text-slate-950 font-black rounded flex items-center justify-center font-mono shadow-md shadow-teal-500/10">
                        {history.per_cell_val_accuracy_v4["early-fork"]}%
                      </div>
                    </div>
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer copyright */}
      <footer className="border-t border-slate-900 bg-slate-950 text-slate-500 text-center py-6 text-[10px] font-mono mt-auto">
        © March 2026 Tesseract Loss Innovation Research project. Built with React + TypeScript. Deployed to Netlify.
      </footer>
    </div>
  );
}
