import { useState } from 'react';
import { Play, Clock, Zap, HardDrive, Type, Mic, AlertTriangle, GitBranch } from 'lucide-react';
import { travelOrderAPI } from '../services/api';
import AudioRecorder from './AudioRecorder';
import type { NLPBenchmarkResult, GraphBenchmarkResult } from '../types';

type Phase = 'idle' | 'nlp' | 'graph' | 'done' | 'error';
type InputMode = 'text' | 'voice';

const OPT_LABELS: Record<string, string> = {
  time: 'Temps',
  pollution: 'Pollution',
  hops: 'Correspondances',
};

const EXAMPLES = [
  { text: 'Je veux aller de Paris a Lyon', dep: 'Paris Montparnasse Hall 1 - 2', dest: 'Lyon Part Dieu' },
  { text: 'Un billet Marseille direction Bordeaux', dep: 'Marseille Saint-Charles', dest: 'Bordeaux Saint-Jean' },
  { text: 'Emmene moi de Lille a Strasbourg', dep: 'Lille Flandres', dest: 'Strasbourg' },
  { text: 'Je voudrais partir de Nantes pour aller a Rennes', dep: 'Nantes', dest: 'Rennes' },
];

function formatEnergy(uwh: number): string {
  if (uwh < 1000) return `${uwh.toFixed(1)} \u00b5Wh`;
  return `${(uwh / 1000).toFixed(2)} mWh`;
}

function MatchBadge({ match }: { match: NLPBenchmarkResult['match'] }) {
  const config = {
    correct: { label: 'correct', cls: 'match-correct' },
    inverted: { label: 'inverse', cls: 'match-inverted' },
    wrong: { label: 'faux', cls: 'match-wrong' },
    missing: { label: 'manquant', cls: 'match-missing' },
  };
  const { label, cls } = config[match] || config.missing;
  return <span className={`match-badge ${cls}`}>{match === 'correct' ? '\u2713' : '\u2717'} {label}</span>;
}

function BenchmarkPanel() {
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [text, setText] = useState('');
  const [expectedDep, setExpectedDep] = useState('');
  const [expectedDest, setExpectedDest] = useState('');
  const [phase, setPhase] = useState<Phase>('idle');
  const [nlpResults, setNlpResults] = useState<NLPBenchmarkResult[]>([]);
  const [graphResults, setGraphResults] = useState<GraphBenchmarkResult[]>([]);
  const [nlpTotal, setNlpTotal] = useState(0);
  const [graphTotal, setGraphTotal] = useState(0);
  const [errorMsg, setErrorMsg] = useState('');
  const [transcribing, setTranscribing] = useState(false);

  const isRunning = phase === 'nlp' || phase === 'graph';
  const canRun = text.trim() && expectedDep.trim() && expectedDest.trim() && !isRunning;

  const runBenchmark = async (inputText: string, dep: string, dest: string) => {
    setNlpResults([]);
    setGraphResults([]);
    setNlpTotal(0);
    setGraphTotal(0);
    setErrorMsg('');
    setPhase('nlp');

    try {
      for await (const event of travelOrderAPI.benchmarkStream(inputText, dep, dest)) {
        switch (event.type) {
          case 'phase':
            setPhase(event.phase);
            if (event.phase === 'nlp') setNlpTotal(event.total);
            if (event.phase === 'graph') setGraphTotal(event.total);
            break;
          case 'nlp_result':
            setNlpResults(prev => [...prev, event as unknown as NLPBenchmarkResult]);
            break;
          case 'graph_result':
            setGraphResults(prev => [...prev, event as unknown as GraphBenchmarkResult]);
            break;
          case 'done':
            setPhase('done');
            break;
        }
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Unknown error');
      setPhase('error');
    }
  };

  const handleRun = () => {
    if (!canRun) return;
    runBenchmark(text, expectedDep, expectedDest);
  };

  const handleExample = (ex: typeof EXAMPLES[0]) => {
    setText(ex.text);
    setExpectedDep(ex.dep);
    setExpectedDest(ex.dest);
  };

  const handleAudioUpload = async (file: File) => {
    setTranscribing(true);
    try {
      const result = await travelOrderAPI.transcribe(file);
      setText(result.transcribed_text);
    } catch (err) {
      setErrorMsg('Transcription failed: ' + (err instanceof Error ? err.message : 'unknown error'));
    } finally {
      setTranscribing(false);
    }
  };

  return (
    <div className="benchmark-panel">
      <p className="benchmark-description">
        Compare all NLP and pathfinding engines side by side on the same input.
        Measures extraction accuracy, response time, energy consumption and memory usage for each strategy.
      </p>
      {/* Input mode toggle */}
      <div className="benchmark-mode-toggle">
        <button
          className={`mode-btn ${inputMode === 'text' ? 'active' : ''}`}
          onClick={() => setInputMode('text')}
          disabled={isRunning}
        >
          <Type size={16} /> Texte
        </button>
        <button
          className={`mode-btn ${inputMode === 'voice' ? 'active' : ''}`}
          onClick={() => setInputMode('voice')}
          disabled={isRunning}
        >
          <Mic size={16} /> Voix
        </button>
      </div>

      {/* Text input or audio recorder */}
      {inputMode === 'text' ? (
        <div className="benchmark-text-input">
          <input
            type="text"
            className="benchmark-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ex: Je veux aller de Paris a Lyon..."
            disabled={isRunning}
          />
          <div className="examples">
            <p className="examples-title">Exemples rapides :</p>
            <div className="examples-grid">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  className="example-chip"
                  onClick={() => handleExample(ex)}
                  disabled={isRunning}
                >
                  {ex.text}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="benchmark-voice-input">
          <AudioRecorder
            onAudioUpload={handleAudioUpload}
            loading={transcribing || isRunning}
            showMetric={false}
          />
          {text && (
            <div className="benchmark-transcription">
              Texte transcrit : <strong>{text}</strong>
            </div>
          )}
        </div>
      )}

      {/* Ground truth inputs */}
      <div className="benchmark-ground-truth">
        <div className="ground-truth-field">
          <label>Depart attendu</label>
          <input
            type="text"
            className="benchmark-input"
            value={expectedDep}
            onChange={(e) => setExpectedDep(e.target.value)}
            placeholder="Ex: Paris Gare de Lyon"
            disabled={isRunning}
          />
        </div>
        <div className="ground-truth-field">
          <label>Destination attendue</label>
          <input
            type="text"
            className="benchmark-input"
            value={expectedDest}
            onChange={(e) => setExpectedDest(e.target.value)}
            placeholder="Ex: Lyon Part Dieu"
            disabled={isRunning}
          />
        </div>
      </div>

      {/* Run button */}
      <button
        className="benchmark-run-btn benchmark-run-main"
        onClick={handleRun}
        disabled={!canRun || isRunning}
      >
        <Play size={18} />
        {isRunning ? 'Benchmark en cours...' : 'Lancer le benchmark'}
      </button>

      {/* Error */}
      {errorMsg && (
        <div className="benchmark-error">
          {errorMsg}
        </div>
      )}

      {/* Phase 1: NLP results */}
      {(phase !== 'idle' || nlpResults.length > 0) && (
        <div className="benchmark-phase">
          <div className="phase-header">
            <h3>Phase 1 : Analyse NLP</h3>
            <span className="phase-counter">{nlpResults.length}/{nlpTotal || '?'}</span>
          </div>
          {nlpTotal > 0 && (
            <div className="phase-progress">
              <div
                className="progress-bar"
                style={{ width: `${(nlpResults.length / nlpTotal) * 100}%` }}
              />
            </div>
          )}
          <div className="phase-ground-truth">
            Verite terrain : {expectedDep} → {expectedDest}
          </div>
          <div className="engine-cards">
            {nlpResults.map((r, i) => (
              <div
                key={i}
                className={`engine-card ${r.error ? 'error' : r.is_valid ? 'valid' : 'invalid'} fade-in-up`}
              >
                <div className="engine-card-header">
                  <span className="engine-card-name">{r.engine_name}</span>
                  <MatchBadge match={r.match} />
                </div>
                {r.error ? (
                  <div className="engine-card-error">{r.error}</div>
                ) : (
                  <>
                    <div className="engine-card-extract">
                      {r.departure || '(vide)'} → {r.destination || '(vide)'}
                    </div>
                    <div className="engine-card-metrics">
                      <span className="metric"><Clock size={14} /> {r.time_ms.toFixed(1)}ms</span>
                      <span className="metric"><Zap size={14} /> {formatEnergy(r.energy_uwh)}</span>
                      <span className="metric"><HardDrive size={14} /> {r.memory_kb.toFixed(0)}KB</span>
                    </div>
                  </>
                )}
              </div>
            ))}
            {phase === 'nlp' && nlpResults.length < nlpTotal && (
              <div className="engine-card pending">
                <div className="engine-card-header">
                  <span className="engine-card-name">...</span>
                  <span className="mini-spinner" />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Phase 2: Graph results grouped by optimization mode */}
      {(phase === 'graph' || phase === 'done' || graphResults.length > 0) && (
        <div className="benchmark-phase">
          <div className="phase-header">
            <h3>Phase 2 : Pathfinding</h3>
            <span className="phase-counter">{graphResults.length}/{graphTotal || '?'}</span>
          </div>
          {graphTotal > 0 && (
            <div className="phase-progress">
              <div
                className="progress-bar"
                style={{ width: `${(graphResults.length / graphTotal) * 100}%` }}
              />
            </div>
          )}
          <div className="phase-ground-truth">
            Route : {expectedDep} → {expectedDest} (verite terrain)
          </div>
          {['time', 'pollution', 'hops'].map((optMode) => {
            const modeResults = graphResults.filter((r) => r.optimization === optMode);
            if (modeResults.length === 0 && phase !== 'graph') return null;
            return (
              <div key={optMode} className="optimization-group">
                <div className="optimization-group-header">
                  <GitBranch size={14} />
                  <span>{OPT_LABELS[optMode] || optMode}</span>
                </div>
                <div className="engine-cards">
                  {modeResults.map((r, i) => (
                    <div
                      key={i}
                      className={`engine-card ${r.error ? 'error' : r.found ? 'valid' : 'invalid'} fade-in-up`}
                    >
                      <div className="engine-card-header">
                        <span className="engine-card-name">{r.engine_name}</span>
                        <span className={`validity-badge ${r.found ? 'valid' : 'invalid'}`}>
                          {r.found ? `\u2713 ${r.steps?.length || 0} gares` : '\u2717 pas de route'}
                        </span>
                      </div>
                      {r.error ? (
                        <div className="engine-card-error">{r.error}</div>
                      ) : (
                        <>
                          {r.warning && (
                            <div className="engine-card-warning">
                              <AlertTriangle size={14} /> {r.warning}
                            </div>
                          )}
                          <div className="engine-card-metrics">
                            <span className="metric"><Clock size={14} /> {r.time_ms.toFixed(1)}ms</span>
                            <span className="metric"><Zap size={14} /> {formatEnergy(r.energy_uwh)}</span>
                            <span className="metric"><HardDrive size={14} /> {r.memory_kb.toFixed(0)}KB</span>
                          </div>
                        </>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
          {phase === 'graph' && graphResults.length < graphTotal && (
            <div className="engine-card pending">
              <div className="engine-card-header">
                <span className="engine-card-name">...</span>
                <span className="mini-spinner" />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default BenchmarkPanel;
