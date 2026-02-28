import { useState, useRef } from 'react';
import { Mic, Upload, Clock } from 'lucide-react';

interface AudioRecorderProps {
  onAudioUpload: (file: File, metric?: 'time' | 'distance') => void;
  loading: boolean;
  showMetric?: boolean;
}

function AudioRecorder({ onAudioUpload, loading, showMetric = true }: AudioRecorderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [metric, setMetric] = useState<'time' | 'distance'>('time');
  const [recording, setRecording] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      if (!showMetric) {
        onAudioUpload(file);
      }
    }
  };

  const handleUpload = () => {
    if (selectedFile) {
      onAudioUpload(selectedFile, metric);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const file = new File([audioBlob], 'recording.wav', { type: 'audio/wav' });
        setSelectedFile(file);
        stream.getTracks().forEach((track) => track.stop());
        if (!showMetric) {
          onAudioUpload(file);
        }
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (error) {
      console.error('Erreur lors du démarrage de l\'enregistrement:', error);
      alert('Impossible d\'accéder au microphone. Vérifiez les permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  return (
    <div className="audio-recorder">
      <div className="recorder-section">
        <div className="recorder-card">
          <Mic size={48} className="recorder-icon" />
          <h3>Enregistrer votre commande</h3>
          <p className="recorder-help">
            Cliquez pour enregistrer votre commande vocale
          </p>

          {!recording ? (
            <button
              className="record-button"
              onClick={startRecording}
              disabled={loading}
            >
              <Mic size={20} />
              Commencer l'enregistrement
            </button>
          ) : (
            <button
              className="record-button recording"
              onClick={stopRecording}
            >
              <div className="recording-indicator"></div>
              Arrêter l'enregistrement
            </button>
          )}
        </div>

        <div className="divider">
          <span>OU</span>
        </div>

        <div className="upload-card">
          <Upload size={48} className="upload-icon" />
          <h3>Importer un fichier audio</h3>
          <p className="upload-help">
            Formats supportés : .wav, .mp3, .ogg
          </p>

          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileSelect}
            className="file-input"
            disabled={loading}
          />

          <button
            className="upload-button"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
          >
            <Upload size={20} />
            Sélectionner un fichier
          </button>

          {selectedFile && (
            <div className="selected-file">
              Fichier sélectionné : <strong>{selectedFile.name}</strong>
            </div>
          )}
        </div>
      </div>

      {selectedFile && showMetric && (
        <div className="submit-section">
          <div className="metric-select">
            <label className="metric-label">Optimisation :</label>
            <div className="metric-buttons">
              <button
                type="button"
                className={`metric-btn ${metric === 'time' ? 'active' : ''}`}
                onClick={() => setMetric('time')}
                disabled={loading}
              >
                <Clock size={16} />
                Temps
              </button>
              <button
                type="button"
                className={`metric-btn ${metric === 'distance' ? 'active' : ''}`}
                onClick={() => setMetric('distance')}
                disabled={loading}
              >
                Distance
              </button>
            </div>
          </div>

          <button
            className="submit-button"
            onClick={handleUpload}
            disabled={loading}
          >
            {loading ? 'Transcription en cours...' : 'Transcrire et rechercher'}
          </button>
        </div>
      )}
    </div>
  );
}

export default AudioRecorder;
