import React, { useState, useRef, useEffect } from 'react';
import { Upload, Camera, AlertCircle, CheckCircle, Loader, Shield, Wifi, WifiOff } from 'lucide-react';
import axios from 'axios';
import './App.css';

// Config - you can change this if your backend runs on different port
const API_BASE = 'http://localhost:5000/api';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [demoMode, setDemoMode] = useState(false);
  const [serverOnline, setServerOnline] = useState(false);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);

  // Check if server is running
  const checkServerStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE}/health`, { timeout: 5000 });
      setServerOnline(true);
      setError('');
      console.log('Server is online:', response.data);
    } catch (err) {
      setServerOnline(false);
      console.log('Server is offline');
    }
  };

  useEffect(() => {
    checkServerStatus();
    // Check every 5 seconds
    const interval = setInterval(checkServerStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please upload an image file (JPEG, PNG, etc.)');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target.result);
        setPrediction(null);
        setError('');
        setDemoMode(false);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    if (!serverOnline) {
      setError('Backend server is not available. Please make sure it\'s running on port 5000.');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);
    
    try {
      const response = await axios.post(`${API_BASE}/predict`, {
        image: selectedImage
      }, {
        timeout: 30000,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      if (response.data.status === 'success') {
        setPrediction(response.data);
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleApiError = (err) => {
    if (err.code === 'NETWORK_ERROR' || err.message.includes('Network Error')) {
      setError('Cannot connect to server. Please check: 1) Backend is running, 2) No firewall blocking, 3) Correct port (5000)');
    } else if (err.response?.status === 404) {
      setError('Server endpoint not found. The backend might be running on a different port.');
    } else if (err.response?.data?.error) {
      setError(err.response.data.error);
    } else if (err.message.includes('timeout')) {
      setError('Request timeout. The server is taking too long to respond.');
    } else {
      setError(`Connection failed: ${err.message}`);
    }
  };

  const handleDemoMode = async () => {
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.get(`${API_BASE}/demo`, { timeout: 10000 });
      setPrediction({ ...response.data, demo: true });
      setDemoMode(true);
    } catch (err) {
      setError('Demo mode unavailable. Server might be offline.');
    } finally {
      setLoading(false);
    }
  };

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      setStream(mediaStream);
      videoRef.current.srcObject = mediaStream;
      setError('');
    } catch (err) {
      setError(`Camera access denied: ${err.message}`);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
  };

  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const imageData = canvas.toDataURL('image/jpeg');
    setSelectedImage(imageData);
    setPrediction(null);
    setDemoMode(false);
    stopCamera();
  };

  const reset = () => {
    setSelectedImage(null);
    setPrediction(null);
    setError('');
    setDemoMode(false);
    stopCamera();
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const retryConnection = () => {
    checkServerStatus();
    setError('');
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <Shield size={32} />
          <h1>Traffic Sign Classifier</h1>
          <div className={`server-status ${serverOnline ? 'online' : 'offline'}`}>
            {serverOnline ? <Wifi size={20} /> : <WifiOff size={20} />}
            <span>Backend {serverOnline ? 'Online' : 'Offline'}</span>
          </div>
        </div>
        <p>Upload or capture an image to identify traffic signs</p>
      </header>

      <main className="app-main">
        <div className="container">
          {/* Server Status Alert */}
          {!serverOnline && (
            <div className="server-alert">
              <AlertCircle size={20} />
              <div>
                <strong>Backend Server Offline</strong>
                <p>Please make sure the Python backend is running on port 5000</p>
                <button className="retry-btn" onClick={retryConnection}>
                  Retry Connection
                </button>
              </div>
            </div>
          )}

          {/* Upload Section */}
          <div className="upload-section">
            <div className="upload-options">
              <button
                className="upload-btn"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading}
              >
                <Upload size={20} />
                Upload Image
              </button>
              
              <button
                className="camera-btn"
                onClick={stream ? stopCamera : startCamera}
                disabled={loading}
              >
                <Camera size={20} />
                {stream ? 'Stop Camera' : 'Use Camera'}
              </button>

              <button
                className="demo-btn"
                onClick={handleDemoMode}
                disabled={loading || !serverOnline}
              >
                <CheckCircle size={20} />
                Try Demo
              </button>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
            </div>

            {/* Camera Preview */}
            {stream && (
              <div className="camera-preview">
                <video ref={videoRef} autoPlay playsInline />
                <button className="capture-btn" onClick={captureImage}>
                  Capture Image
                </button>
              </div>
            )}

            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>

          {/* Image Preview */}
          {selectedImage && (
            <div className="image-preview">
              <h3>Selected Image</h3>
              <div className="image-container">
                <img src={selectedImage} alt="Selected for classification" />
              </div>
              <div className="preview-actions">
                <button 
                  className="predict-btn" 
                  onClick={handlePredict}
                  disabled={loading || !serverOnline}
                >
                  {loading ? (
                    <>
                      <Loader size={16} className="spinner" />
                      Analyzing...
                    </>
                  ) : (
                    'Classify Sign'
                  )}
                </button>
                <button className="reset-btn" onClick={reset} disabled={loading}>
                  Reset
                </button>
              </div>
            </div>
          )}

          {/* Prediction Result */}
          {prediction && (
            <div className={`prediction-result ${demoMode ? 'demo-result' : ''}`}>
              <div className="result-header">
                <CheckCircle size={24} className="success-icon" />
                <div>
                  <h3>Classification Result</h3>
                  {demoMode && <span className="demo-badge">Demo Mode</span>}
                </div>
              </div>
              
              <div className="result-details">
                <div className="result-item">
                  <strong>Sign Type:</strong>
                  <span className="class-name">{prediction.class_name}</span>
                </div>
                <div className="result-item">
                  <strong>Confidence:</strong>
                  <span className="confidence">{prediction.confidence}%</span>
                </div>
                <div className="result-item">
                  <strong>Class ID:</strong>
                  <span>{prediction.class_id}</span>
                </div>
              </div>
              
              {demoMode && (
                <div className="demo-note">
                  <p>This is a demo prediction. Upload a real traffic sign image for actual classification.</p>
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="error-message">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;