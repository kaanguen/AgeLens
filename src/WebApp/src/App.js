import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  
  const [stream, setStream] = useState(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);

  // Start Camera
  const startCamera = async () => {
    try {
      setError(null);
      
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 224 },
          height: { ideal: 224 }
        },
        audio: false
      });
      
      console.log('✅ Stream obtained:', mediaStream);
      console.log('✅ Video tracks:', mediaStream.getVideoTracks());
      
      setStream(mediaStream);
      setIsCameraActive(true);
      
      // Attach stream to video element
      if (videoRef.current) {
        console.log('Attaching stream to video element...');
        videoRef.current.srcObject = mediaStream;
        
        // Try to play immediately
        videoRef.current.play()
          .then(() => console.log('✅ Video playing'))
          .catch(err => {
            console.error('Play error:', err);
            // Try again after a short delay
            setTimeout(() => {
              videoRef.current?.play()
                .then(() => console.log('✅ Video playing on retry'))
                .catch(e => console.error('Retry play error:', e));
            }, 500);
          });
      }
      
    } catch (err) {
      console.error('❌ Camera access error:', err);
      setIsCameraActive(false);
      
      let errorMsg = 'Failed to access camera: ' + err.message;
      
      if (err.name === 'NotAllowedError') {
        errorMsg = '❌ Camera permission denied. Please allow camera access.';
      } else if (err.name === 'NotFoundError') {
        errorMsg = '❌ No camera found. Check if camera is connected.';
      } else if (err.name === 'NotReadableError') {
        errorMsg = '❌ Camera is in use by another app. Close it and try again.';
      }
      
      setError(errorMsg);
    }
  };

  // Stop Camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCameraActive(false);
  };

  // Capture Photo from Camera
  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const context = canvasRef.current.getContext('2d');
    const video = videoRef.current;
    
    canvasRef.current.width = video.videoWidth;
    canvasRef.current.height = video.videoHeight;
    
    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    
    const dataUrl = canvasRef.current.toDataURL('image/jpeg');
    setCapturedImage(dataUrl);
    
    stopCamera();
  };

  // Handle File Upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setCapturedImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Make Prediction
  const makePrediction = async () => {
    if (!capturedImage) {
      setError('Please capture or upload an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(capturedImage);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');

      const apiResponse = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setPrediction(apiResponse.data);
    } catch (err) {
      setError('Prediction failed: ' + (err.response?.data?.detail || err.message));
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Clear Everything
  const clearAll = () => {
    setCapturedImage(null);
    setPrediction(null);
    setError(null);
  };

  // Save Prediction (placeholder)
  const savePrediction = () => {
    // TODO: Implement save functionality
    console.log('Saving prediction:', prediction);
  };

  useEffect(() => {
    // Wenn Kamera aktiv ist und Video-Element vorhanden
    if (isCameraActive && stream && videoRef.current) {
      console.log('useEffect: Binding stream to video element');
      videoRef.current.srcObject = stream;
      
      videoRef.current.play()
        .then(() => console.log('✅ Video playing from useEffect'))
        .catch(err => console.error('Play error from useEffect:', err));
    }
    
    return () => {
      // Cleanup: Stop all tracks when component unmounts
      if (stream && !isCameraActive) {
        console.log('Cleaning up stream');
        stream.getTracks().forEach(track => {
          track.stop();
          console.log('Track stopped:', track.kind);
        });
      }
    };
  }, [isCameraActive, stream]);

  return (
    <div className="bg-background text-on-background font-body-lg min-h-screen pb-24">
      {/* Header */}
      <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-gutter py-stack-sm bg-surface/60 backdrop-blur-md dark:bg-surface-container-lowest/60 border-b border-outline-variant/10">
        <div className="flex items-center gap-2">
          <span className="font-headline-md text-headline-md font-bold text-primary">AgeLens</span>
        </div>
        <div className="flex items-center gap-gutter">
          <button className="p-2 rounded-full hover:bg-surface-variant/20 transition-colors text-on-surface-variant">
            <span className="material-symbols-outlined">settings</span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-[72px] pb-[100px] max-w-4xl mx-auto px-container-padding-mobile md:px-container-padding-desktop">
        {/* Camera Preview */}
        <div className="relative w-full bg-surface-container-lowest rounded-xl overflow-hidden mb-stack-lg border border-outline-variant/20 shadow-2xl" style={{ aspectRatio: '1 / 1', maxWidth: '224px', margin: '0 auto 24px', minHeight: '224px' }}>
          {isCameraActive ? (
            <video
              ref={videoRef}
              autoPlay={true}
              playsInline={true}
              muted={true}
              controls={false}
              className="w-full h-full object-cover bg-black"
              style={{ 
                display: 'block', 
                width: '100%', 
                height: '100%',
                backgroundColor: '#000',
                objectFit: 'cover'
              }}
            />
          ) : capturedImage ? (
            <img src={capturedImage} alt="Captured" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-surface-container-low">
              <div className="text-center">
                <span className="material-symbols-outlined text-6xl text-on-surface-variant opacity-30">image_not_supported</span>
                <p className="text-on-surface-variant mt-stack-md">No image selected</p>
              </div>
            </div>
          )}
          
          {/* Scanning Overlay */}
          {isCameraActive && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="w-[80%] h-[70%] border-2 border-primary/40 rounded-lg relative overflow-hidden">
                <div className="scanning-line"></div>
                <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-primary"></div>
                <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-primary"></div>
                <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-primary"></div>
                <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-primary"></div>
              </div>
            </div>
          )}
          
          {/* Auto-focus indicator */}
          {(isCameraActive || capturedImage) && (
            <div className="absolute bottom-gutter left-gutter">
              <div className="flex items-center gap-2 bg-surface-container-high/80 backdrop-blur-md px-3 py-1.5 rounded-full border border-outline-variant/30">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
                <span className="font-label-caps text-label-caps text-on-surface">{isCameraActive ? 'CAMERA ACTIVE' : 'IMAGE READY'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Camera Controls */}
        <div className="flex flex-col items-center mb-stack-lg gap-stack-md">
          {!isCameraActive && !capturedImage && (
            <>
              <button 
                onClick={startCamera}
                className="shutter-ring rounded-full active:scale-95 transition-transform duration-200 group">
                <div className="w-16 h-16 bg-primary-container rounded-full flex items-center justify-center text-on-primary-container shadow-lg group-hover:bg-primary transition-colors cursor-pointer">
                  <span className="material-symbols-outlined text-[32px]">photo_camera</span>
                </div>
              </button>
              <span className="font-label-caps text-label-caps text-on-surface-variant">START CAMERA</span>
            </>
          )}

          {isCameraActive && (
            <>
              <button 
                onClick={capturePhoto}
                className="shutter-ring rounded-full active:scale-95 transition-transform duration-200 group">
                <div className="w-16 h-16 bg-primary-container rounded-full flex items-center justify-center text-on-primary-container shadow-lg group-hover:bg-primary transition-colors cursor-pointer">
                  <span className="material-symbols-outlined text-[32px]">photo_camera</span>
                </div>
              </button>
              <span className="font-label-caps text-label-caps text-on-surface-variant">CAPTURE PHOTO</span>
            </>
          )}

          {capturedImage && !isCameraActive && (
            <>
              <button 
                onClick={clearAll}
                className="shutter-ring rounded-full active:scale-95 transition-transform duration-200 group">
                <div className="w-16 h-16 bg-error rounded-full flex items-center justify-center text-on-primary-container shadow-lg group-hover:bg-error transition-colors cursor-pointer">
                  <span className="material-symbols-outlined text-[32px]">refresh</span>
                </div>
              </button>
              <span className="font-label-caps text-label-caps text-on-surface-variant">RESET</span>
            </>
          )}
        </div>

        {/* File Upload & Control Buttons */}
        <div className="flex gap-stack-md mb-stack-lg justify-center flex-wrap">
          {!isCameraActive && (
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="px-stack-lg py-stack-md bg-surface-container hover:bg-surface-container-high border border-outline-variant text-on-surface font-label-caps text-label-caps rounded-xl transition-colors flex items-center gap-2">
              <span className="material-symbols-outlined text-[18px]">folder_open</span>
              UPLOAD PHOTO
            </button>
          )}
          
          {isCameraActive && (
            <button 
              onClick={stopCamera}
              className="px-stack-lg py-stack-md bg-error-container hover:bg-error text-error font-label-caps text-label-caps rounded-xl transition-colors flex items-center gap-2">
              <span className="material-symbols-outlined text-[18px]">stop</span>
              STOP CAMERA
            </button>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
        </div>

        {/* Canvas (hidden) */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Error Display */}
        {error && (
          <div className="mb-stack-lg p-stack-md bg-error-container/30 border border-error/50 rounded-xl">
            <p className="text-error font-body-sm">⚠️ {error}</p>
          </div>
        )}

        {/* Results Section */}
        {(prediction || capturedImage) && (
          <section className="space-y-stack-lg">
            <div className="flex items-center justify-between">
              <h2 className="font-headline-md text-headline-md text-on-surface">Results</h2>
              {prediction && (
                <div className="flex items-center gap-2 px-3 py-1 bg-tertiary-container/30 border border-tertiary/20 rounded-full">
                  <span className="material-symbols-outlined text-sm text-tertiary">verified</span>
                  <span className="font-label-caps text-label-caps text-tertiary">
                    {prediction.gender_confidence && 
                      (Math.max(prediction.gender_confidence, prediction.race_confidence || 0) * 100).toFixed(1)}% Confidence
                    }
                  </span>
                </div>
              )}
            </div>

            {prediction ? (
              <>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-stack-md">
                  <div className="bg-surface-container-low p-stack-md rounded-xl border border-outline-variant/10 hover:border-primary/30 transition-colors group">
                    <label className="block font-label-caps text-label-caps text-on-surface-variant mb-stack-sm">AGE</label>
                    <div className="flex items-center justify-between">
                      <span className="font-data-mono text-data-mono text-primary">{Math.round(prediction.age)} years</span>
                    </div>
                  </div>

                  <div className="bg-surface-container-low p-stack-md rounded-xl border border-outline-variant/10 hover:border-primary/30 transition-colors group">
                    <label className="block font-label-caps text-label-caps text-on-surface-variant mb-stack-sm">GENDER</label>
                    <div className="flex items-center justify-between">
                      <span className="font-data-mono text-data-mono text-on-surface capitalize">{prediction.gender}</span>
                      <span className="text-xs text-on-surface-variant">{(prediction.gender_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>

                  <div className="bg-surface-container-low p-stack-md rounded-xl border border-outline-variant/10 hover:border-primary/30 transition-colors group md:col-span-2">
                    <label className="block font-label-caps text-label-caps text-on-surface-variant mb-stack-sm">ETHNICITY</label>
                    <div className="flex items-center justify-between">
                      <span className="font-data-mono text-data-mono text-on-surface capitalize">{prediction.race}</span>
                      <span className="text-xs text-on-surface-variant">{(prediction.race_confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                <div className="flex gap-stack-md">
                  <button 
                    onClick={savePrediction}
                    className="flex-1 py-stack-md px-stack-lg bg-primary text-on-primary font-label-caps text-label-caps rounded-xl hover:opacity-90 transition-opacity flex items-center justify-center gap-2 cursor-pointer">
                    <span className="material-symbols-outlined text-[18px]">save</span>
                    SAVE RESULTS
                  </button>
                  <button 
                    onClick={clearAll}
                    className="py-stack-md px-stack-lg border border-outline-variant text-on-surface font-label-caps text-label-caps rounded-xl hover:bg-surface-variant/20 transition-colors cursor-pointer">
                    NEW SCAN
                  </button>
                </div>
              </>
            ) : capturedImage && !loading ? (
              <div className="flex gap-stack-md">
                <button 
                  onClick={makePrediction}
                  className="flex-1 py-stack-md px-stack-lg bg-primary text-on-primary font-label-caps text-label-caps rounded-xl hover:opacity-90 transition-opacity flex items-center justify-center gap-2 cursor-pointer">
                  <span className="material-symbols-outlined text-[18px]">psychology</span>
                  ANALYZE
                </button>
                <button 
                  onClick={clearAll}
                  className="py-stack-md px-stack-lg border border-outline-variant text-on-surface font-label-caps text-label-caps rounded-xl hover:bg-surface-variant/20 transition-colors cursor-pointer">
                  CANCEL
                </button>
              </div>
            ) : loading ? (
              <div className="flex items-center justify-center py-stack-lg">
                <div className="flex flex-col items-center gap-stack-md">
                  <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
                  <span className="font-label-caps text-label-caps text-on-surface-variant">ANALYZING...</span>
                </div>
              </div>
            ) : null}
          </section>
        )}
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 w-full z-50 flex justify-around items-center h-20 px-4 pb-safe bg-surface-container dark:bg-surface-container-low border-t border-outline-variant/30 shadow-sm">
        <button className="flex flex-col items-center justify-center text-on-surface-variant px-stack-lg py-1 hover:text-primary transition-colors" onClick={startCamera}>
          <span className="material-symbols-outlined">photo_camera</span>
          <span className="font-label-caps text-label-caps text-xs">Scan</span>
        </button>
        <button className="flex flex-col items-center justify-center text-on-secondary-fixed-variant px-stack-lg py-1 hover:bg-secondary-container/50">
          <span className="material-symbols-outlined">history</span>
          <span className="font-label-caps text-label-caps text-xs">History</span>
        </button>
        <button className="flex flex-col items-center justify-center text-on-secondary-fixed-variant px-stack-lg py-1 hover:bg-secondary-container/50">
          <span className="material-symbols-outlined">manage_accounts</span>
          <span className="font-label-caps text-label-caps text-xs">Settings</span>
        </button>
      </nav>
    </div>
  );
}

export default App;
