import { useState, useRef } from 'react'
import './ImageUpload.css'

function ImageUpload({ onUpload, loading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      alert('Vui lòng chọn file ảnh!');
      return;
    }

    setSelectedFile(file);

    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleVerify = () => {
    if (selectedFile) {
      onUpload(selectedFile);
    }
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };

  return (
    <div className="image-upload">
      <div className="upload-header">
        <h2 className="upload-title brand-font">Verify Media Intelligence</h2>
        <p className="upload-subtitle">Upload image to analyze deepfake traces using advanced neural networks.</p>
      </div>

      {!preview ? (
        <div
          className={`drop-zone ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
          <svg className="drop-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
          </svg>
          <p className="drop-text">
            Drag & drop image here or <span className="highlight">Browse</span>
          </p>
          <p className="drop-hint">Supports JPG, PNG (Max 10MB)</p>
        </div>
      ) : (
        <div className={`preview-section ${loading ? 'scanning' : ''}`}>
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <div className="scan-overlay"></div>
            {!loading && <button className="clear-btn" onClick={clearSelection}>×</button>}
          </div>

          <div className="file-info-bar">
            <span className="file-name">{selectedFile?.name}</span>
            <span className="file-size">
              {(selectedFile?.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>

          <button
            className="verify-btn"
            onClick={handleVerify}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Analyzing Media...
              </>
            ) : (
              <>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                Verify Authenticity
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
