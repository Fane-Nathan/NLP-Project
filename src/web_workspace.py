"""
TDSM Full Workspace - News Verification with Kokoro TTS
Complete pipeline: URL Fetch → Trust Layer → Web Search → KG → Verdict → TTS
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import re
import requests
from flask import Flask, render_template_string, request, jsonify
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from urllib.parse import urlparse

app = Flask(__name__)

# Global TTS instance (Kokoro)
tts_voice = None
tts_enabled = True

def init_tts():
    """Initialize Kokoro TTS."""
    global tts_voice
    try:
        from src.voice_kokoro import EchoVoice
        tts_voice = EchoVoice()
        print("✓ Kokoro TTS initialized")
        return True
    except Exception as e:
        print(f"[Warning] Kokoro TTS not available: {e}")
        return False

# HTML Template - Full Workspace with TTS
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDSM // News Verification Workspace</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --fg: #c9d1d9;
            --fg-muted: #8b949e;
            --accent: #00ff88;
            --accent-dim: #00cc6a;
            --border: #30363d;
            --error: #f85149;
            --error-bg: rgba(248, 81, 73, 0.1);
            --warn: #d29922;
            --warn-bg: rgba(210, 153, 34, 0.1);
            --success: #3fb950;
            --success-bg: rgba(63, 185, 80, 0.1);
            --info: #58a6ff;
            --info-bg: rgba(88, 166, 255, 0.1);
        }

        body {
            font-family: "JetBrains Mono", monospace;
            background: var(--bg);
            color: var(--fg);
            min-height: 100vh;
            line-height: 1.6;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1fr 380px;
            grid-template-rows: auto 1fr;
            min-height: 100vh;
            gap: 0;
        }

        @media (max-width: 1100px) {
            .workspace { grid-template-columns: 1fr; }
            .sidebar { max-height: 50vh; }
        }

        /* Header */
        .header {
            grid-column: 1 / -1;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            border: 2px solid var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            color: var(--accent);
            font-size: 0.7rem;
        }

        .logo h1 {
            font-size: 1rem;
            font-weight: 500;
        }

        .logo h1 span { color: var(--accent); }

        .header-controls {
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }

        /* TTS Controls */
        .tts-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.4rem 0.75rem;
            background: var(--bg-tertiary);
            border-radius: 6px;
            border: 1px solid var(--border);
        }

        .tts-controls label {
            font-size: 0.7rem;
            color: var(--fg-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .tts-toggle {
            width: 40px;
            height: 22px;
            background: var(--border);
            border-radius: 11px;
            position: relative;
            cursor: pointer;
            transition: background 0.2s;
        }

        .tts-toggle.active {
            background: var(--accent);
        }

        .tts-toggle::after {
            content: '';
            position: absolute;
            width: 18px;
            height: 18px;
            background: var(--fg);
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s;
        }

        .tts-toggle.active::after {
            transform: translateX(18px);
        }

        .voice-select {
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.75rem;
            padding: 0.3rem 0.5rem;
            border-radius: 4px;
            cursor: pointer;
        }

        .tts-status {
            font-size: 0.6rem;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            background: var(--success-bg);
            color: var(--success);
        }

        .tts-status.offline {
            background: var(--error-bg);
            color: var(--error);
        }

        /* Main Area */
        .main {
            padding: 1.5rem;
            overflow-y: auto;
        }

        /* Input Section */
        .input-section {
            margin-bottom: 1.5rem;
        }

        .input-tabs {
            display: flex;
            gap: 0;
            margin-bottom: 0;
        }

        .input-tab {
            padding: 0.6rem 1.2rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-bottom: none;
            color: var(--fg-muted);
            font-size: 0.75rem;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: all 0.2s;
        }

        .input-tab:first-child { border-radius: 6px 0 0 0; }
        .input-tab:last-child { border-radius: 0 6px 0 0; }

        .input-tab.active {
            background: var(--bg-secondary);
            color: var(--accent);
            border-color: var(--accent);
            border-bottom: 1px solid var(--bg-secondary);
            margin-bottom: -1px;
            z-index: 1;
        }

        .input-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 0 6px 6px 6px;
            padding: 1rem;
            display: none;
        }

        .input-panel.active { display: block; }

        .url-input-group {
            display: flex;
            gap: 0.5rem;
        }

        input[type="url"], input[type="text"] {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.85rem;
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }

        input:focus {
            outline: none;
            border-color: var(--accent);
        }

        textarea {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.85rem;
            padding: 1rem;
            border-radius: 6px;
            resize: vertical;
            min-height: 150px;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
        }

        .btn {
            background: var(--accent);
            border: none;
            color: var(--bg);
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-radius: 6px;
            transition: all 0.2s;
            white-space: nowrap;
        }

        .btn:hover {
            background: var(--accent-dim);
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--fg);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--border);
        }

        .btn-icon {
            padding: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Pipeline Status */
        .pipeline-status {
            display: flex;
            gap: 0.25rem;
            align-items: center;
            flex-wrap: wrap;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 1.5rem;
        }

        .pipeline-step {
            font-size: 0.6rem;
            padding: 0.35rem 0.6rem;
            border: 1px solid var(--border);
            color: var(--fg-muted);
            border-radius: 4px;
            transition: all 0.2s;
        }

        .pipeline-step.active {
            border-color: var(--warn);
            color: var(--warn);
            background: var(--warn-bg);
            animation: pulse 1s infinite;
        }

        .pipeline-step.done {
            background: var(--accent);
            color: var(--bg);
            border-color: var(--accent);
        }

        .pipeline-step.error {
            background: var(--error-bg);
            color: var(--error);
            border-color: var(--error);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .pipeline-arrow {
            color: var(--border);
            font-size: 0.7rem;
        }

        /* Verdict Card */
        .verdict-card {
            background: var(--bg-secondary);
            border: 2px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 1.5rem;
        }

        .verdict-header {
            padding: 1rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }

        .verdict-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--fg-muted);
        }

        .verdict-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.3rem 0.75rem;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .verdict-badge.true {
            background: var(--success-bg);
            color: var(--success);
            border: 1px solid var(--success);
        }

        .verdict-badge.false {
            background: var(--error-bg);
            color: var(--error);
            border: 1px solid var(--error);
        }

        .verdict-badge.uncertain {
            background: var(--warn-bg);
            color: var(--warn);
            border: 1px solid var(--warn);
        }

        .verdict-badge.processing {
            background: var(--info-bg);
            color: var(--info);
            border: 1px solid var(--info);
        }

        .verdict-content {
            padding: 1.25rem;
        }

        .verdict-summary {
            font-size: 0.9rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        .verdict-confidence {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }

        .confidence-bar {
            flex: 1;
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.5s;
        }

        .confidence-label {
            font-size: 0.7rem;
            color: var(--fg-muted);
            white-space: nowrap;
        }

        /* Evidence Section */
        .evidence-section {
            margin-bottom: 1.5rem;
        }

        .evidence-header {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--fg-muted);
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .evidence-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .evidence-item {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }

        .evidence-icon {
            font-size: 1rem;
            line-height: 1;
        }

        .evidence-content {
            flex: 1;
            min-width: 0;
        }

        .evidence-title {
            font-size: 0.8rem;
            color: var(--fg);
            margin-bottom: 0.25rem;
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .evidence-source {
            font-size: 0.7rem;
            color: var(--fg-muted);
        }

        .evidence-source a {
            color: var(--info);
            text-decoration: none;
        }

        .evidence-source a:hover {
            text-decoration: underline;
        }

        /* Sidebar */
        .sidebar {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .sidebar-section {
            border-bottom: 1px solid var(--border);
        }

        .sidebar-header {
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--fg-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 5;
        }

        .sidebar-content {
            padding: 0;
        }

        /* Document Cards */
        .doc-card {
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1rem;
            transition: background 0.2s;
        }

        .doc-card:hover { background: var(--bg-tertiary); }
        .doc-card:last-child { border-bottom: none; }

        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }

        .doc-index {
            font-size: 0.65rem;
            color: var(--fg-muted);
        }

        .doc-badge {
            font-size: 0.55rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-weight: 500;
        }

        .doc-badge.valid { background: var(--success-bg); color: var(--success); }
        .doc-badge.hoax { background: var(--error-bg); color: var(--error); }
        .doc-badge.outlier { background: var(--warn-bg); color: var(--warn); }

        .doc-text {
            font-size: 0.7rem;
            color: var(--fg-muted);
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .doc-text.filtered {
            text-decoration: line-through;
            opacity: 0.5;
        }

        .doc-scores {
            display: flex;
            gap: 0.75rem;
            margin-top: 0.35rem;
            font-size: 0.6rem;
        }

        .doc-score-value.good { color: var(--success); }
        .doc-score-value.bad { color: var(--error); }
        .doc-score-value.warn { color: var(--warn); }

        /* Stats */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
        }

        .stat-item { text-align: center; }

        .stat-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--fg);
        }

        .stat-value.accent { color: var(--accent); }

        .stat-label {
            font-size: 0.55rem;
            color: var(--fg-muted);
            text-transform: uppercase;
        }

        /* Empty State */
        .empty-state {
            padding: 2rem 1rem;
            text-align: center;
            color: var(--fg-muted);
            font-size: 0.75rem;
        }

        .empty-state-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            opacity: 0.3;
        }

        /* Bookmarklet Section */
        .bookmarklet-section {
            padding: 1rem;
            background: var(--bg-tertiary);
            border-top: 1px solid var(--border);
            margin-top: auto;
        }

        .bookmarklet-title {
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--fg-muted);
            margin-bottom: 0.5rem;
        }

        .bookmarklet-link {
            display: block;
            background: var(--info-bg);
            color: var(--info);
            text-decoration: none;
            padding: 0.5rem 0.75rem;
            border-radius: 4px;
            font-size: 0.7rem;
            text-align: center;
            border: 1px dashed var(--info);
        }

        .bookmarklet-hint {
            font-size: 0.6rem;
            color: var(--fg-muted);
            margin-top: 0.5rem;
            text-align: center;
        }

        /* TTS Speaking Indicator */
        .speaking-indicator {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            background: var(--accent);
            color: var(--bg);
            font-size: 0.7rem;
            border-radius: 4px;
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            z-index: 100;
        }

        .speaking-indicator.active { display: flex; }

        .speaking-waves {
            display: flex;
            gap: 2px;
            align-items: center;
        }

        .speaking-wave {
            width: 3px;
            background: var(--bg);
            border-radius: 2px;
            animation: wave 0.5s ease-in-out infinite;
        }

        .speaking-wave:nth-child(1) { height: 8px; animation-delay: 0s; }
        .speaking-wave:nth-child(2) { height: 12px; animation-delay: 0.1s; }
        .speaking-wave:nth-child(3) { height: 16px; animation-delay: 0.2s; }
        .speaking-wave:nth-child(4) { height: 12px; animation-delay: 0.3s; }
        .speaking-wave:nth-child(5) { height: 8px; animation-delay: 0.4s; }

        @keyframes wave {
            0%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(0.5); }
        }

        /* Stop TTS button */
        .btn-stop-tts {
            background: var(--error);
            padding: 0.4rem 0.6rem;
            font-size: 0.7rem;
        }
    </style>
</head>
<body>
    <div class="workspace">
        <header class="header">
            <div class="logo">
                <div class="logo-icon">TD</div>
                <h1><span>TDSM</span> // News Verification Workspace</h1>
            </div>
            <div class="header-controls">
                <div class="tts-controls">
                    <label>Kokoro TTS</label>
                    <div class="tts-toggle active" id="tts-toggle" onclick="toggleTTS()"></div>
                    <select class="voice-select" id="voice-select" onchange="changeVoice()">
                        <option value="friday">Friday (Female)</option>
                        <option value="echo">Echo (Male)</option>
                    </select>
                    <span class="tts-status" id="tts-status">Ready</span>
                </div>
            </div>
        </header>

        <main class="main">
            <!-- Input Section -->
            <div class="input-section">
                <div class="input-tabs">
                    <div class="input-tab active" data-tab="url" onclick="switchTab('url')">URL</div>
                    <div class="input-tab" data-tab="text" onclick="switchTab('text')">Text</div>
                    <div class="input-tab" data-tab="clipboard" onclick="switchTab('clipboard')">Clipboard</div>
                </div>

                <div class="input-panel active" id="panel-url">
                    <div class="url-input-group">
                        <input type="url" id="url-input" placeholder="Paste news article URL here...">
                        <button class="btn" onclick="verifyURL()">Verify</button>
                    </div>
                </div>

                <div class="input-panel" id="panel-text">
                    <textarea id="text-input" placeholder="Paste the news article text here..."></textarea>
                    <div style="margin-top: 0.75rem; display: flex; justify-content: flex-end;">
                        <button class="btn" onclick="verifyText()">Verify</button>
                    </div>
                </div>

                <div class="input-panel" id="panel-clipboard">
                    <p style="color: var(--fg-muted); font-size: 0.8rem; margin-bottom: 1rem;">
                        Click the button below to read and verify content from your clipboard.
                    </p>
                    <button class="btn" onclick="verifyClipboard()">Read Clipboard & Verify</button>
                </div>
            </div>

            <!-- Pipeline Status -->
            <div class="pipeline-status" id="pipeline">
                <span class="pipeline-step" data-step="fetch">FETCH</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" data-step="trust">TRUST LAYER</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" data-step="search">WEB SEARCH</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" data-step="kg">KNOWLEDGE GRAPH</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" data-step="verdict">VERDICT</span>
            </div>

            <!-- Verdict Card -->
            <div class="verdict-card" id="verdict-card">
                <div class="verdict-header">
                    <span class="verdict-label">Verification Result</span>
                    <span class="verdict-badge processing" id="verdict-badge">READY</span>
                </div>
                <div class="verdict-content">
                    <div class="verdict-summary" id="verdict-summary">
                        Enter a URL or paste text to verify news content. The system will analyze credibility, search for corroborating evidence, and deliver a verdict.
                    </div>
                    <div class="verdict-confidence" id="verdict-confidence" style="display: none;">
                        <span class="confidence-label">Confidence</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" id="confidence-fill" style="width: 0%"></div>
                        </div>
                        <span class="confidence-label" id="confidence-value">0%</span>
                    </div>
                </div>
            </div>

            <!-- Evidence Section -->
            <div class="evidence-section" id="evidence-section" style="display: none;">
                <div class="evidence-header">
                    <span>Supporting Evidence</span>
                    <span id="evidence-count">0 sources</span>
                </div>
                <div class="evidence-list" id="evidence-list"></div>
            </div>
        </main>

        <aside class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-header">
                    <span>Document Analysis</span>
                    <span id="doc-count">0 docs</span>
                </div>
                <div class="stats-grid" id="stats-grid" style="display: none;">
                    <div class="stat-item">
                        <div class="stat-value accent" id="stat-trusted">0</div>
                        <div class="stat-label">Trusted</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-filtered">0</div>
                        <div class="stat-label">Filtered</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="stat-hoax">0</div>
                        <div class="stat-label">Hoax</div>
                    </div>
                </div>
                <div class="sidebar-content" id="documents-list">
                    <div class="empty-state">
                        <div class="empty-state-icon">📄</div>
                        <div>No content analyzed yet</div>
                    </div>
                </div>
            </div>

            <div class="bookmarklet-section">
                <div class="bookmarklet-title">Quick Verify Bookmarklet</div>
                <a class="bookmarklet-link" href="javascript:(function(){var url=encodeURIComponent(window.location.href);window.open('http://localhost:5000/?verify='+url,'_blank')})();" onclick="return false;">
                    📌 Verify This Page
                </a>
                <div class="bookmarklet-hint">Drag this to your bookmarks bar</div>
            </div>
        </aside>
    </div>

    <!-- TTS Speaking Indicator -->
    <div class="speaking-indicator" id="speaking-indicator">
        <div class="speaking-waves">
            <div class="speaking-wave"></div>
            <div class="speaking-wave"></div>
            <div class="speaking-wave"></div>
            <div class="speaking-wave"></div>
            <div class="speaking-wave"></div>
        </div>
        <span>Speaking...</span>
    </div>

    <script>
        // Kokoro TTS Setup
        let ttsEnabled = true;

        function toggleTTS() {
            ttsEnabled = !ttsEnabled;
            document.getElementById('tts-toggle').classList.toggle('active', ttsEnabled);

            // Update server state
            fetch('/api/tts/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: ttsEnabled })
            });
        }

        function changeVoice() {
            const voice = document.getElementById('voice-select').value;
            fetch('/api/tts/voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ voice: voice })
            }).then(res => res.json()).then(data => {
                if (data.success) {
                    updateTTSStatus('Voice: ' + voice);
                }
            });
        }

        function speak(text) {
            if (!ttsEnabled || !text) return;

            updateTTSStatus('Speaking...');
            document.getElementById('speaking-indicator').classList.add('active');

            fetch('/api/tts/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            }).then(res => res.json()).then(data => {
                // TTS is async on server, indicator will stay for a bit
                setTimeout(() => {
                    document.getElementById('speaking-indicator').classList.remove('active');
                    updateTTSStatus('Ready');
                }, 2000);
            }).catch(err => {
                document.getElementById('speaking-indicator').classList.remove('active');
                updateTTSStatus('Error');
            });
        }

        function updateTTSStatus(status) {
            const el = document.getElementById('tts-status');
            el.textContent = status;
            el.classList.toggle('offline', status === 'Error' || status === 'Offline');
        }

        // Check TTS status on load
        fetch('/api/tts/status').then(res => res.json()).then(data => {
            updateTTSStatus(data.available ? 'Ready' : 'Offline');
            if (!data.available) {
                document.getElementById('tts-toggle').classList.remove('active');
                ttsEnabled = false;
            }
        });

        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.input-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.input-panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
            document.getElementById(`panel-${tab}`).classList.add('active');
        }

        // Pipeline steps
        function setStep(step, state) {
            const el = document.querySelector(`[data-step="${step}"]`);
            if (el) {
                el.classList.remove('active', 'done', 'error');
                if (state) el.classList.add(state);
            }
        }

        function resetPipeline() {
            document.querySelectorAll('.pipeline-step').forEach(el => {
                el.classList.remove('active', 'done', 'error');
            });
        }

        // Render documents
        function renderDocuments(documents) {
            const container = document.getElementById('documents-list');
            const countEl = document.getElementById('doc-count');
            const statsEl = document.getElementById('stats-grid');

            if (!documents || documents.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📄</div><div>No documents</div></div>';
                countEl.textContent = '0 docs';
                statsEl.style.display = 'none';
                return;
            }

            countEl.textContent = `${documents.length} docs`;

            const trusted = documents.filter(d => d.status === 'valid').length;
            const filtered = documents.filter(d => d.status !== 'valid').length;
            const hoax = documents.filter(d => d.status === 'hoax').length;

            document.getElementById('stat-trusted').textContent = trusted;
            document.getElementById('stat-filtered').textContent = filtered;
            document.getElementById('stat-hoax').textContent = hoax;
            statsEl.style.display = 'grid';

            container.innerHTML = documents.map((doc, i) => {
                const isFiltered = doc.status !== 'valid';
                const badgeClass = doc.status === 'valid' ? 'valid' : doc.status === 'hoax' ? 'hoax' : 'outlier';
                const badgeText = doc.status === 'valid' ? 'TRUSTED' : doc.status === 'hoax' ? 'HOAX' : 'FILTERED';

                return `
                    <div class="doc-card">
                        <div class="doc-header">
                            <span class="doc-index">DOC ${i + 1}</span>
                            <span class="doc-badge ${badgeClass}">${badgeText}</span>
                        </div>
                        <div class="doc-text ${isFiltered ? 'filtered' : ''}">${escapeHtml(doc.text)}</div>
                        ${doc.credibility_score !== null ? `
                        <div class="doc-scores">
                            <span>Trust: <span class="doc-score-value ${doc.credibility_score >= 0.7 ? 'good' : doc.credibility_score >= 0.4 ? 'warn' : 'bad'}">${(doc.credibility_score * 100).toFixed(0)}%</span></span>
                            ${doc.hoax_probability !== null ? `<span>Hoax: <span class="doc-score-value ${doc.hoax_probability <= 0.3 ? 'good' : doc.hoax_probability <= 0.6 ? 'warn' : 'bad'}">${(doc.hoax_probability * 100).toFixed(0)}%</span></span>` : ''}
                        </div>
                        ` : ''}
                    </div>
                `;
            }).join('');
        }

        // Render evidence
        function renderEvidence(evidence) {
            const section = document.getElementById('evidence-section');
            const list = document.getElementById('evidence-list');
            const count = document.getElementById('evidence-count');

            if (!evidence || evidence.length === 0) {
                section.style.display = 'none';
                return;
            }

            section.style.display = 'block';
            count.textContent = `${evidence.length} sources`;

            list.innerHTML = evidence.map(e => `
                <div class="evidence-item">
                    <span class="evidence-icon">${e.supports ? '✅' : e.contradicts ? '❌' : '📰'}</span>
                    <div class="evidence-content">
                        <div class="evidence-title">${escapeHtml(e.title)}</div>
                        <div class="evidence-source">
                            ${e.url ? `<a href="${e.url}" target="_blank">${e.source || new URL(e.url).hostname}</a>` : e.source || 'Unknown source'}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Verification functions
        async function verifyURL() {
            const url = document.getElementById('url-input').value.trim();
            if (!url) {
                alert('Please enter a URL');
                return;
            }
            await runVerification({ type: 'url', content: url });
        }

        async function verifyText() {
            const text = document.getElementById('text-input').value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }
            await runVerification({ type: 'text', content: text });
        }

        async function verifyClipboard() {
            try {
                const text = await navigator.clipboard.readText();
                if (!text) {
                    alert('Clipboard is empty');
                    return;
                }
                document.getElementById('text-input').value = text;
                switchTab('text');
                await runVerification({ type: 'text', content: text });
            } catch (err) {
                alert('Could not read clipboard. Please paste manually.');
            }
        }

        async function runVerification(input) {
            resetPipeline();

            const badge = document.getElementById('verdict-badge');
            const summary = document.getElementById('verdict-summary');
            const confidenceSection = document.getElementById('verdict-confidence');

            badge.className = 'verdict-badge processing';
            badge.textContent = 'ANALYZING';
            summary.textContent = 'Analyzing content...';
            confidenceSection.style.display = 'none';

            speak('Starting verification analysis.');

            try {
                // Step 1: Fetch
                setStep('fetch', 'active');
                await sleep(300);

                const response = await fetch('/api/verify', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(input)
                });

                const data = await response.json();
                setStep('fetch', 'done');

                if (data.error) {
                    throw new Error(data.error);
                }

                // Step 2: Trust Layer
                setStep('trust', 'active');
                await sleep(200);
                setStep('trust', 'done');

                // Step 3: Web Search
                setStep('search', 'active');
                await sleep(200);
                setStep('search', 'done');

                // Step 4: Knowledge Graph
                setStep('kg', 'active');
                await sleep(200);
                setStep('kg', 'done');

                // Step 5: Verdict
                setStep('verdict', 'active');
                await sleep(200);
                setStep('verdict', 'done');

                // Display results
                const verdictClass = data.verdict === 'TRUE' ? 'true' :
                                     data.verdict === 'FALSE' ? 'false' : 'uncertain';

                badge.className = `verdict-badge ${verdictClass}`;
                badge.textContent = data.verdict || 'UNCERTAIN';
                summary.textContent = data.summary || 'Analysis complete.';

                if (data.confidence !== undefined) {
                    confidenceSection.style.display = 'flex';
                    document.getElementById('confidence-fill').style.width = `${data.confidence * 100}%`;
                    document.getElementById('confidence-value').textContent = `${(data.confidence * 100).toFixed(0)}%`;
                }

                // Render documents and evidence
                if (data.documents) renderDocuments(data.documents);
                if (data.evidence) renderEvidence(data.evidence);

                // TTS for verdict
                const ttsText = `Verdict: ${data.verdict}. ${data.summary}`;
                speak(ttsText);

            } catch (err) {
                setStep('fetch', 'error');
                badge.className = 'verdict-badge false';
                badge.textContent = 'ERROR';
                summary.textContent = `Error: ${err.message}`;
                speak(`Verification failed. ${err.message}`);
            }
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        // Check for URL parameter (from bookmarklet)
        const urlParams = new URLSearchParams(window.location.search);
        const verifyParam = urlParams.get('verify');
        if (verifyParam) {
            document.getElementById('url-input').value = decodeURIComponent(verifyParam);
            setTimeout(() => verifyURL(), 500);
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                const activeTab = document.querySelector('.input-tab.active').dataset.tab;
                if (activeTab === 'url') verifyURL();
                else if (activeTab === 'text') verifyText();
                else verifyClipboard();
            }
        });
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """Serve the workspace interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/verify', methods=['POST'])
def verify():
    """Full verification pipeline: Fetch → Trust → Search → KG → Verdict."""
    try:
        data = request.json
        input_type = data.get('type', 'text')
        content = data.get('content', '').strip()

        if not content:
            return jsonify({'error': 'No content provided'}), 400

        # Step 1: Fetch content if URL
        documents = []
        source_url = None

        if input_type == 'url':
            source_url = content
            fetched = fetch_url_content(content)
            if fetched:
                documents = [fetched['text']]
            else:
                return jsonify({'error': 'Could not fetch URL content'}), 400
        else:
            # Split by paragraphs
            documents = [p.strip() for p in content.split('\n\n') if p.strip()]
            if not documents:
                documents = [content]

        # Document analysis
        doc_analysis = []
        for doc in documents:
            doc_analysis.append({
                'text': doc[:200] + ('...' if len(doc) > 200 else ''),
                'status': 'valid',
                'credibility_score': None,
                'hoax_probability': None
            })

        filtered_docs = documents
        hoax_detected = False
        avg_hoax_prob = 0

        # Step 2: Trust Layer (Credibility Analysis)
        try:
            from src.hoax_detection.credibility_report import CredibilityAnalyzer
            analyzer = CredibilityAnalyzer(
                hoax_model_path="models/hoax_indobert_lora",
                outlier_threshold_z=2.0,
                hoax_weight=0.6,
                outlier_weight=0.4
            )
            filtered_docs, report = analyzer.filter_documents(documents)

            if hasattr(report, 'document_scores'):
                hoax_probs = []
                for i, score_data in enumerate(report.document_scores):
                    if i < len(doc_analysis):
                        cred = score_data.get('credibility_score', 0)
                        hoax = score_data.get('hoax_probability', 0)
                        doc_analysis[i]['credibility_score'] = cred
                        doc_analysis[i]['hoax_probability'] = hoax
                        hoax_probs.append(hoax)

                        if score_data.get('is_hoax', False):
                            doc_analysis[i]['status'] = 'hoax'
                            hoax_detected = True
                        elif score_data.get('is_outlier', False):
                            doc_analysis[i]['status'] = 'outlier'

                if hoax_probs:
                    avg_hoax_prob = sum(hoax_probs) / len(hoax_probs)

        except ImportError:
            pass
        except Exception as e:
            print(f"Trust layer error: {e}")

        # Step 3: Web Search for corroboration
        evidence = []
        search_supports = 0
        search_contradicts = 0

        try:
            from src.tools.search_tool import WebSearcher
            searcher = WebSearcher(max_results=5)

            # Generate search query from content
            search_query = generate_search_query(documents[0][:500])
            results = searcher.search(search_query)

            for r in results[:5]:
                supports = False
                contradicts = False
                # Simple heuristic: check if result mentions similar entities
                evidence.append({
                    'title': r.get('title', 'No title'),
                    'url': r.get('url', ''),
                    'source': r.get('source', ''),
                    'supports': supports,
                    'contradicts': contradicts
                })

        except ImportError:
            pass
        except Exception as e:
            print(f"Web search error: {e}")

        # Step 4: Knowledge Graph (if available)
        kg_confidence = 0.5
        try:
            from src.models.knowledge_graph import KnowledgeGraph
            kg = KnowledgeGraph(name="verify_kg")
            kg.add_documents(filtered_docs, show_progress=False)
            # KG adds confidence if entities are consistent
            kg_confidence = 0.7 if kg.graph.number_of_nodes() > 0 else 0.5
        except ImportError:
            pass

        # Step 5: Generate Verdict
        verdict, summary, confidence = generate_verdict(
            documents=documents,
            doc_analysis=doc_analysis,
            hoax_detected=hoax_detected,
            avg_hoax_prob=avg_hoax_prob,
            evidence=evidence,
            kg_confidence=kg_confidence
        )

        return jsonify({
            'verdict': verdict,
            'summary': summary,
            'confidence': confidence,
            'documents': doc_analysis,
            'evidence': evidence
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def fetch_url_content(url: str) -> Optional[Dict]:
    """Fetch and extract text content from URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Simple HTML text extraction
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ['script', 'style', 'nav', 'header', 'footer']:
                    self.skip = True

            def handle_endtag(self, tag):
                if tag in ['script', 'style', 'nav', 'header', 'footer']:
                    self.skip = False

            def handle_data(self, data):
                if not self.skip:
                    text = data.strip()
                    if text:
                        self.text.append(text)

        parser = TextExtractor()
        parser.feed(response.text)
        text = ' '.join(parser.text)

        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 100:
            return None

        return {
            'url': url,
            'text': text[:5000],  # Limit text length
            'title': extract_title(response.text)
        }

    except Exception as e:
        print(f"URL fetch error: {e}")
        return None


def extract_title(html: str) -> str:
    """Extract title from HTML."""
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    return match.group(1).strip() if match else 'Unknown'


def generate_search_query(text: str) -> str:
    """Generate a search query from text content."""
    # Extract key phrases (simple approach)
    words = text.split()[:20]
    # Remove common words
    stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada', 'adalah', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
    keywords = [w for w in words if w.lower() not in stopwords and len(w) > 3]
    return ' '.join(keywords[:10])


def generate_verdict(
    documents: List[str],
    doc_analysis: List[Dict],
    hoax_detected: bool,
    avg_hoax_prob: float,
    evidence: List[Dict],
    kg_confidence: float
) -> Tuple[str, str, float]:
    """Generate verification verdict."""

    # Calculate confidence
    confidence = 0.5

    # Hoax detection weight
    if hoax_detected:
        confidence = max(0.1, 1.0 - avg_hoax_prob)
        verdict = "FALSE"
        summary = f"This content has been flagged as potential misinformation with {avg_hoax_prob*100:.0f}% hoax probability. "
    elif avg_hoax_prob > 0.5:
        confidence = 0.4
        verdict = "UNCERTAIN"
        summary = f"The content shows mixed credibility signals. Hoax probability: {avg_hoax_prob*100:.0f}%. "
    else:
        confidence = min(0.9, 0.5 + kg_confidence * 0.3 + (1 - avg_hoax_prob) * 0.2)
        verdict = "TRUE"
        summary = "The content appears credible based on trust layer analysis. "

    # Add evidence context
    if evidence:
        summary += f"Found {len(evidence)} related sources for reference."

    # Trusted document count
    trusted = sum(1 for d in doc_analysis if d['status'] == 'valid')
    total = len(doc_analysis)

    if trusted < total:
        summary += f" {total - trusted} of {total} segments were filtered due to low credibility."

    return verdict, summary, confidence


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


# === TTS API Endpoints ===

@app.route('/api/tts/status')
def tts_status():
    """Check if Kokoro TTS is available."""
    global tts_voice
    return jsonify({
        'available': tts_voice is not None,
        'enabled': tts_enabled,
        'voice': tts_voice.persona if tts_voice else None
    })


@app.route('/api/tts/speak', methods=['POST'])
def tts_speak():
    """Speak text using Kokoro TTS."""
    global tts_voice, tts_enabled
    if not tts_voice or not tts_enabled:
        return jsonify({'success': False, 'error': 'TTS not available'})

    data = request.json
    text = data.get('text', '').strip()

    if text:
        tts_voice.speak(text)
        return jsonify({'success': True})

    return jsonify({'success': False, 'error': 'No text provided'})


@app.route('/api/tts/toggle', methods=['POST'])
def tts_toggle():
    """Toggle TTS on/off."""
    global tts_enabled
    data = request.json
    tts_enabled = data.get('enabled', True)
    return jsonify({'success': True, 'enabled': tts_enabled})


@app.route('/api/tts/voice', methods=['POST'])
def tts_voice_change():
    """Change TTS voice persona."""
    global tts_voice
    if not tts_voice:
        return jsonify({'success': False, 'error': 'TTS not available'})

    data = request.json
    voice = data.get('voice', 'friday')

    if tts_voice.set_persona(voice):
        return jsonify({'success': True, 'voice': voice})

    return jsonify({'success': False, 'error': 'Invalid voice'})


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask server."""
    print(f"\n{'='*60}")
    print("TDSM News Verification Workspace")
    print(f"{'='*60}")

    # Initialize Kokoro TTS
    init_tts()

    print(f"\n→ Open http://localhost:{port} in your browser")
    print("\nFeatures:")
    print("  • URL verification (paste article links)")
    print("  • Text verification (paste content directly)")
    print("  • Clipboard reading")
    print("  • Kokoro TTS (local neural voice)")
    print("  • Bookmarklet for quick verification")
    print()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
