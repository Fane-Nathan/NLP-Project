"""
Minimalist Web Playground for NLP Pipeline
Terminal-style brutalist interface with Trust Layer visualization
"""

import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import base64
from flask import Flask, render_template_string, request, jsonify
from typing import Optional, List, Dict

app = Flask(__name__)

# HTML Template - Enhanced Brutalist Design
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDSM // Playground</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

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
        }

        body {
            font-family: "JetBrains Mono", "SF Mono", "Fira Code", monospace;
            background: var(--bg);
            color: var(--fg);
            min-height: 100vh;
            line-height: 1.6;
        }

        .app {
            display: grid;
            grid-template-columns: 1fr 400px;
            min-height: 100vh;
        }

        @media (max-width: 1200px) {
            .app {
                grid-template-columns: 1fr;
            }
            .sidebar {
                order: -1;
                max-height: 400px;
            }
        }

        /* Main Panel */
        .main {
            padding: 2rem;
            border-right: 1px solid var(--border);
            overflow-y: auto;
        }

        /* Sidebar - Documents & Trust Layer */
        .sidebar {
            background: var(--bg-secondary);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .sidebar-section {
            border-bottom: 1px solid var(--border);
        }

        .sidebar-header {
            padding: 1rem 1.25rem;
            background: var(--bg-tertiary);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--fg-muted);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .sidebar-content {
            padding: 0;
        }

        /* Header */
        header {
            margin-bottom: 2rem;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.5rem;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            border: 2px solid var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            color: var(--accent);
            font-size: 0.8rem;
        }

        h1 {
            font-size: 1.1rem;
            font-weight: 500;
            letter-spacing: 0.05em;
        }

        h1 span {
            color: var(--accent);
        }

        .subtitle {
            color: var(--fg-muted);
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }

        /* Sections */
        .section {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            color: var(--fg-muted);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }

        /* Input */
        textarea {
            width: 100%;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.85rem;
            padding: 1rem;
            resize: vertical;
            min-height: 120px;
            border-radius: 6px;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(0, 255, 136, 0.1);
        }

        textarea::placeholder {
            color: var(--fg-muted);
        }

        /* Pipeline */
        .pipeline {
            display: flex;
            gap: 0.25rem;
            align-items: center;
            flex-wrap: wrap;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 6px;
            border: 1px solid var(--border);
        }

        .pipeline-step {
            font-size: 0.65rem;
            padding: 0.4rem 0.75rem;
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
            font-size: 0.8rem;
        }

        /* Controls */
        .controls {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
            margin: 1.5rem 0;
        }

        select {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.8rem;
            padding: 0.6rem 1rem;
            cursor: pointer;
            border-radius: 6px;
        }

        select:focus {
            outline: none;
            border-color: var(--accent);
        }

        select option {
            background: var(--bg-secondary);
        }

        button {
            background: var(--accent);
            border: none;
            color: var(--bg);
            font-family: inherit;
            font-size: 0.8rem;
            font-weight: 500;
            padding: 0.6rem 1.5rem;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-radius: 6px;
            transition: all 0.2s;
        }

        button:hover {
            background: var(--accent-dim);
            transform: translateY(-1px);
        }

        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }

        .checkbox-group {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-size: 0.8rem;
            color: var(--fg-muted);
        }

        .checkbox-item:hover {
            color: var(--fg);
        }

        .checkbox-item input {
            accent-color: var(--accent);
            width: 16px;
            height: 16px;
        }

        /* Output */
        .output-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border);
        }

        .output-header label {
            margin: 0;
        }

        .output {
            padding: 1rem;
            min-height: 150px;
            white-space: pre-wrap;
            font-size: 0.85rem;
            line-height: 1.7;
        }

        .output.loading::after {
            content: "█";
            animation: blink 0.8s infinite;
            color: var(--accent);
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        .status {
            font-size: 0.65rem;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-weight: 500;
            letter-spacing: 0.05em;
        }

        .status.success {
            color: var(--success);
            background: var(--success-bg);
        }

        .status.error {
            color: var(--error);
            background: var(--error-bg);
        }

        .status.processing {
            color: var(--warn);
            background: var(--warn-bg);
        }

        /* Metrics */
        .metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .metric {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent);
        }

        .metric-value.error {
            color: var(--error);
        }

        .metric-label {
            font-size: 0.65rem;
            color: var(--fg-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-top: 0.25rem;
        }

        /* Document Cards */
        .doc-card {
            border-bottom: 1px solid var(--border);
            padding: 1rem 1.25rem;
            transition: background 0.2s;
        }

        .doc-card:hover {
            background: var(--bg-tertiary);
        }

        .doc-card:last-child {
            border-bottom: none;
        }

        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
            gap: 0.5rem;
        }

        .doc-index {
            font-size: 0.7rem;
            color: var(--fg-muted);
            font-weight: 500;
        }

        .doc-badge {
            font-size: 0.6rem;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-weight: 500;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }

        .doc-badge.valid {
            background: var(--success-bg);
            color: var(--success);
        }

        .doc-badge.hoax {
            background: var(--error-bg);
            color: var(--error);
        }

        .doc-badge.outlier {
            background: var(--warn-bg);
            color: var(--warn);
        }

        .doc-badge.filtered {
            background: var(--bg-tertiary);
            color: var(--fg-muted);
        }

        .doc-text {
            font-size: 0.75rem;
            color: var(--fg-muted);
            line-height: 1.5;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .doc-text.filtered {
            text-decoration: line-through;
            opacity: 0.5;
        }

        .doc-scores {
            display: flex;
            gap: 1rem;
            margin-top: 0.5rem;
            font-size: 0.65rem;
        }

        .doc-score {
            display: flex;
            align-items: center;
            gap: 0.3rem;
        }

        .doc-score-label {
            color: var(--fg-muted);
        }

        .doc-score-value {
            font-weight: 500;
        }

        .doc-score-value.good {
            color: var(--success);
        }

        .doc-score-value.bad {
            color: var(--error);
        }

        .doc-score-value.warn {
            color: var(--warn);
        }

        /* Progress Bar */
        .progress-bar {
            width: 100%;
            height: 3px;
            background: var(--border);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.3s;
        }

        .progress-fill.low {
            background: var(--error);
        }

        .progress-fill.medium {
            background: var(--warn);
        }

        /* Empty State */
        .empty-state {
            padding: 2rem;
            text-align: center;
            color: var(--fg-muted);
            font-size: 0.8rem;
        }

        .empty-state-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            opacity: 0.3;
        }

        /* Footer */
        footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--fg-muted);
            font-size: 0.65rem;
            display: flex;
            justify-content: space-between;
        }

        /* Stats Summary */
        .stats-summary {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            padding: 1rem 1.25rem;
            background: var(--bg-tertiary);
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--fg);
        }

        .stat-value.accent {
            color: var(--accent);
        }

        .stat-label {
            font-size: 0.6rem;
            color: var(--fg-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Keyboard hint */
        .kbd {
            font-size: 0.65rem;
            padding: 0.2rem 0.4rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 3px;
            color: var(--fg-muted);
        }
    </style>
</head>
<body>
    <div class="app">
        <main class="main">
            <header>
                <div class="logo">
                    <div class="logo-icon">TD</div>
                    <div>
                        <h1><span>TDSM</span> // Trust-Driven Summarization</h1>
                        <p class="subtitle">Indonesian Multi-Document Summarization with Knowledge Graph Verification</p>
                    </div>
                </div>
            </header>

            <div class="section">
                <label>Input Documents <span style="color: var(--fg-muted); text-transform: none;">(separate with blank lines)</span></label>
                <textarea id="input" placeholder="Paste your Indonesian news articles here...

Separate each document with a blank line.

Or leave empty to use demo documents about Indonesian vaccination program."></textarea>
            </div>

            <div class="section">
                <label>Pipeline</label>
                <div class="pipeline" id="pipeline">
                    <span class="pipeline-step" data-step="input">INPUT</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step" data-step="credibility">TRUST LAYER</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step" data-step="kg">KNOWLEDGE GRAPH</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step" data-step="summarize">SUMMARIZE</span>
                    <span class="pipeline-arrow">→</span>
                    <span class="pipeline-step" data-step="verify">VERIFY</span>
                </div>
            </div>

            <div class="controls">
                <select id="model">
                    <option value="hybrid">Hybrid</option>
                    <option value="textrank">TextRank</option>
                    <option value="lexrank">LexRank</option>
                    <option value="gemini">Gemini LLM</option>
                </select>

                <div class="checkbox-group">
                    <label class="checkbox-item">
                        <input type="checkbox" id="credibility" checked>
                        <span>Trust Layer</span>
                    </label>
                    <label class="checkbox-item">
                        <input type="checkbox" id="verify" checked>
                        <span>KG Verification</span>
                    </label>
                </div>

                <button id="run-btn" onclick="runPipeline()">Execute</button>
                <span class="kbd">Ctrl+Enter</span>
            </div>

            <div class="section">
                <div class="output-container">
                    <div class="output-header">
                        <label>Summary Output</label>
                        <span class="status" id="status" style="display:none;"></span>
                    </div>
                    <div class="output" id="output">Ready. Press Execute to run the pipeline.</div>
                </div>
            </div>

            <div class="metrics" id="metrics" style="display:none;">
                <div class="metric">
                    <div class="metric-value" id="metric-confidence">-</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-docs">-</div>
                    <div class="metric-label">Input Docs</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-filtered">-</div>
                    <div class="metric-label">Trusted</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-hallucination">-</div>
                    <div class="metric-label">Verified</div>
                </div>
            </div>

            <footer>
                <span>TDSM v1.0 // Knowledge Graph Verification Pipeline</span>
                <span id="timestamp"></span>
            </footer>
        </main>

        <aside class="sidebar">
            <div class="sidebar-section">
                <div class="sidebar-header">
                    <span>Trust Layer Analysis</span>
                    <span id="doc-count">0 docs</span>
                </div>
                <div class="stats-summary" id="stats-summary" style="display:none;">
                    <div class="stat-item">
                        <div class="stat-value accent" id="stat-valid">0</div>
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
                        <div>No documents analyzed yet</div>
                        <div style="margin-top: 0.5rem; font-size: 0.7rem;">Execute the pipeline to see trust analysis</div>
                    </div>
                </div>
            </div>
        </aside>
    </div>

    <script>
        // Update timestamp
        function updateTime() {
            document.getElementById('timestamp').textContent = new Date().toISOString().slice(0,19).replace('T', ' ');
        }
        updateTime();
        setInterval(updateTime, 1000);

        // Pipeline step management
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

        function renderDocuments(documents) {
            const container = document.getElementById('documents-list');
            const countEl = document.getElementById('doc-count');
            const statsEl = document.getElementById('stats-summary');

            if (!documents || documents.length === 0) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">📄</div>
                        <div>No documents to display</div>
                    </div>
                `;
                countEl.textContent = '0 docs';
                statsEl.style.display = 'none';
                return;
            }

            countEl.textContent = `${documents.length} docs`;

            // Calculate stats
            const valid = documents.filter(d => d.status === 'valid').length;
            const filtered = documents.filter(d => d.status !== 'valid').length;
            const hoax = documents.filter(d => d.status === 'hoax').length;

            document.getElementById('stat-valid').textContent = valid;
            document.getElementById('stat-filtered').textContent = filtered;
            document.getElementById('stat-hoax').textContent = hoax;
            statsEl.style.display = 'grid';

            container.innerHTML = documents.map((doc, i) => {
                const isFiltered = doc.status !== 'valid';
                const badgeClass = doc.status === 'valid' ? 'valid' :
                                   doc.status === 'hoax' ? 'hoax' :
                                   doc.status === 'outlier' ? 'outlier' : 'filtered';
                const badgeText = doc.status === 'valid' ? 'TRUSTED' :
                                  doc.status === 'hoax' ? 'HOAX' :
                                  doc.status === 'outlier' ? 'OUTLIER' : 'FILTERED';

                const credScore = doc.credibility_score !== undefined ? doc.credibility_score : null;
                const hoaxProb = doc.hoax_probability !== undefined ? doc.hoax_probability : null;

                const scoreClass = (score) => {
                    if (score === null) return '';
                    if (score >= 0.7) return 'good';
                    if (score >= 0.4) return 'warn';
                    return 'bad';
                };

                const hoaxClass = (prob) => {
                    if (prob === null) return '';
                    if (prob <= 0.3) return 'good';
                    if (prob <= 0.6) return 'warn';
                    return 'bad';
                };

                return `
                    <div class="doc-card">
                        <div class="doc-header">
                            <span class="doc-index">DOC ${i + 1}</span>
                            <span class="doc-badge ${badgeClass}">${badgeText}</span>
                        </div>
                        <div class="doc-text ${isFiltered ? 'filtered' : ''}">${escapeHtml(doc.text)}</div>
                        ${credScore !== null || hoaxProb !== null ? `
                        <div class="doc-scores">
                            ${credScore !== null ? `
                            <div class="doc-score">
                                <span class="doc-score-label">Trust:</span>
                                <span class="doc-score-value ${scoreClass(credScore)}">${(credScore * 100).toFixed(0)}%</span>
                            </div>
                            ` : ''}
                            ${hoaxProb !== null ? `
                            <div class="doc-score">
                                <span class="doc-score-label">Hoax:</span>
                                <span class="doc-score-value ${hoaxClass(hoaxProb)}">${(hoaxProb * 100).toFixed(0)}%</span>
                            </div>
                            ` : ''}
                        </div>
                        ${credScore !== null ? `
                        <div class="progress-bar">
                            <div class="progress-fill ${credScore < 0.4 ? 'low' : credScore < 0.7 ? 'medium' : ''}" style="width: ${credScore * 100}%"></div>
                        </div>
                        ` : ''}
                        ` : ''}
                    </div>
                `;
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function runPipeline() {
            const btn = document.getElementById('run-btn');
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            const metrics = document.getElementById('metrics');

            btn.disabled = true;
            output.textContent = '';
            output.classList.add('loading');
            status.style.display = 'inline';
            status.textContent = 'PROCESSING';
            status.className = 'status processing';
            metrics.style.display = 'none';
            resetPipeline();

            const input = document.getElementById('input').value.trim();
            const model = document.getElementById('model').value;
            const credibility = document.getElementById('credibility').checked;
            const verify = document.getElementById('verify').checked;

            try {
                setStep('input', 'active');
                await sleep(200);
                setStep('input', 'done');

                if (credibility) {
                    setStep('credibility', 'active');
                }

                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input, model, credibility, verify })
                });

                const data = await response.json();

                output.classList.remove('loading');

                // Render documents
                if (data.documents) {
                    renderDocuments(data.documents);
                }

                if (data.error) {
                    status.textContent = 'ERROR';
                    status.className = 'status error';
                    output.textContent = `Error: ${data.error}`;
                    if (credibility) setStep('credibility', 'error');
                } else {
                    // Animate pipeline
                    if (credibility) setStep('credibility', 'done');
                    await sleep(100);
                    if (verify) {
                        setStep('kg', 'done');
                        await sleep(100);
                    }
                    setStep('summarize', 'done');
                    await sleep(100);
                    if (verify) setStep('verify', 'done');

                    status.textContent = 'COMPLETE';
                    status.className = 'status success';
                    output.textContent = data.summary;

                    // Metrics
                    if (data.metrics) {
                        metrics.style.display = 'grid';
                        document.getElementById('metric-confidence').textContent =
                            data.metrics.confidence ? `${(data.metrics.confidence * 100).toFixed(0)}%` : '-';
                        document.getElementById('metric-docs').textContent = data.metrics.input_docs || '-';
                        document.getElementById('metric-filtered').textContent = data.metrics.filtered_docs || '-';

                        const hallEl = document.getElementById('metric-hallucination');
                        if (data.metrics.hallucination_free !== undefined) {
                            hallEl.textContent = data.metrics.hallucination_free ? '✓' : '✗';
                            hallEl.className = 'metric-value' + (data.metrics.hallucination_free ? '' : ' error');
                        } else {
                            hallEl.textContent = '-';
                            hallEl.className = 'metric-value';
                        }
                    }
                }
            } catch (err) {
                output.classList.remove('loading');
                status.textContent = 'ERROR';
                status.className = 'status error';
                output.textContent = `Network error: ${err.message}`;
            }

            btn.disabled = false;
        }

        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }

        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                runPipeline();
            }
        });
    </script>
</body>
</html>
'''


# Demo documents
DEMO_DOCUMENTS = [
    "Presiden Joko Widodo mengumumkan kebijakan vaksinasi COVID-19 baru pada 15 Januari 2024 di Istana Negara Jakarta. Program ini ditargetkan mencapai 70% cakupan dalam 6 bulan.",
    "VIRAL! Vaksin COVID-19 mengandung microchip 5G untuk pelacakan warga! Bagikan sebelum dihapus pemerintah! Dokter terkenal sudah konfirmasi!",
    "Kementerian Kesehatan melaporkan peningkatan cakupan vaksinasi di seluruh provinsi pada Februari 2024. Menteri Kesehatan Budi Gunadi Sadikin menyatakan target 70% dapat tercapai.",
    "Resep masakan rendang padang yang enak dan mudah dibuat di rumah untuk keluarga tercinta.",
    "WHO memuji keberhasilan program vaksinasi Indonesia pada Maret 2024. Direktur WHO Tedros memberikan apresiasi kepada Kementerian Kesehatan.",
    "DPR menyetujui anggaran tambahan sebesar Rp 150 miliar untuk program vaksinasi nasional pada Januari 2024."
]


@app.route('/')
def index():
    """Serve the main playground interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/summarize', methods=['POST'])
def summarize():
    """Run the summarization pipeline with document-level analysis."""
    try:
        data = request.json
        input_text = data.get('input', '').strip()
        model = data.get('model', 'hybrid')
        use_credibility = data.get('credibility', True)
        use_verify = data.get('verify', True)

        # Use demo documents if no input
        if input_text:
            documents = [d.strip() for d in input_text.split('\n\n') if d.strip()]
            if not documents:
                documents = [input_text]
        else:
            documents = DEMO_DOCUMENTS.copy()

        input_doc_count = len(documents)

        # Document analysis results
        doc_analysis = []
        for doc in documents:
            doc_analysis.append({
                'text': doc[:200] + ('...' if len(doc) > 200 else ''),
                'status': 'valid',
                'credibility_score': None,
                'hoax_probability': None
            })

        filtered_docs = documents
        filtered_indices = set(range(len(documents)))

        # Step 1: Credibility filtering
        if use_credibility:
            try:
                from src.hoax_detection.credibility_report import CredibilityAnalyzer
                analyzer = CredibilityAnalyzer(
                    hoax_model_path="models/hoax_indobert_lora",
                    outlier_threshold_z=2.0,
                    hoax_weight=0.6,
                    outlier_weight=0.4
                )
                filtered_docs, credibility_report = analyzer.filter_documents(documents)

                # Update document analysis with credibility scores
                if hasattr(credibility_report, 'document_scores'):
                    for i, score_data in enumerate(credibility_report.document_scores):
                        if i < len(doc_analysis):
                            doc_analysis[i]['credibility_score'] = score_data.get('credibility_score', 0)
                            doc_analysis[i]['hoax_probability'] = score_data.get('hoax_probability', 0)

                            # Determine status
                            if score_data.get('is_hoax', False):
                                doc_analysis[i]['status'] = 'hoax'
                            elif score_data.get('is_outlier', False):
                                doc_analysis[i]['status'] = 'outlier'
                            elif score_data.get('credibility_score', 1) < 0.5:
                                doc_analysis[i]['status'] = 'filtered'

                # Mark filtered documents
                filtered_set = set(filtered_docs)
                for i, doc in enumerate(documents):
                    if doc not in filtered_set:
                        if doc_analysis[i]['status'] == 'valid':
                            doc_analysis[i]['status'] = 'filtered'

                if not filtered_docs:
                    return jsonify({
                        'error': 'All documents filtered by Trust Layer',
                        'documents': doc_analysis,
                        'metrics': {
                            'input_docs': input_doc_count,
                            'filtered_docs': 0
                        }
                    })

            except ImportError:
                # Credibility module not available
                pass
            except Exception as e:
                print(f"Credibility analysis error: {e}")

        # Step 2: Build Knowledge Graph
        kg = None
        if use_verify:
            try:
                from src.models.knowledge_graph import KnowledgeGraph
                from datetime import datetime
                kg = KnowledgeGraph(name=f"web_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                kg.add_documents(filtered_docs, show_progress=False)
            except ImportError:
                pass

        # Step 3: Generate Summary
        result = {'summary': '', 'metrics': {}, 'documents': doc_analysis}

        if kg and use_verify:
            try:
                from src.models.constrained_summarizer import ConstrainedSummarizer, SummarizationMode

                if model == "gemini":
                    mode = SummarizationMode.ABSTRACTIVE
                elif model in ["textrank", "lexrank"]:
                    mode = SummarizationMode.EXTRACTIVE
                else:
                    mode = SummarizationMode.HYBRID

                summarizer = ConstrainedSummarizer(
                    kg=kg,
                    max_refinement_iterations=3,
                    min_verification_rate=0.7
                )

                sum_result = summarizer.summarize(
                    documents=filtered_docs,
                    mode=mode,
                    num_sentences=5,
                    build_timeline=True
                )

                result = {
                    'summary': sum_result.summary,
                    'documents': doc_analysis,
                    'metrics': {
                        'confidence': sum_result.confidence,
                        'input_docs': input_doc_count,
                        'filtered_docs': len(filtered_docs),
                        'hallucination_free': sum_result.is_hallucination_free,
                        'verification_rate': sum_result.verification_report.verification_rate
                    }
                }
            except ImportError:
                result = _basic_summarize(filtered_docs, model, input_doc_count, doc_analysis)
        else:
            result = _basic_summarize(filtered_docs, model, input_doc_count, doc_analysis)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _basic_summarize(documents: list, model: str, input_doc_count: int, doc_analysis: list) -> dict:
    """Fallback to basic summarization without KG verification."""
    combined_text = ' '.join(documents)
    summary = ""

    try:
        if model == "textrank":
            from src.models.textrank import TextRankSummarizer
            summarizer = TextRankSummarizer(num_sentences=5)
            summary = summarizer.summarize(combined_text)
        elif model == "lexrank":
            from src.models.lexrank import LexRankSummarizer
            summarizer = LexRankSummarizer(num_sentences=5)
            summary = summarizer.summarize(combined_text)
        elif model == "gemini":
            from src.models.gemini_summarizer import GeminiSummarizer
            summarizer = GeminiSummarizer()
            result = summarizer.summarize(documents)
            summary = result.summary
        else:
            try:
                from src.models.gemini_summarizer import GeminiSummarizer
                summarizer = GeminiSummarizer()
                result = summarizer.summarize(documents)
                summary = result.summary
            except:
                from src.models.textrank import TextRankSummarizer
                summarizer = TextRankSummarizer(num_sentences=5)
                summary = summarizer.summarize(combined_text)
    except Exception as e:
        summary = f"Summarization failed: {str(e)}"

    return {
        'summary': summary,
        'documents': doc_analysis,
        'metrics': {
            'input_docs': input_doc_count,
            'filtered_docs': len(documents),
            'confidence': None,
            'hallucination_free': None
        }
    }


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


def run_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask server."""
    print(f"\n{'='*60}")
    print("TDSM Web Playground")
    print(f"{'='*60}")
    print(f"\n→ Open http://localhost:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
