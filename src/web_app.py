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
                <label>Add from URL</label>
                <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <input type="text" id="url-input" placeholder="https://kompas.com/article-url..." style="flex: 1; padding: 0.5rem; background: var(--bg-tertiary); border: 1px solid var(--border); color: var(--fg); border-radius: 4px;">
                    <button id="fetch-btn" onclick="fetchUrl()" style="padding: 0.5rem 1rem; background: var(--accent); border: none; color: var(--bg); border-radius: 4px; cursor: pointer; font-weight: 600;">Fetch</button>
                </div>
                <div id="fetch-status" style="font-size: 0.8rem; color: var(--fg-muted);"></div>
            </div>

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
                    <option value="gemini">Abstractive</option>
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

            <div class="section" id="verification-section" style="display:none;">
                <label>Verification Details <span style="color: var(--fg-muted); text-transform: none;">(click to expand)</span></label>
                <details>
                    <summary style="cursor:pointer; color: var(--accent);">Show claim-by-claim analysis</summary>
                    <div id="verification-claims" style="margin-top: 0.5rem; font-size: 0.85rem; max-height: 300px; overflow-y: auto;"></div>
                </details>
            </div>

            <div class="section" id="corroboration-section" style="display:none;">
                <label>Source Corroboration <span style="color: var(--fg-muted); text-transform: none;">(web search verification)</span></label>
                <div id="corroboration-content" style="font-size: 0.85rem;">
                    <div style="color: var(--fg-muted); padding: 0.5rem;">Searching for corroborating sources...</div>
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
                    <span>Trustworthiness Analysis</span>
                    <span id="trust-verdict" style="font-weight: 600;"></span>
                </div>
                <div class="sidebar-content" id="trust-analysis" style="padding: 1rem; line-height: 1.6; font-size: 0.85rem;">
                    <div class="empty-state">
                        <div class="empty-state-icon">🔍</div>
                        <div>No analysis yet</div>
                        <div style="margin-top: 0.5rem; font-size: 0.7rem;">Execute the pipeline to see LLM trustworthiness analysis</div>
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

        function renderTrustAnalysis(analysis) {
            const container = document.getElementById('trust-analysis');
            const verdictEl = document.getElementById('trust-verdict');

            if (!analysis || !analysis.summary) {
                container.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">🔍</div>
                        <div>No analysis available</div>
                    </div>
                `;
                verdictEl.textContent = '';
                return;
            }

            // Set verdict badge
            const verdict = analysis.verdict || 'UNCERTAIN';
            const verdictColors = {
                'TRUSTABLE': 'var(--success)',
                'UNCERTAIN': 'var(--warn)',
                'NOT TRUSTABLE': 'var(--error)'
            };
            verdictEl.textContent = verdict;
            verdictEl.style.color = verdictColors[verdict] || 'var(--fg-muted)';

            // Render LLM analysis text (trim whitespace)
            const cleanText = (analysis.summary || '').trim();
            // Replace paragraph breaks with HTML
            const formattedText = escapeHtml(cleanText)
                .split('\\n\\n').join('</p><p style="margin-top: 0.8rem;">')
                .split('\\n').join('<br>');
            container.innerHTML = '<div style="color: var(--fg);">' + formattedText + '</div>';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function fetchUrl() {
            const urlInput = document.getElementById('url-input');
            const fetchStatus = document.getElementById('fetch-status');
            const inputArea = document.getElementById('input');
            const url = urlInput.value.trim();
            
            if (!url) {
                fetchStatus.textContent = 'Please enter a URL';
                fetchStatus.style.color = 'var(--error)';
                return;
            }
            
            fetchStatus.textContent = 'Fetching...';
            fetchStatus.style.color = 'var(--fg-muted)';
            
            try {
                const response = await fetch('/api/fetch-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    fetchStatus.textContent = 'Error: ' + data.error;
                    fetchStatus.style.color = 'var(--error)';
                } else if (data.content) {
                    inputArea.value = data.content;
                    fetchStatus.textContent = 'Done!';
                    fetchStatus.style.color = 'var(--success)';
                }
            } catch (err) {
                fetchStatus.textContent = 'Network error: ' + err.message;
                fetchStatus.style.color = 'var(--error)';
            }
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

                // Render trustworthiness analysis
                if (data.trust_analysis) {
                    renderTrustAnalysis(data.trust_analysis);
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
                    
                    // Verification Details
                    if (data.verification_details && data.verification_details.claims) {
                        const verSection = document.getElementById('verification-section');
                        const verClaims = document.getElementById('verification-claims');
                        verSection.style.display = 'block';
                        
                        const statusEmoji = {
                            'VERIFIED': '✓',
                            'PARTIALLY_VERIFIED': '⚠',
                            'UNVERIFIED': '✗',
                            'CONTRADICTED': '❌',
                            'HALLUCINATION': '🚨'
                        };
                        
                        const statusClass = {
                            'VERIFIED': 'color: var(--success);',
                            'PARTIALLY_VERIFIED': 'color: var(--warning);',
                            'UNVERIFIED': 'color: var(--fg-muted);',
                            'CONTRADICTED': 'color: var(--error);',
                            'HALLUCINATION': 'color: var(--error);'
                        };
                        
                        verClaims.innerHTML = data.verification_details.claims.map((c, i) => `
                            <div style="padding: 0.5rem; border-left: 3px solid ${c.status === 'VERIFIED' ? 'var(--success)' : c.status === 'PARTIALLY_VERIFIED' ? 'var(--warning)' : 'var(--fg-muted)'}; margin-bottom: 0.5rem; background: var(--bg-tertiary); border-radius: 0 4px 4px 0;">
                                <div style="${statusClass[c.status] || ''} font-weight: 600;">
                                    ${statusEmoji[c.status] || '?'} Claim ${i + 1}: ${c.status}
                                </div>
                                <div style="margin-top: 0.3rem; color: var(--fg);">"${escapeHtml(c.claim)}"</div>
                                ${c.explanation ? `<div style="margin-top: 0.2rem; color: var(--fg-muted); font-size: 0.8rem;">📋 ${escapeHtml(c.explanation)}</div>` : ''}
                            </div>
                        `).join('');
                    }
                    
                    // Render Source Corroboration
                    if (data.corroboration_sources && data.corroboration_sources.length > 0) {
                        const corSection = document.getElementById('corroboration-section');
                        const corContent = document.getElementById('corroboration-content');
                        corSection.style.display = 'block';
                        
                        const sources = data.corroboration_sources;
                        corContent.innerHTML = `
                            <div style="margin-bottom: 0.5rem; color: var(--success);">
                                ✓ Found ${sources.length} corroborating sources
                            </div>
                            ${sources.map((s, i) => `
                                <div style="padding: 0.5rem; background: var(--bg-tertiary); border-radius: 4px; margin-bottom: 0.5rem;">
                                    <div style="font-weight: 600; color: var(--accent);">
                                        <a href="${escapeHtml(s.url)}" target="_blank" style="color: var(--accent); text-decoration: none;">
                                            ${escapeHtml(s.title || 'Source ' + (i + 1))}
                                        </a>
                                    </div>
                                    <div style="font-size: 0.75rem; color: var(--fg-muted);">${escapeHtml(s.source || s.domain || '')}</div>
                                    ${s.snippet ? `<div style="margin-top: 0.3rem; color: var(--fg); font-size: 0.8rem;">${escapeHtml(s.snippet.substring(0, 150))}...</div>` : ''}
                                </div>
                            `).join('')}
                        `;
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

        // Fetch URL content
        async function fetchUrl() {
            const urlInput = document.getElementById('url-input');
            const btn = document.getElementById('fetch-btn');
            const status = document.getElementById('fetch-status');
            const textarea = document.getElementById('input');
            const url = urlInput.value.trim();

            if (!url) {
                status.textContent = 'Please enter a URL';
                status.style.color = 'var(--error)';
                return;
            }

            btn.disabled = true;
            btn.textContent = '...';
            status.textContent = 'Fetching article...';
            status.style.color = 'var(--fg-muted)';

            try {
                const response = await fetch('/api/fetch-url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url })
                });

                const data = await response.json();

                if (data.error) {
                    status.textContent = 'Error: ' + data.error;
                    status.style.color = 'var(--error)';
                } else if (data.content) {
                    textarea.value = data.content;
                    status.textContent = `Fetched ${data.content.length} chars. Running pipeline...`;
                    status.style.color = 'var(--success)';
                    urlInput.value = '';
                    
                    // Auto-run pipeline
                    await runPipeline();
                    status.textContent = 'Done!';
                } else {
                    status.textContent = 'No content found';
                    status.style.color = 'var(--warning)';
                }
            } catch (e) {
                status.textContent = 'Network error: ' + e.message;
                status.style.color = 'var(--error)';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Fetch';
            }
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

                # Get Gemini API key from environment for hybrid/abstractive modes
                import os
                gemini_key = os.environ.get('GEMINI_API_KEY')
                
                summarizer = ConstrainedSummarizer(
                    kg=kg,
                    gemini_api_key=gemini_key,
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
                    },
                    'verification_details': {
                        'overall_status': sum_result.verification_report.overall_status.value,
                        'claims': [
                            {
                                'claim': r.claim[:150] + '...' if len(r.claim) > 150 else r.claim,
                                'status': r.status.value,
                                'confidence': round(r.confidence, 2),
                                'explanation': r.explanation
                            }
                            for r in sum_result.verification_report.claim_results
                        ]
                    }
                }
                
                # Generate LLM trustworthiness analysis with web search grounding
                try:
                    from src.models.gemini_summarizer import GeminiSummarizer
                    from src.tools.enhanced_search import EnhancedSearcher
                    
                    llm = GeminiSummarizer()
                    combined_text = ' '.join(filtered_docs)
                    
                    # Generate search query from article content
                    search_query = llm.generate_search_query(combined_text[:1000])
                    print(f"[Web Search] Query: {search_query}")
                    
                    # Search for related articles to corroborate claims
                    search_results = []
                    try:
                        searcher = EnhancedSearcher(max_results=5)
                        raw_results = searcher.search_sync(search_query, max_results=5)
                        search_results = [
                            {
                                'title': r.title,
                                'source': r.domain,
                                'url': r.url,
                                'snippet': r.snippet
                            }
                            for r in raw_results
                        ]
                        print(f"[Web Search] Found {len(search_results)} related sources")
                    except Exception as se:
                        print(f"[Web Search] Search error: {se}")
                    
                    # Verify article with web search context
                    trust_result = llm.verify_article(
                        title="Article Analysis",
                        content=combined_text,
                        hoax_probability=doc_analysis[0].get('hoax_probability') if doc_analysis else None,
                        search_results=search_results
                    )
                    result['trust_analysis'] = trust_result
                    result['corroboration_sources'] = search_results  # Add web search sources
                    
                    # Override confidence with LLM verdict (more accurate for abstractive)
                    if trust_result and trust_result.get('verdict'):
                        verdict_confidence = {
                            'TRUSTABLE': 0.85,
                            'UNCERTAIN': 0.50,
                            'NOT TRUSTABLE': 0.15
                        }
                        result['metrics']['confidence'] = verdict_confidence.get(
                            trust_result['verdict'], 0.50
                        )
                        result['metrics']['hallucination_free'] = trust_result['verdict'] == 'TRUSTABLE'
                        
                except Exception as e:
                    print(f"Trust analysis error: {e}")
                    result['trust_analysis'] = None
                    result['corroboration_sources'] = []
                    
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

    # Generate LLM trustworthiness analysis with web search grounding
    trust_analysis = None
    try:
        from src.models.gemini_summarizer import GeminiSummarizer
        from src.tools.enhanced_search import EnhancedSearcher
        
        llm = GeminiSummarizer()
        
        # Generate search query and find related sources
        search_query = llm.generate_search_query(combined_text[:1000])
        search_results = []
        try:
            searcher = EnhancedSearcher(max_results=5)
            raw_results = searcher.search_sync(search_query, max_results=5)
            search_results = [
                {'title': r.title, 'source': r.domain, 'url': r.url, 'snippet': r.snippet}
                for r in raw_results
            ]
        except Exception as se:
            print(f"[Web Search] Error: {se}")
        
        trust_result = llm.verify_article(
            title="Article Analysis",
            content=combined_text,
            hoax_probability=doc_analysis[0].get('hoax_probability') if doc_analysis else None,
            search_results=search_results
        )
        trust_analysis = trust_result
    except Exception as e:
        print(f"Trust analysis error: {e}")

    # Calculate confidence from LLM verdict
    confidence = None
    hallucination_free = None
    if trust_analysis and trust_analysis.get('verdict'):
        verdict_confidence = {
            'TRUSTABLE': 0.85,
            'UNCERTAIN': 0.50,
            'NOT TRUSTABLE': 0.15
        }
        confidence = verdict_confidence.get(trust_analysis['verdict'], 0.50)
        hallucination_free = trust_analysis['verdict'] == 'TRUSTABLE'

    return {
        'summary': summary,
        'documents': doc_analysis,
        'trust_analysis': trust_analysis,
        'corroboration_sources': search_results,
        'metrics': {
            'input_docs': input_doc_count,
            'filtered_docs': len(documents),
            'confidence': confidence,
            'hallucination_free': hallucination_free
        }
    }


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


@app.route('/api/fetch-url', methods=['POST'])
def fetch_url():
    """Fetch content from a news URL using crawl4ai."""
    try:
        data = request.json
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        print(f"[API] Fetching URL: {url}")
        
        from src.tools.enhanced_search import EnhancedSearcher
        searcher = EnhancedSearcher()
        
        content = searcher.fetch_url_content(url)
        
        if content and len(content) > 50:
            print(f"[API] Fetched {len(content)} chars")
            return jsonify({'content': content})
        else:
            return jsonify({'error': 'Failed to extract content from URL'}), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def run_server(host: str = None, port: int = 5000, debug: bool = False):
    """
    Run the Flask server.
    
    Args:
        host: Host to bind to. Defaults to BIND_HOST env var or '127.0.0.1' (localhost only).
              Set BIND_HOST=0.0.0.0 for Docker/LAN access.
        port: Port to bind to.
        debug: Enable debug mode.
    """
    import os
    if host is None:
        host = os.environ.get('BIND_HOST', '127.0.0.1')
    
    print(f"\n{'='*60}")
    print("TDSM Web Playground")
    print(f"{'='*60}")
    print(f"\n-> Open http://localhost:{port} in your browser\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
