"""
Minimalist Web Playground for NLP Pipeline
Terminal-style brutalist interface
"""

import os
import json
import base64
from flask import Flask, render_template_string, request, jsonify
from typing import Optional

app = Flask(__name__)

# HTML Template - Brutalist Terminal Style
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TDSM // Playground</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --bg: #0a0a0a;
            --fg: #e0e0e0;
            --accent: #00ff88;
            --dim: #444;
            --error: #ff4444;
            --warn: #ffaa00;
        }

        body {
            font-family: "IBM Plex Mono", "SF Mono", "Fira Code", monospace;
            background: var(--bg);
            color: var(--fg);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.6;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        header {
            border-bottom: 1px solid var(--dim);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        h1 {
            font-size: 1.2rem;
            font-weight: 400;
            letter-spacing: 0.1em;
        }

        h1 span {
            color: var(--accent);
        }

        .subtitle {
            color: var(--dim);
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }

        .section {
            margin-bottom: 2rem;
        }

        label {
            display: block;
            color: var(--dim);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.5rem;
        }

        textarea {
            width: 100%;
            background: transparent;
            border: 1px solid var(--dim);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.9rem;
            padding: 1rem;
            resize: vertical;
            min-height: 150px;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
        }

        textarea::placeholder {
            color: var(--dim);
        }

        .controls {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            align-items: center;
            margin: 1.5rem 0;
        }

        select {
            background: transparent;
            border: 1px solid var(--dim);
            color: var(--fg);
            font-family: inherit;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
            cursor: pointer;
        }

        select:focus {
            outline: none;
            border-color: var(--accent);
        }

        select option {
            background: var(--bg);
        }

        button {
            background: transparent;
            border: 1px solid var(--accent);
            color: var(--accent);
            font-family: inherit;
            font-size: 0.85rem;
            padding: 0.5rem 1.5rem;
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            transition: all 0.1s;
        }

        button:hover {
            background: var(--accent);
            color: var(--bg);
        }

        button:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .checkbox-group {
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }

        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
        }

        .checkbox-item input {
            accent-color: var(--accent);
        }

        .output {
            border: 1px solid var(--dim);
            min-height: 200px;
            padding: 1rem;
            white-space: pre-wrap;
            font-size: 0.85rem;
            position: relative;
        }

        .output.loading::after {
            content: "▋";
            animation: blink 1s infinite;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .status {
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border: 1px solid;
        }

        .status.success {
            color: var(--accent);
            border-color: var(--accent);
        }

        .status.error {
            color: var(--error);
            border-color: var(--error);
        }

        .status.processing {
            color: var(--warn);
            border-color: var(--warn);
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--dim);
        }

        .metric {
            text-align: center;
        }

        .metric-value {
            font-size: 1.5rem;
            color: var(--accent);
        }

        .metric-label {
            font-size: 0.7rem;
            color: var(--dim);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--dim);
            color: var(--dim);
            font-size: 0.7rem;
            display: flex;
            justify-content: space-between;
        }

        .ascii-art {
            color: var(--dim);
            font-size: 0.6rem;
            line-height: 1.2;
            opacity: 0.5;
        }

        /* Pipeline visualization */
        .pipeline {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            margin: 1rem 0;
            flex-wrap: wrap;
        }

        .pipeline-step {
            font-size: 0.7rem;
            padding: 0.25rem 0.5rem;
            border: 1px solid var(--dim);
            color: var(--dim);
        }

        .pipeline-step.active {
            border-color: var(--accent);
            color: var(--accent);
        }

        .pipeline-step.done {
            background: var(--accent);
            color: var(--bg);
            border-color: var(--accent);
        }

        .pipeline-arrow {
            color: var(--dim);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>TDSM</span> // Trust-Driven Summarization</h1>
            <p class="subtitle">Indonesian Multi-Document Summarization with Knowledge Graph Verification</p>
        </header>

        <div class="section">
            <label>Input Documents</label>
            <textarea id="input" placeholder="Paste your Indonesian text here...&#10;&#10;Or leave empty to use demo documents about Indonesian vaccination program."></textarea>
        </div>

        <div class="section">
            <label>Pipeline</label>
            <div class="pipeline" id="pipeline">
                <span class="pipeline-step" data-step="input">INPUT</span>
                <span class="pipeline-arrow">→</span>
                <span class="pipeline-step" data-step="credibility">CREDIBILITY</span>
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
                <option value="gemini">Gemini</option>
            </select>

            <div class="checkbox-group">
                <label class="checkbox-item">
                    <input type="checkbox" id="credibility" checked>
                    <span>Credibility Filter</span>
                </label>
                <label class="checkbox-item">
                    <input type="checkbox" id="verify" checked>
                    <span>KG Verification</span>
                </label>
            </div>

            <button id="run-btn" onclick="runPipeline()">Execute</button>
        </div>

        <div class="section">
            <div class="output-header">
                <label>Output</label>
                <span class="status" id="status" style="display:none;"></span>
            </div>
            <div class="output" id="output">Ready. Press Execute to run the pipeline.</div>
        </div>

        <div class="metrics" id="metrics" style="display:none;">
            <div class="metric">
                <div class="metric-value" id="metric-confidence">-</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metric-docs">-</div>
                <div class="metric-label">Documents</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metric-filtered">-</div>
                <div class="metric-label">After Filter</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="metric-hallucination">-</div>
                <div class="metric-label">Hallucination-Free</div>
            </div>
        </div>

        <footer>
            <span>TDSM v1.0 // Knowledge Graph Verification</span>
            <span id="timestamp"></span>
        </footer>
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
                el.classList.remove('active', 'done');
                if (state) el.classList.add(state);
            }
        }

        function resetPipeline() {
            document.querySelectorAll('.pipeline-step').forEach(el => {
                el.classList.remove('active', 'done');
            });
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
                // Simulate pipeline steps
                setStep('input', 'active');
                await sleep(300);
                setStep('input', 'done');

                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ input, model, credibility, verify })
                });

                const data = await response.json();

                output.classList.remove('loading');

                if (data.error) {
                    status.textContent = 'ERROR';
                    status.className = 'status error';
                    output.textContent = `Error: ${data.error}`;
                } else {
                    // Animate pipeline completion
                    if (credibility) {
                        setStep('credibility', 'done');
                    }
                    if (verify) {
                        setStep('kg', 'done');
                    }
                    setStep('summarize', 'done');
                    if (verify) {
                        setStep('verify', 'done');
                    }

                    status.textContent = 'COMPLETE';
                    status.className = 'status success';
                    output.textContent = data.summary;

                    // Show metrics
                    if (data.metrics) {
                        metrics.style.display = 'grid';
                        document.getElementById('metric-confidence').textContent =
                            data.metrics.confidence ? `${(data.metrics.confidence * 100).toFixed(0)}%` : '-';
                        document.getElementById('metric-docs').textContent = data.metrics.input_docs || '-';
                        document.getElementById('metric-filtered').textContent = data.metrics.filtered_docs || '-';
                        document.getElementById('metric-hallucination').textContent =
                            data.metrics.hallucination_free !== undefined ?
                            (data.metrics.hallucination_free ? '✓' : '✗') : '-';
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
    """Run the summarization pipeline."""
    try:
        data = request.json
        input_text = data.get('input', '').strip()
        model = data.get('model', 'hybrid')
        use_credibility = data.get('credibility', True)
        use_verify = data.get('verify', True)

        # Use demo documents if no input
        if input_text:
            # Split by double newlines or treat as single document
            documents = [d.strip() for d in input_text.split('\n\n') if d.strip()]
            if not documents:
                documents = [input_text]
        else:
            documents = DEMO_DOCUMENTS

        input_doc_count = len(documents)
        filtered_docs = documents
        credibility_report = None
        kg = None

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
                if not filtered_docs:
                    return jsonify({
                        'error': 'All documents filtered by Trust Layer',
                        'metrics': {
                            'input_docs': input_doc_count,
                            'filtered_docs': 0
                        }
                    })
            except ImportError:
                # Credibility module not available, continue without it
                pass

        # Step 2: Build Knowledge Graph (if verify enabled)
        if use_verify:
            try:
                from src.models.knowledge_graph import KnowledgeGraph
                from datetime import datetime
                kg = KnowledgeGraph(name=f"web_kg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                kg.add_documents(filtered_docs, show_progress=False)
            except ImportError:
                pass

        # Step 3: Generate Summary
        result = {'summary': '', 'metrics': {}}

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
                    'metrics': {
                        'confidence': sum_result.confidence,
                        'input_docs': input_doc_count,
                        'filtered_docs': len(filtered_docs),
                        'hallucination_free': sum_result.is_hallucination_free,
                        'verification_rate': sum_result.verification_report.verification_rate
                    }
                }
            except ImportError as e:
                # Fall back to basic summarization
                result = _basic_summarize(filtered_docs, model, input_doc_count)
        else:
            result = _basic_summarize(filtered_docs, model, input_doc_count)

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _basic_summarize(documents: list, model: str, input_doc_count: int) -> dict:
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
            # Hybrid - try Gemini first, fall back to TextRank
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
