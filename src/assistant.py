import keyboard
import time
import pyperclip
import threading
import re
import os
import json
from src.screen_capture import ScreenCapturer
from src.models.llm_groq import GroqSummarizer
from src.models.gemini_summarizer import GeminiSummarizer
from src.models.knowledge_graph import KnowledgeGraph
from src.voice_kokoro import JarvisVoice
from src.monitor import WindowMonitor
from src.tools.search_tool import WebSearcher
from src.preprocessing import TextPreprocessor
from src.pipeline_live import LiveHistorian
from src.models.llama_kg import LlamaIndexManager

# Hoax detection imports
try:
    from src.hoax_detection import HoaxClassifier
    HOAX_DETECTION_AVAILABLE = True
except ImportError as e:
    HOAX_DETECTION_AVAILABLE = False
    print(f"[Warning] Hoax detection module not available: {e}")


def main():
    print("Initializing JARVIS System...")
    
    try:
        print("[DEBUG] Initializing ScreenCapturer...")
        capturer = ScreenCapturer()
        print("[DEBUG] Initializing GroqSummarizer...")
        summarizer = GroqSummarizer()
        print("[DEBUG] Initializing GeminiSummarizer...")
        gemini_summarizer = GeminiSummarizer()
        print("[DEBUG] Initializing JarvisVoice...")
        voice = JarvisVoice()
        print("[DEBUG] Initializing WebSearcher...")
        searcher = WebSearcher(max_results=4)
        print("[DEBUG] Initializing TextPreprocessor...")
        preprocessor = TextPreprocessor(use_stemmer=False) # Stemmer not needed for cleaning
        
        # Initialize hoax classifier
        hoax_classifier = None
        if HOAX_DETECTION_AVAILABLE:
            print("[DEBUG] Hoax detection available. Loading classifier...")
            for attempt in range(3):
                try:
                    hoax_classifier = HoaxClassifier("models/hoax_indobert_lora")
                    print("✓ Hoax Detector loaded (99.5% accuracy)")
                    break
                except Exception as e:
                    print(f"[Warning] Hoax classifier load attempt {attempt+1} failed: {e}")
                    if attempt < 2:
                        print("Retrying in 2 seconds...")
                        time.sleep(2)
            
            if not hoax_classifier:
                print("[Error] Failed to load Hoax Classifier after 3 attempts.")
        else:
            print("[DEBUG] Hoax detection NOT available.")

        # Initialize Live Historian (reusing existing models)
        print("[DEBUG] Initializing Live Historian...")
        live_historian = LiveHistorian(verifier=hoax_classifier, historian=gemini_summarizer)
        
        # Initialize LlamaIndex KG
        print("[DEBUG] Initializing LlamaIndex Knowledge Graph...")
        llama_kg = LlamaIndexManager()
                
    except Exception as e:
        print(f"Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n=== JARVIS Online ===")
    
    print("Hotkeys:")
    print("  Ctrl+Alt+S: Summarize Screen")
    print("  Ctrl+Alt+D: Describe Screen")
    print("  Ctrl+Alt+H: Hoax Check")
    print("  Ctrl+Alt+J: Enhanced Hoax Check (LLM)")
    print("  Ctrl+Alt+K: Save to Knowledge Graph")
    print("  Ctrl+Alt+W: Web Research (Clipboard)")
    print("  Ctrl+Alt+L: Live Historian (Clipboard)")
    print("  Ctrl+Alt+Enter: Unified Hoax Pipeline (Super Check)")
    print("  Ctrl+Alt+V: Toggle Voice")
    print("  Esc: Exit")

    # --- Proactive Monitoring ---
    def on_interesting_window(title):
        msg = f"Sir, I see you are looking at {title}. Shall I summarize it?"
        print(f"\n[Monitor] {msg}")
        voice.speak(msg)

    monitor = WindowMonitor(callback=on_interesting_window)
    monitor.start()

    # --- Helper Functions ---
    def clean_text_for_speech(text):
        """Removes markdown and special characters for better TTS."""
        if not text: return ""
        text = re.sub(r'[*_#`]', '', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'http\S+', 'website', text)
        return text.strip()

    def generate_search_params(text):
        """Uses LLM to generate an optimal search query and time limit."""
        prompt = f"""
        You are a search expert. Given this user input, generate the optimal search query to verify it.
        Also determine if we need 'latest' news (past 24h).
        
        Input: "{text[:500]}"
        
        Output JSON ONLY:
        {{
            "query": "refined search query",
            "timelimit": "d" (for breaking news/today), "w" (recent/week), or null (general/history)
        }}
        """
        try:
            response = gemini_summarizer.summarize(prompt).summary
            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[SearchGen] Error: {e}")
        
        # Fallback
        return {"query": text, "timelimit": None}

    # --- Action Handlers ---
    def process_screen(mode="summarize"):
        print(f"\n[Processing: {mode.upper()}]")
        
        try:
            # 1. Capture
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            result = None
            
            # 2. Vision Analysis
            if mode == "summarize":
                # Use Gemini for Vision
                result = gemini_summarizer.summarize_image(
                    image_base64, 
                    prompt="Analyze this screen. Tell me the story of what is shown here in a natural, professional way."
                )
                if not result:
                    voice.speak("Vision sensors unclear.")
            
            elif mode == "describe":
                # Use Gemini for Description
                result = gemini_summarizer.describe_image(image_base64)

            # 3. Output
            if result:
                print("\n=== RESULT ===\n")
                print(result)
                print("\n==============\n")
                pyperclip.copy(result)
                voice.speak(clean_text_for_speech(result))
            else:
                voice.speak("I could not analyze the screen content, sir.")
                
        except Exception as e:
            print(f"Error: {e}")
            voice.speak("An error occurred during processing.")

    # --- NEW: Web Research Mode ---
    def web_research_mode():
        """Reads clipboard, searches web, and summarizes."""
        print("\n[Processing: WEB RESEARCH]")
        
        try:
            query = pyperclip.paste().strip()
            if not query:
                voice.speak("Clipboard is empty, sir.")
                return

            voice.speak(f"Searching the web for: {query[:20]}...")
            print(f"[Web] Input: {query}")

            # Smart Query Generation
            search_params = generate_search_params(query)
            search_query = search_params.get("query", query)
            timelimit = search_params.get("timelimit")
            
            print(f"[Web] Smart Query: '{search_query}' (Time: {timelimit})")

            # 1. Search
            search_context = searcher.get_formatted_results(search_query, timelimit=timelimit)
            
            if "No search results" in search_context:
                voice.speak("I couldn't find anything relevant online.")
                return

            # 2. Synthesize
            answer_result = summarizer.summarize_with_search(query, search_context)
            answer = answer_result 
            
            if answer:
                print("\n=== WEB RESULT ===\n")
                print(answer)
                print("\n==================\n")
                voice.speak(clean_text_for_speech(answer))
                pyperclip.copy(answer)
            else:
                voice.speak("I failed to synthesize an answer.")
                
        except Exception as e:
            print(f"Error during web research: {e}")
            voice.speak("Web research failed.")

    # --- NEW: Live Historian Mode ---
    def run_live_historian():
        """Reads clipboard, runs Live Historian pipeline."""
        print("\n[Processing: LIVE HISTORIAN]")
        
        try:
            query = pyperclip.paste().strip()
            if not query:
                voice.speak("Clipboard is empty, sir.")
                return

            voice.speak(f"Gathering intelligence on: {query[:30]}...")
            print(f"[Historian] Query: {query}")

            # Run pipeline
            narrative = live_historian.generate_live_history(query)
            
            if narrative and "No credible intelligence" not in narrative:
                print("\n=== HISTORICAL RECORD ===\n")
                print(narrative)
                print("\n=========================\n")
                voice.speak("Intelligence gathered and verified.")
                pyperclip.copy(narrative)
            else:
                voice.speak("No credible intelligence found.")
                print(f"[Historian] Result: {narrative}")
                
        except Exception as e:
            print(f"Error during live historian: {e}")
            voice.speak("Live historian protocol failed.")

    # --- NEW: Capture to Knowledge Graph ---
    def process_to_kg():
        """Capture screen, summarize, and save to Knowledge Graph."""
        print("\n[Processing: SAVE TO MEMORY]")
        voice.speak("Capturing for long-term memory.")
        
        try:
            # 1. Capture & Extract Text (Gemini Vision Only)
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            print("[Vision] Extracting text with Gemini...")
            text = gemini_summarizer.summarize_image(
                image_base64, 
                prompt="Transcribe the text in this image exactly. Do not add commentary. If there is no text, return empty string."
            )
            
            if not text or len(text.strip()) < 20:
                voice.speak("Text unclear. Cannot save to memory.")
                return

            print(f"[Vision] Extracted {len(text)} chars")
            
            # 2. Preprocessing (Cleaning)
            print("[Preprocessing] Cleaning text...")
            cleaned_text = preprocessor.clean_for_kg(text)
            print(f"[Preprocessing] Cleaned text length: {len(cleaned_text)}")
            
            if len(cleaned_text) < 20:
                voice.speak("Content too sparse after cleaning. Aborting.")
                return

            # 3. Summarize with Gemini (Better for KG)
            print("[Gemini] Generating summary...")
            summary_result = gemini_summarizer.summarize(cleaned_text, style="detailed")
            summary = summary_result.summary
            
            print(f"\n[Summary] {summary[:100]}...\n")

            # 4. Save to KG
            kg_path = "data/persistent_kg.json"
            kg = KnowledgeGraph(name="jarvis_memory")
            
            # Load existing if available
            if os.path.exists(kg_path):
                print(f"[KG] Loading existing memory from {kg_path}")
                kg.load(kg_path)
            
            # Add document
            print("[KG] Extracting facts and updating memory...")
            kg.add_documents([cleaned_text])
            
            # Save back
            kg.save(kg_path)
            print(f"[KG] Saved to {kg_path}")
            
            voice.speak("Information processed and saved to knowledge graph.")
            
        except Exception as e:
            print(f"Error saving to KG: {e}")
            voice.speak("Failed to save to memory.")

    # --- Hoax Check Handlers ---
    def check_hoax():
        """Extract text from screen and check for hoax/misinformation."""
        print("\n[Processing: HOAX CHECK]")
        
        if not hoax_classifier:
            voice.speak("Hoax detection is not available. Please ensure the model is trained.")
            return
        
        try:
            # 1. Capture screen
            voice.speak("Scanning for misinformation.")
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            # 2. Extract text via Gemini Vision
            print("[Vision] Extracting text...")
            text = gemini_summarizer.summarize_image(
                image_base64, 
                prompt="Transcribe the text in this image exactly. Do not add commentary."
            )
            
            if not text or len(text.strip()) < 20:
                voice.speak("I couldn't extract enough text from the screen. Please try again with more visible text.")
                return
            
            print(f"[Vision] Extracted {len(text)} characters")
            print(f"[Vision] Preview: {text[:200]}...")
            
            # 3. Run hoax classification
            result = hoax_classifier.predict(text)
            
            # 4. Generate response
            print("\n=== HOAX ANALYSIS ===")
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Hoax Probability: {result.hoax_probability:.1%}")
            print("=====================\n")
            
            # 5. Speak result
            if result.hoax_probability >= 0.7:
                # High confidence hoax
                response = f"Warning! This content has a {result.hoax_probability:.0%} probability of being misinformation. I recommend fact-checking before sharing."
                print(f"🚨 {response}")
            elif result.hoax_probability >= 0.5:
                # Uncertain
                response = f"This content is uncertain. There's a {result.hoax_probability:.0%} chance it could be misinformation. Please verify with trusted sources."
                print(f"⚠️ {response}")
            else:
                # Likely valid
                response = f"This content appears credible with {result.valid_probability:.0%} confidence. However, always verify important information."
                print(f"✅ {response}")
            
            voice.speak(response)
            
            # Copy result to clipboard
            clipboard_text = f"""
HOAX ANALYSIS RESULT
====================
Label: {result.label}
Hoax Probability: {result.hoax_probability:.1%}
Valid Probability: {result.valid_probability:.1%}
Confidence: {result.confidence:.1%}

Text Analyzed:
{text[:500]}{'...' if len(text) > 500 else ''}
"""
            pyperclip.copy(clipboard_text)
            print("[Clipboard] Analysis copied")
                
        except Exception as e:
            print(f"Error during hoax check: {e}")
            import traceback
            traceback.print_exc()
            voice.speak("An error occurred during the hoax analysis.")

    def check_hoax_with_llm():
        """Extract text, check hoax, and get LLM explanation."""
        print("\n[Processing: ENHANCED HOAX CHECK]")
        
        if not hoax_classifier:
            voice.speak("Hoax detection is not available.")
            return
        
        try:
            voice.speak("Performing deep credibility analysis.")
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            # 1. Extract text (Gemini Vision)
            print("[Vision] Extracting text...")
            text = gemini_summarizer.summarize_image(
                image_base64, 
                prompt="Transcribe the text in this image exactly. Do not add commentary."
            )
            
            if not text or len(text.strip()) < 20:
                voice.speak("Not enough text to analyze.")
                return
            
            # 2. Run hoax classification
            result = hoax_classifier.predict(text)
            
            # 3. KG Context Retrieval
            kg_context = ""
            try:
                kg_path = "data/persistent_kg.json"
                if os.path.exists(kg_path):
                    print("[KG] Checking memory for context...")
                    kg = KnowledgeGraph(name="jarvis_memory")
                    kg.load(kg_path)
                    
                    # Simple keyword matching
                    found_entities = []
                    text_lower = text.lower()
                    
                    for key, node_data in kg.graph.nodes(data=True):
                        names = {node_data.get('normalized', '').lower()}
                        names.update(a.lower() for a in node_data.get('aliases', []))
                        
                        if any(n in text_lower for n in names if len(n) > 3):
                            found_entities.append(key)
                    
                    # Get relations
                    facts = []
                    for entity_key in found_entities[:5]:
                        relations = kg.get_relations_for_entity(entity_key)
                        for rel in relations[:3]:
                            facts.append(f"- {rel['subject']} {rel['predicate']} {rel['object']}")
                    
                    if facts:
                        kg_context = "\nKNOWN FACTS FROM MEMORY:\n" + "\n".join(facts) + "\n"
                        print(f"[KG] Found {len(facts)} relevant facts.")
            except Exception as kg_e:
                print(f"[KG] Retrieval failed: {kg_e}")

            # 4. Build prompt for LLM
            analysis_prompt = f"""Analyze this content for misinformation. My AI detector classified it as {result.label} with {result.hoax_probability:.0%} hoax probability.

Content:
"{text[:1000]}"

{kg_context}
Task:
Tell me the story of this content's credibility. Is it trustworthy?
Compare it with the known facts if provided.
Explain your reasoning naturally in 1-2 fluid paragraphs.
Avoid bullet points. Speak like a professional colleague.
"""

            # 5. Get LLM analysis
            llm_result = gemini_summarizer.summarize(analysis_prompt)
            llm_analysis = llm_result.summary
            
            if llm_analysis:
                print("\n=== LLM ANALYSIS ===")
                print(llm_analysis)
                print("====================\n")
                
                # Combine results
                full_response = f"Hoax probability: {result.hoax_probability:.0%}. {llm_analysis}"
                voice.speak(clean_text_for_speech(full_response))
                pyperclip.copy(llm_analysis)
            else:
                # Fallback to basic response
                check_hoax()
                
        except Exception as e:
            print(f"Error: {e}")
            voice.speak("Analysis failed.")

    # --- NEW: Unified Pipeline (Super Check) ---
    def run_unified_pipeline():
        """Combines Screen/Clipboard -> Web Search -> Hoax Check -> LlamaIndex KG."""
        print("\n[Processing: UNIFIED SUPER CHECK]")
        voice.speak("Initiating full verification protocol.")
        
        try:
            # 1. Get Context (Screen or Clipboard)
            # Priority: Clipboard if text (unless it's our own output), else Screen
            query = pyperclip.paste().strip()
            source_text = ""
            
            # Check if clipboard contains our own output or startup message to avoid loops
            ignore_clipboard = False
            if "UNIFIED VERIFICATION RESULT" in query or "=== JARVIS Online ===" in query:
                print("[Input] Clipboard contains JARVIS output/context. Ignoring to avoid loop.")
                ignore_clipboard = True
            
            if query and len(query) > 10 and not ignore_clipboard:
                print(f"[Input] Using clipboard: {query[:50]}...")
                source_text = query
            else:
                print("[Input] Capturing screen (Clipboard empty or ignored)...")
                image = capturer.capture_screen()
                image_base64 = capturer.image_to_base64(image)
                source_text = gemini_summarizer.summarize_image(
                    image_base64, 
                    prompt="Extract the main news or claim from this screen. Return only the text."
                )
            
            if not source_text:
                voice.speak("No input detected.")
                return

            # 2. Web Search (Live Context)
            print("[Step 1] Searching web...")
            
            # Smart Query Generation
            search_params = generate_search_params(source_text)
            search_query = search_params.get("query", source_text)
            timelimit = search_params.get("timelimit")
            
            print(f"[Web] Smart Query: '{search_query}' (Time: {timelimit})")
            
            search_results = searcher.get_formatted_results(search_query, timelimit=timelimit)
            
            # 3. LlamaIndex Retrieval (Historical Context)
            print("[Step 2] Querying Knowledge Graph...")
            kg_context = llama_kg.query(source_text)
            if kg_context:
                print(f"[KG] Found context: {kg_context[:100]}...")
            else:
                print("[KG] No prior knowledge found.")

            # 4. Hoax Check (IndoBERT)
            print("[Step 3] Running Hoax Detector...")
            hoax_result = None
            if hoax_classifier:
                hoax_result = hoax_classifier.predict(source_text)
            
            # 5. Synthesis (Gemini)
            print("[Step 4] Synthesizing Verdict...")
            prompt = f"""
            Verify this claim/news.
            
            CLAIM: "{source_text}"
            
            WEB EVIDENCE:
            {search_results}
            
            INTERNAL MEMORY (KG):
            {kg_context}
            
            HOAX DETECTOR: {hoax_result.label if hoax_result else "N/A"} ({hoax_result.hoax_probability:.1%} hoax prob)
            
            TASK:
            1. Determine if this is TRUE, FALSE, or UNCERTAIN.
            2. Provide a concise explanation citing the evidence.
            3. If it's a confirmed fact, explicitly say "WORTH SAVING".
            """
            
            verdict_result = gemini_summarizer.summarize(prompt)
            verdict = verdict_result.summary
            print(f"\n=== VERDICT ===\n{verdict}\n===============\n")
            
            # Copy to clipboard
            clipboard_content = f"""
UNIFIED VERIFICATION RESULT
===========================
VERDICT:
{verdict}

Hoax Detector: {hoax_result.label if hoax_result else "N/A"} ({hoax_result.hoax_probability:.1%} hoax prob)

SOURCE CLAIM:
{source_text[:500]}{'...' if len(source_text) > 500 else ''}
"""
            pyperclip.copy(clipboard_content)
            print("[Clipboard] Verdict copied")

            voice.speak(clean_text_for_speech(verdict))
            
            # 6. Save to KG if worth saving
            if "WORTH SAVING" in verdict.upper() or "TRUE" in verdict.upper():
                print("[Step 5] Saving to LlamaIndex...")
                llama_kg.add_document(
                    text=f"Claim: {source_text}\nVerdict: {verdict}",
                    metadata={"source": "unified_pipeline", "timestamp": time.time()}
                )
                voice.speak("Verified and saved to memory.")
            
        except Exception as e:
            print(f"Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            voice.speak("Protocol failed.")

    def toggle_voice():
        new_persona = "jarvis" if voice.persona == "friday" else "friday"
        voice.set_persona(new_persona)
        voice.speak(f"Voice protocol switched to {new_persona.capitalize()}.")

    # Register Hotkeys
    keyboard.add_hotkey('ctrl+alt+s', lambda: process_screen("summarize"))
    keyboard.add_hotkey('ctrl+alt+d', lambda: process_screen("describe"))
    keyboard.add_hotkey('ctrl+alt+h', check_hoax)
    keyboard.add_hotkey('ctrl+alt+j', check_hoax_with_llm)
    keyboard.add_hotkey('ctrl+alt+k', process_to_kg)
    keyboard.add_hotkey('ctrl+alt+w', web_research_mode)
    keyboard.add_hotkey('ctrl+alt+l', run_live_historian)
    keyboard.add_hotkey('ctrl+alt+enter', run_unified_pipeline)
    keyboard.add_hotkey('ctrl+alt+v', toggle_voice)
    
    # Keep running
    keyboard.wait('esc')
    
    # Cleanup
    monitor.stop()
    voice.speak("Shutting down.")
    print("Exiting...")


if __name__ == "__main__":
    main()