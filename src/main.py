import argparse
import os
from src.preprocessing import TextPreprocessor
from src.evaluation import Evaluator
from src.models.textrank import TextRankSummarizer
from src.models.lexrank import LexRankSummarizer
from src.models.llm_groq import GroqSummarizer

def main():
    parser = argparse.ArgumentParser(description="Indonesian Multi-Document Summarization")
    parser.add_argument('--mode', type=str, choices=['summarize', 'evaluate'], required=True, help="Mode of operation")
    parser.add_argument('--model', type=str, choices=['textrank', 'lexrank', 'groq'], required=True, help="Model to use")
    parser.add_argument('--input_text', type=str, help="Input text or path to file (for single run)")
    
    args = parser.parse_args()
    
    print(f"Starting {args.mode} with {args.model}...")
    
    # Initialize Model
    summarizer = None
    if args.model == 'textrank':
        summarizer = TextRankSummarizer()
    elif args.model == 'lexrank':
        summarizer = LexRankSummarizer()
    elif args.model == 'groq':
        summarizer = GroqSummarizer()
        
    # Execution Logic
    if args.mode == 'summarize':
        if not args.input_text:
            print("Please provide --input_text for summarization.")
            return
            
        # For demonstration, treating input_text as a single doc or list if it's a file
        # In a real scenario, we'd load multiple docs here.
        docs = [args.input_text] 
        
        # If input is a file, read it
        if os.path.exists(args.input_text):
            with open(args.input_text, 'r', encoding='utf-8') as f:
                docs = [f.read()]
        
        # If using Groq, we might pass a list. For baselines, we usually pass a single concatenated string or handle differently.
        # Adjusting interface for consistency:
        if args.model == 'groq':
            summary = summarizer.summarize(docs)
        else:
            # Baselines currently expect a single string in this simple implementation
            summary = summarizer.summarize(docs[0])
            
        print("\n--- Generated Summary ---\n")
        print(summary)
        print("\n-------------------------\n")

    elif args.mode == 'evaluate':
        print("Evaluation pipeline not yet fully implemented.")

if __name__ == "__main__":
    main()
