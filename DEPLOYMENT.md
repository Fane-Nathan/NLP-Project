# Deployment Guide: Hugging Face Spaces

This guide explains how to deploy your **Text-Driven Summarization & Verification (TDSM)** app to the web using **Hugging Face Spaces**.

## Why Hugging Face Spaces?

- **Free**: Generous free tier (2 vCPU, 16GB RAM) sufficient for this demo.
- **ML-Ready**: Built for AI apps.
- **Easy**: Connects directly to your GitHub repository.

## Prerequisites

- A [Hugging Face Account](https://huggingface.co/join)
- Your **GitHub Repository** pushed with the latest code (which we just did!).
- API Keys ready:
  - `GROQ_API_KEY`
  - `GEMINI_API_KEY`

## Step-by-Step Deployment

1.  **Create a New Space**

    - Go to [huggingface.co/spaces](https://huggingface.co/spaces).
    - Click **"Create new Space"**.

2.  **Configure the Space**

    - **Space Name**: `TDSM-News-Verifier` (or similar).
    - **License**: `MIT` (or your choice).
    - **SDK**: Select **Docker**. (This is critical!).
    - **Space Hardware**: `CPU basic (2 vCPU, 16 GB, free)` is enough.

3.  **Connect Code**

    - You will see an option to "Connect a repository".
    - Select your GitHub repo: `Fane-Nathan/NLP-Project`.
    - Authorize Hugging Face to access your repo.

4.  **Set Environment Variables (Secrets)**

    - Once the Space is created, go to the **"Settings"** tab of your Space.
    - Scroll to **"Variables and secrets"**.
    - Click **"New secret"** and add:
      - Name: `GROQ_API_KEY` | Value: (Your actual key)
      - Name: `GEMINI_API_KEY` | Value: (Your actual key)

5.  **Build & Run**
    - Hugging Face will automatically pull your `Dockerfile`, build it, and run it.
    - Watch the **"Logs"** tab. It might take 5-10 minutes to build the first time (installing dependencies).
    - Once finished, you will see your app running live! 🚀

## Troubleshooting

- **Build Failures**: Check the "Logs". If it says "Out of Memory", we might need to reduce the `workers` in `Dockerfile` (I already set it to 1 to be safe).
- **Slow Startup**: The model downloads might take time. This is normal for the first run.

## Sharing

- Once running, copied the URL (e.g., `https://huggingface.co/spaces/username/TDSM-News-Verifier`).
- Send it to your friends/testers!
