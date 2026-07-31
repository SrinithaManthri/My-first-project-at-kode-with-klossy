# HerPath 🚀

HerPath is an AI-powered career companion designed to help women and girls easily discover scholarships, internships, STEM programs, and mentorships. It simplifies resource accessibility to empower the next generation of female leaders to achieve their academic and career aspirations.

🤗 **Originally built as a Hugging Face Space: https://huggingface.co/spaces/kode-with-klossy/3.4-groupA1-capstone

![Screenshot of my chatbot](Screenshot%20(29).png)

## What it does
* **Personalized Guidance**: Users can type natural language questions to receive tailored academic and career recommendations.
* **Smart Opportunity Search**: Instantly scans a curated database to match girls with specific STEM opportunities and hackathons.
* **One-Click Deadlines**: Features an automated reminder system that links and adds application dates to Google Calendar with a single click.

## How it works
When a user types a message asking for opportunities, HerPath uses semantic search to find the most relevant information within our curated knowledge base. The system processes the query using Retrieval-Augmented Generation (RAG) to ensure the chatbot answers with accurate, context-aware advice rather than making things up.

## Built with
* **Gradio** — the interface
* **Retrieval-Augmented Generation (RAG)** — for accurate information retrieval
* **Google Calendar API** — for the one-click deadline reminder feature
* **Python** — the core programming language 

## What I learned
The most challenging part of this project was setting up the Retrieval-Augmented Generation (RAG) pipeline to pull correct data and cleanly integrating the automated Google Calendar event links. Through teamwork, debugging, and collaboration, we successfully combined our data sources with the AI model to create a seamless user experience.

## About
Built at [Kode With Klossy](https://www.kodewithklossy.com) AI/ML Camp, Summer 2026, through amazing teamwork.
