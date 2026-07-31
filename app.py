#Imports------------------------------------------------------------------------- 
import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch
import re 
import urllib.parse 
from datetime import datetime 


#Initialize models---------------------------------------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
client = InferenceClient("Qwen/Qwen2.5-Coder-32B-Instruct")

#Preprocessing-----------------------------------------------------------------------
def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    cleaned_chunk=chunk.strip()
    if len(cleaned_chunk)>0:
      cleaned_chunks.append(cleaned_chunk)

  # Print cleaned_chunks
  #print(cleaned_chunks)
  # Print the length of cleaned_chunks
  #print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks
#Embeddings form---------------------------------------------------------------------
def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  #print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  #print(chunk_embeddings.shape)


  # Return the chunk_embeddings
  return chunk_embeddings

#Top Chunks----------------------------------------------------------------------
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  #print(similarities)


  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  #print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    chunk=text_chunks[i]
    top_chunks.append(chunk)
  # Return the list of most relevant chunks
  return top_chunks
    
 #Loading---------------------------------------------------------------------
with open("knowledge.txt", "r", encoding="utf-8") as file:
    # Read the entire contents of the file and store it in a variable
    knowledge_text = file.read()

    # Call the preprocess_text function and store the result in a cleaned_chunks variable
cleaned_chunks = preprocess_text(knowledge_text)

chunk_embeddings = create_embeddings(cleaned_chunks)



def create_google_calendar_link(response_text):
    """
    Looks for a date like:
    Deadline: 15 October 2026
    Application Deadline: 15 October 2026
    """

    pattern = r"(?:Deadline|Application Deadline)\s*:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})"

    match = re.search(pattern, response_text)

    if not match:
        return ""

    try:
        deadline = datetime.strptime(match.group(1), "%d %B %Y")
        start = deadline.strftime("%Y%m%d")
        end = deadline.strftime("%Y%m%d")
        title = urllib.parse.quote("Application Deadline")
        details = urllib.parse.quote("Reminder created by HerPath 🌸")
        url = (
            "https://calendar.google.com/calendar/render?"
            "action=TEMPLATE"
            f"&text={title}"
            f"&dates={start}/{end}"
            f"&details={details}"
        )

        return (
            "\n\n📅 **Deadline Reminder**\n"
            f"🔔 Add this deadline to your Google Calendar:\n{url}"
        )

    except:
        return ""

#Response Function------------------------------------------------------------------
def respond(message, history):
   
    #Response Function added------------------------------------------------------------------
    top_results = get_top_chunks( message, chunk_embeddings, cleaned_chunks)
    context = "\n\n".join(top_results)
    
    system_prompt = (
        "You are an empathetic, empowering AI guide supporting women in STEM, education, career growth, scholarships, and mentorship.\n\n"
        "Use the following context from our knowledge base to help answer the user's question:\n"
        f"--- CONTEXT ---\n{context}\n---------------\n\n"
        "Guidelines:\n"
        "1. Personalize recommendations based on the user's age, level, goals, and location if provided.\n"
        "2. Recommend the most relevant opportunities from the knowledge base, explaining why they fit.\n"
        "3. Include eligibility, benefits, application period, process, and official link when available.\n"
        "4. End with simple, actionable next steps.\n"
        "5. Keep responses concise, well-structured, and encouraging.\n"
        "6. Use tasteful, aesthetic emojis sparingly for emphasis (e.g., ✨, 🌿, 💡, 🎓, 🚀, 💬, 💖)."
        "7. Whenever an opportunity includes an application deadline, provide a **📅 Deadline Reminder** with a Google Calendar link so the user can easily save the deadline. If no application deadline is available, do not generate a Google Calendar reminder and instead advise the user to check the official website for the latest deadline information.\n"
    )

    messages = [{"role": "system", "content": system_prompt}]

#History--------------------------------------------------------------------------
    if history:
        messages.extend(history)

        messages.append({"role": "user", "content": message})
#Calling Model--------------------------------------------------------------------
    response = client.chat_completion(
        messages,
        max_tokens=500,
        temperature =.7,
        top_p=0.9,
    )
    reply = response.choices[0].message.content.strip()

    calendar_link = create_google_calendar_link(reply)

    # Only show reminder for opportunity recommendations
    if ("Application Period" in reply or
        "Official Link" in reply or
        "Eligibility" in reply or
        "Benefits" in reply):
        if calendar_link:
            reply += calendar_link
        else:
            reply += """

    📅 **Deadline Reminder**

    No application deadline is currently available for this opportunity.

    🌐 Check the official website for the latest application deadlines and updates.
    """

    return reply
    
#Launch------------------------------------------------------------------------

# --- Launch Interface ---

my_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="violet"
)

# --- CSS code for details in Interface ---
custom_css = """
/* MAIN BACKGROUND */
:root, html, body, #root, [class*="gradio-container"] { 
    background-image: linear-gradient(135deg, #736686 0%, #9889A5 100%) !important;
    background-color: #f2f1f6 !important;
}


div[class*="row"], div[class*="column"], [data-testid="block-container"], .tabs, 
div[class*="gap"], .form, .block, [class*="gr-box"], [class*="gr-panel"], .metadata, 
div[class*="wrapper"], .padded, .gap, .container, .layout, fieldset, [class*="prose"] {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-container .markdown-text, .gradio-container div[class*="prose"] {
    background-color: transparent !important;
    background: transparent !important;
}


.gradio-container p, .gradio-container h1, .gradio-container h2, .gradio-container span, .gradio-container .markdown-text, .gradio-container label, .gradio-container h3 {
    color: white !important;
}

.chatbot, .message-wrap, .bubble-wrap, div.message-list, div[id="chatbot"], .gradio-chatbot, .chat-view {
    background-color: #A696B3 !important;
    background: #A696B3 !important;
    border: 1px solid #736686 !important;
    border-radius: 12px !important;
}

.user, [class*="user"], .message.user { 
    background-color: #C4B4C8 !important; 
    color: #2A2235 !important; 
}
.user p, .user span, .user strong { color: #2A2235 !important; }

.bot, [class*="bot"], .message.bot, blockquote, pre, code, .prose, 
.bot p, .bot span, .bot strong, .bot li, .bot div { 
    background-color: #E0CFDB !important; 
    background: #E0CFDB !important;
    color: #2A2235 !important; 
}

.chat-suggestions button, [class*="suggestion"], .chatbot .slots button, .form button.primary, .examples button, .example-btn, button[class*="slot"] {
    background-color: #C4B4C8 !important;
    background: #C4B4C8 !important;
    color: #2A2235 !important;
    border: none !important;
    box-shadow: none !important;
}

textarea, div[class*="input-box"], .input-container {
    background-color: #E0CFDB !important;
    background: #E0CFDB !important;
    color: #2A2235 !important;
    border: 1px solid #736686 !important;
    border-radius: 8px !important;
}
textarea::placeholder { color: #736686 !important; opacity: 0.6; }

.submit-button, button[class*="submit"], div[class*="pending"], .generating, [class*="loading"] {
    background-color: #736686 !important;
    background: #736686 !important;
    color: white !important;
}

.message.bot a {
    color: #736686 !important;
    text-decoration: underline !important;
}

div[data-testid="block-container"] img { 
    background: transparent !important; 
    border: none !important; 
    box-shadow: none !important; 
}
"""

# Initialize the interface
with gr.Blocks( ) as demo:


    # 1. Cover Banner (Top)
    cover_image = gr.Image(
        value="updatedbanner.jpeg",
        show_label=False,
        container=False,
        height=180,
        interactive=False
    )

    # 2. Logo & Header Title
    with gr.Row():
        with gr.Column(scale=1, min_width=80):
            logo = gr.Image(
                value="logo.png",
                show_label=False,
                container=False,
                height=80,
                interactive=False
            )

        with gr.Column(scale=5):
            gr.Markdown("<h1 style='color:#d63384; margin: 0;'>HerPath🌸</h1>")
            gr.Markdown("<p style='color: #6f42c1; font-weight: 500;'>Your AI guide for women and girls to discover scholarships, internships, STEM programs, research opportunities, hackathons, competitions, mentorship, and career guidance.</p>")
    gr.ChatInterface(respond,
                examples=[
                    "What STEM scholarships are available for high school seniors?",
                    "Can you suggest hackathons for beginners?",
                    "How do I find career guidance or mentorship in tech?",
                    "What summer research programs or internships are open now?"
                ],
                cache_examples=False)

# 4. Launch the application
demo.launch(theme=my_theme, css=custom_css)


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
