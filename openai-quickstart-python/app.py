from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import distances_from_embeddings
import os

app = Flask(__name__)

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    
    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    # If debug, print the raw model response
    content = "Context: " + context + "\n\n---\n\nQuestion: " + question + "\nAnswer:"
    if debug:
        print("Context:\n" + content)
        print("\n\n")
    try:
        # Create a chat completion using the question and context
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0,
            max_tokens=12,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""

@app.route("/", methods=["GET", "POST"])
def index():
    
     # Set the OpenAI API key from environment variables
    with open("/Users/jubaidatasnim/openAI/openai-quickstart-python/.env") as env:
        for line in env:
            key, value = line.strip().split("=")
            os.environ[key] = value
    openai.api_key = os.getenv('API_KEY')
    df=pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    output = None
    if request.method == "POST":
        # Get the text input from the form
        user_input = request.form["text_input"]
        res = answer_question(df, question=user_input, debug=False)
        output = f"You entered: {res}"
        
    # Render the HTML template and pass the output to be displayed
    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)

