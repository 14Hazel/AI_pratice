question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text translated into mandarin.
{context}
Question: {question}
Relevant text, if any, in Mandarin:"""


combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer chinese. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
Answer in Mandarin:"""
