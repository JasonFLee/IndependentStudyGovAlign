LLM Quiz Evaluation GUI

A minimal Python GUI application built with Tkinter for creating, storing, and evaluating multiple-choice questions using OpenAI’s GPT models (default: GPT-4).

note: since this is not public my api key is there so please be kind :D

Quick Start
-----------
1. Clone or download this code (or copy/paste the code into `quiz_app.py`).
2. Install Dependencies:
   pip install openai
   Alternatively, you can pin your installation:
   pip install openai==0.28
   If tkinter is missing on Linux:
   sudo apt-get install python3-tk

3. Run the code:
   python quiz_app.py

How It Works
------------
1. A window appears.  
2. Enter your question text.  
3. Provide four possible answers.  
4. Check exactly one “Correct Answer” box.  
5. Click “Next Question” to store it in memory.  
6. After adding questions, click “Evaluate Quiz” to see GPT-4’s predicted answers, along with an accuracy score.

Note: Change `model="gpt-4"` in the code to another model (e.g., `"gpt-3.5-turbo"`) if you prefer.
