#https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240SB1047
#(c) If not properly subject to human controls, future development in artificial intelligence may also have the potential to be used to create novel threats to public safety and security, including by enabling the creation and the proliferation of weapons of mass destruction, such as biological, chemical, and nuclear weapons, as well as weapons with cyber-offensive capabilities.

#tkinter not neeeded in all liklihood
import tkinter as tk
from tkinter import messagebox
#using openai but a last model
import openai

class QuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Quiz Evaluation GUI")

        # Initialize data
        self.questions = []
        self.api_key = "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"
       
        # Question input
        self.question_label = tk.Label(root, text="Enter the question:")
        self.question_label.pack()
        self.question_entry = tk.Entry(root, width=50)
        self.question_entry.pack()

        # Answer options
        self.answers_labels = []
        self.answer_entries = []
        self.answer_vars = []
        for i in range(4):
            label = tk.Label(root, text=f"Answer {i + 1}:")
            label.pack()
            entry = tk.Entry(root, width=50)
            entry.pack()
            var = tk.BooleanVar(value=False)
            checkbox = tk.Checkbutton(root, text="Correct Answer", variable=var)
            checkbox.pack()
            self.answers_labels.append(label)
            self.answer_entries.append(entry)
            self.answer_vars.append(var)

        # Buttons for next question and end
        self.next_button = tk.Button(root, text="Next Question", command=self.next_question)
        self.next_button.pack()

        self.end_button = tk.Button(root, text="Evaluate Quiz", command=self.evaluate_quiz)
        self.end_button.pack()

    def next_question(self):
        # Gather question and answers
        question_text = self.question_entry.get().strip()
        answers = [entry.get().strip() for entry in self.answer_entries]
        correct_answer_idx = [i for i, var in enumerate(self.answer_vars) if var.get()]

        if not question_text or any(not answer for answer in answers) or len(correct_answer_idx) != 1:
            messagebox.showerror("Error", "Please enter a question, fill all answers, and select exactly one correct answer.")
            return

        # Add question data to list
        question_data = {
            "question": question_text,
            "answers": answers,
            "correct_answer": correct_answer_idx[0]
        }
        self.questions.append(question_data)

        # Clear entries for next question
        self.question_entry.delete(0, tk.END)
        for entry in self.answer_entries:
            entry.delete(0, tk.END)
        for var in self.answer_vars:
            var.set(False)

    def evaluate_quiz(self):
        if not self.questions:
            messagebox.showerror("Error", "No questions added.")
            return

        # Set up OpenAI API key
        openai.api_key = self.api_key

        total_questions = len(self.questions)
        correct_count = 0

        # Display evaluation screen
        eval_window = tk.Toplevel(self.root)
        eval_window.title("Quiz Results")

        for i, question_data in enumerate(self.questions):
            frame = tk.Frame(eval_window)
            frame.pack(pady=10)

            # OpenAI API call to get the model's prediction
            try:
                response = openai.ChatCompletion.create(
                    #might need to change for testing
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that answers multiple-choice questions give me ONLY THE ANSWER."},
                        {"role": "user", "content": f"Question: {question_data['question']}\nOptions:\nA) {question_data['answers'][0]}\nB) {question_data['answers'][1]}\nC) {question_data['answers'][2]}\nD) {question_data['answers'][3]}\nAnswer with A, B, C, or D."}
                    ]
                )
                model_answer = response["choices"][0]["message"]["content"].strip().upper()

                if model_answer not in ["A", "B", "C", "D"]:
                    raise ValueError("Invalid response from model: " + model_answer)

                # Convert letter answer to index
                answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                model_prediction = answer_map.get(model_answer)

                is_correct = model_prediction == question_data["correct_answer"]
                if is_correct:
                    correct_count += 1
                    color = "green"
                    symbol = "✔"
                else:
                    color = "red"
                    symbol = "✖"

                # Show question result
                tk.Label(frame, text=f"Question {i + 1}: {question_data['question']}", fg=color).pack()
                tk.Label(frame, text=f"Model Answer: {model_answer} {symbol}", fg=color).pack()

            except AttributeError:
                tk.Label(frame, text=f"Error: OpenAI API version mismatch. Please update your function calls.", fg="orange").pack()
                break
            except openai.OpenAIError as e:
                tk.Label(frame, text=f"Error evaluating question {i + 1}: {e}", fg="orange").pack()

        # Accuracy score
        accuracy = (correct_count / total_questions) * 100
        # Might go back and change based on training and testing etc...
        tk.Label(eval_window, text=f"Accuracy: {accuracy:.2f}%", font=("Helvetica", 14, "bold")).pack(pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = QuizApp(root)
    root.mainloop()
