import os
import sys
import asyncio

# Add the parent directory to the path so we can import bot.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot import ask_with_memory

async def run_tests():
    questions_file = os.path.join(os.path.dirname(__file__), "test_questions.txt")
    report_file = os.path.join(os.path.dirname(__file__), "test_report.md")
    
    if not os.path.exists(questions_file):
        print(f"❌ Error: {questions_file} not found.")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("⚠️ No questions found in test_questions.txt")
        return

    print(f"🚀 Starting automated tests for {len(questions)} questions...")
    
    # Use a dummy user_id for testing
    test_user_id = 999999
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Automated Test Report\n\n")
        
        for i, question in enumerate(questions, 1):
            print(f"⏳ Testing Q{i}: {question}")
            try:
                answer = await ask_with_memory(test_user_id, question)
            except Exception as e:
                answer = f"⚠️ ERROR: {str(e)}"
            
            # Write in the exact format requested by the user
            f.write(f'Question . {i}  :"{question}"\n')
            f.write(f'Answer: "{answer}"\n\n')
            
            print(f"✅ Q{i} complete.")
            
    print(f"🎉 Testing complete! Report saved to {report_file}")

if __name__ == "__main__":
    asyncio.run(run_tests())
