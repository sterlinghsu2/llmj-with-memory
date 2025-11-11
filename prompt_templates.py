"""
Prompt templates for LLM response generation and judging.
"""
from typing import List, Any


def format_generation_prompt(question: str) -> str:
    """Format the question into a proper prompt for the model.
    
    Args:
        question: The question to answer
        
    Returns:
        Formatted prompt string
    """
    # For math problems, request boxed answer format for easier extraction
    return f"Question: {question}\n\nProvide your final answer in \\boxed{{}} format.\n\nAnswer:"


def format_best_of_n_prompt(
    question: str, 
    responses: List[Any],
    ground_truth: str,
    tokenizer: Any
) -> str:
    """Format prompt for best-of-N judgment using chat template.
    
    Args:
        question: The question being evaluated
        responses: List of responses to judge
        ground_truth: The correct answer
        tokenizer: Tokenizer with chat template (if available)
        
    Returns:
        Formatted prompt string
    """
    # Build responses text efficiently
    responses_text = "\n\n".join(
        f"Response {i+1}: {response.text}" 
        for i, response in enumerate(responses)
    )
    
    user_message = f"""You are an expert judge evaluating mathematical reasoning responses.

Question: {question}

Here are {len(responses)} different responses to evaluate:

{responses_text}

Please evaluate each response and determine which one is the best. Consider:
1. Correctness of the mathematical reasoning
2. Clarity and completeness of explanation
3. Logical soundness of the approach

Respond in this format:
Best Response: [number]
Reasoning: [your detailed explanation]
Confidence: [score from 0.0 to 1.0]"""

    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback to raw prompt if no chat template
        prompt = user_message
    
    return prompt


def format_score_based_prompt(question: str, response: str, ground_truth: str) -> str:
    """Format prompt for score-based judgment using 0-10 scoring.
    
    Args:
        question: The question being evaluated
        response: The response to score
        ground_truth: The correct answer (not used in prompt, kept for API compatibility)
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert judge. Please evaluate the following [response] to [question] and assign a score from 0-10 based on the quality of mathematical reasoning.

[question]: {question}

[response]: {response}

Your evaluation must be in the exact format specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put 'None' if there is no exact, final answer to extract from the response.

reasoning: Explain your scoring decision. Consider the mathematical reasoning quality, clarity of explanation, logical soundness, and completeness of the solution approach.

score: Give a score from 0-10 where:
- 0-2: No clear answer or reasoning, incoherent or nonsensical response
- 3-4: Some mathematical understanding shown but major gaps in reasoning or logic
- 5-6: Reasonable approach with significant errors or incomplete reasoning
- 7-8: Sound mathematical reasoning with only minor issues in explanation or completeness
- 9-10: Excellent mathematical reasoning, clear and complete explanation, well-justified answer

confidence: Your confidence in this score assignment (0-100)."""

    return prompt

