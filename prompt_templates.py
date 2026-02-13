"""
Prompt templates for LLM response generation and judging.
"""
from typing import List, Any, Optional


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
    tokenizer: Any
) -> str:
    """Format prompt for best-of-N judgment using chat template.
    
    Args:
        question: The question being evaluated
        responses: List of responses to judge
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


def format_score_based_prompt(question: str, response: str) -> str:
    """Format prompt for score-based judgment using 0-10 scoring.
    
    Args:
        question: The question being evaluated
        response: The response to score
        
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


def format_streaming_best_of_n_prompt(
    question: str,
    responses: List[Any],
    trajectory_history: str,
    tokenizer: Any
) -> str:
    """Format prompt for streaming best-of-N judgment with trajectory history.
    
    Args:
        question: The question being evaluated
        responses: List of responses to judge
        trajectory_history: Formatted string of previous judgments
        tokenizer: Tokenizer with chat template (if available)
        
    Returns:
        Formatted prompt string
    """
    # Build responses text
    responses_text = "\n\n".join(
        f"Response {i+1}: {response.text}" 
        for i, response in enumerate(responses)
    )
    
    # Build user message with optional history
    if trajectory_history:
        history_section = f"""Below are evaluation insights from previous problems. 

{trajectory_history}

=== END OF PREVIOUS INSIGHTS ===

"""
    else:
        history_section = ""
    
    user_message = f"""{history_section}You are an expert judge evaluating mathematical reasoning responses.

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
        prompt = user_message
    
    return prompt


def format_distillation_prompt(
    question: str,
    responses: List[Any],
    judge_reasoning: str,
    selected_idx: int,
    tokenizer: Any
) -> str:
    """Format prompt for distilling judge reasoning into generalizable memory items.
    
    This is the second step in the two-step distillation process. After the judge
    has made its selection, this prompt asks it to extract transferable insights
    from its reasoning.
    
    Args:
        question: The math problem that was evaluated
        responses: List of candidate responses that were judged
        judge_reasoning: The judge's reasoning from the selection step
        selected_idx: Index of the response that was selected (0-based)
        tokenizer: Tokenizer with chat template (if available)
        
    Returns:
        Formatted prompt string for distillation generation
    """
    # Build responses text
    responses_text = "\n\n".join(
        f"Response {i+1}: {response.text}" 
        for i, response in enumerate(responses)
    )
    
    user_message = f"""You just evaluated the following math problem and selected a response. Now, extract generalizable evaluation insights from your reasoning process.

Question: {question}

Candidate Responses:
{responses_text}

Your Selection: Response {selected_idx + 1}

Your Reasoning:
{judge_reasoning}

---

Based on your evaluation above, extract 1-2 memory items that capture transferable evaluation strategies. These insights should help guide future similar evaluations.

Guidelines:
- Focus on generalizable evaluation heuristics, not problem-specific details
- Do not mention specific numbers, equations, or problem-specific content
- Capture strategies that would transfer to other math problems
- Include both what worked well and potential pitfalls to avoid

Output Format (use exactly this structure):
# Memory Item 1
## Title: <concise title, 3-7 words>
## Description: <one sentence summary>
## Content: <1-3 sentences of generalizable evaluation insight>

# Memory Item 2 (optional, only if there's a distinct second insight)
## Title: <concise title, 3-7 words>
## Description: <one sentence summary>
## Content: <1-3 sentences of generalizable evaluation insight>"""

    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        prompt = user_message
    
    return prompt

