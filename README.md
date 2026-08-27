## What is this?

CircuitLens investigates how transformer language models implement specific behaviors by analyzing and causally intervening on their internal representations. Really, this is just a way for me to get my feet wet in mechanistic interpretability.

## Research Question

> **How does GPT-2 Small implement induction and token-copying behavior?**

The project aims to identify the neural circuits responsible for this behavior and test their causal importance using mechanistic interpretability techniques.

## Approach

1. **Set up and validate GPT-2 Small**
   - Load the model locally and establish a baseline for its predictions.
   - Understand the model's architecture and internal representations.

2. **Create controlled test cases**
   - Build simple synthetic sequences where the expected behavior is clearly defined.
   - Measure how well GPT-2 performs the target behavior.

3. **Inspect the model's internal activity**
   - Examine how information changes across layers and token positions.
   - Look for attention heads that appear to play a role in the behavior.

4. **Test individual components**
   - Temporarily remove or alter specific attention heads and other components.
   - Measure whether this changes the model's behavior.

5. **Test causal explanations**
   - Transfer internal activity between different inputs to see whether it changes the model's prediction in the expected way.
   - Compare these results against control experiments.

6. **Test the findings**
   - Repeat the experiments with different sequences and conditions.
   - Determine whether the identified components consistently contribute to the behavior.

7. **Document the results**
   - Record experiments, assumptions, failures, and findings.
   - Summarize the final results and conclusions in a research report.

## Project Status

*Very much still in progress...*

## Results

*To be updated.*

## Research Report

*To be updated.*

## On the name

I thought I had come up with a slick name, but it turns out that it was already used [here](https://github.com/egolimblevskaia/CircuitLens). While the names are the same, the projects have different goals. The original CircuitLens focuses on automated interpretability of transcoder features and their underlying circuits, whereas this project focuses on understanding and causally testing specific circuits in GPT-2 Small, starting with induction and token-copying behavior.