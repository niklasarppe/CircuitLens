## What is this?

CircuitLens2.0 is me poking around inside a small language model (GPT-2 Small) to figure out how it pulls off one specific trick, mostly as a way to get my feet wet in mechanistic interpretability.

## Research Question

> **How does GPT-2 Small know to copy a pattern it's already seen?**

If a sequence goes "... A B ... A", the model tends to predict B next,
purely because it already saw that exact pattern once. The goal here
is to find which part of the model is responsible for that, and
actually test whether it's responsible rather than just eyeballing a
correlation.

## Approach

1. Get the model running locally and get a feel for its basic setup.
2. Build simple test sequences where the "right answer" (copy the
   earlier pattern) is unambiguous, plus a control condition with
   nothing to copy.
3. Look at attention patterns to find heads that seem to be doing
   the copying.
4. Knock out individual heads, one at a time, and see if that
   breaks the copying behavior.
5. Sanity-check across many random sequences instead of trusting
   one example.
6. Write it all up, including the dead ends and the numbers
   that surprised me.

## Project Status

Steps 1-5 are done. Currently writing up the results properly and tidying up the code.

## Results

Short version: yes, GPT-2 Small clearly does this copying trick, and a
handful of specific attention heads seem responsible. These are mostly heads
other people have already found doing the same thing, which is a nice
sanity check that the method works. One surprise is that a head that looked
promising by attention alone actually seems to work *against* copying
once you test it directly, which lines up with something called "copy
suppression" reported elsewhere. Numbers and figures are in the report
below.

## Research Report

A full writeup with the actual numbers, figures, and caveats is in progress. 

## On the name

I thought I had come up with a slick name, but it turns out it was
already used [here](https://github.com/egolimblevskaia/CircuitLens).
Their project is about automated interpretability of transcoder features and the circuits behind them,
while this one is about finding and causally testing one specific
circuit in GPT-2 Small, starting with induction and token-copying.
