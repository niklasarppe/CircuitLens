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

1. Get the model running locally.
2. Build simple test sequences where the "right answer" (copy the
   earlier pattern) is unambiguous, plus a control condition with
   nothing to copy.
3. Look at attention patterns to find heads that seem to be doing
   the copying.
4. Knock out individual heads one at a time and see if that
   breaks the copying behavior.
5. Sanity-check across many random sequences.
6. Write it all up in a report.

## How This Actually Works

Here's roughly what the code does:

- **Build test sequences.** Take a random handful of tokens as "the
  first half", then glue a second half onto it. In the *repeated*
  condition, the second half is an exact copy of the first. In the
  *shuffled* condition, it's the same tokens rearranged specifically so nothing
  lines up.
- **Run the model and watch where it looks.** GPT-2 Small has 144
  small sub-parts (attention heads, 12 per layer of the underlying neural network)
  that each decide, at every
  position, how much to "look back" at earlier positions. For every
  head, I check how much it looks back specifically at the token that
  came right after an earlier match.
- **Rank the heads, then test the top ones directly.** Looking in the
  right place doesn't prove a head is actually doing anything as it
  could just be a coincidence. So for the ~20 most promising heads, I
  temporarily disable them one at a time and check whether the model
  gets noticeably worse at completing the pattern. If it does, that's
  real evidence the head matters.
- **Disable heads in two different ways.** I try replacing a head's
  output with plain zero, and separately with that head's "typical"
  output on sequences with nothing to copy. Zeroing it out is the
  obvious thing to try, but it can also wipe out unrelated stuff the
  head normally contributes. Comparing both gives a clearer picture
  of what the head is actually doing.
- **Repeat the process.** Everything above is repeated over hundreds 
  of different random sequences, so the results are an average with some 
  sense of how consistent they are, rather than a fluke from one lucky (or unlucky) example.


## Results

Short version: yes, GPT-2 Small clearly does this copying trick, and a
handful of specific attention heads seem responsible. These are mostly heads
other people have already found doing the same thing, which is a nice
sanity check that the method works. One surprise is that a head that looked
promising by attention alone actually seems to work *against* copying
once you test it directly, which lines up with something called "copy
suppression" reported elsewhere. Numbers and figures are in the report
below.

## On the name

I thought I had come up with a slick name, but it turns out it was
already used [here](https://github.com/egolimblevskaia/CircuitLens).
Their project is about automated interpretability of transcoder features and the circuits behind them, while this one is about finding and causally testing one specific
circuit in GPT-2 Small, starting with induction and token-copying.
