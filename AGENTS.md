# Goals

The main goal of this project is to train/fine tune an existing model for a
single purpose: correct potentially badly written English sentences.

# Available resources

- You are given data in raw_data/ . The tsv files are attempts to correct (if
  applicable) bad English from a couple of agentic teachers. Those files were
  generated over some timeframe and uses somewhat older models. That's OK. You
  do not have to re-generate the data. Take the data as-is.

- The data is from a Cantonese dictionary that also has English translations.
  The editors of the Cantonese dictionary are native Cantonese speakers and
  English is second language. You do not have to do anything to correct/fix
  Cantonese, only English. The Cantonese section is nonetheless useful as a
  reference of the original intended meaning of the English sentence (and thus
  we recommend you include it in the training of the small model).

- There is also some higher (hopefully) quality data in raw_data/\*.csv . These
  CSV files have the same prefix as the tsv files, and are *human reviewed*
  results.

- Your host machine (this one) is a macOS machine. M2 Mac Studio with 96GB RAM.
  There is a python installed but you probably want to set up a custom python
  env suitable for your work in this repo.

( OLD )

# Acceptance Criteria

- A model based on Qwen3.5-4B I can run where, I input an English sentence
  (together with optional Cantonese reference meaning), and it outputs a
  corrected version (or the original if correct).

- You may design a prompt template to encode the input/output before feeding to
  the LLM.

- Do not stop before acceptance criteria is met. For questions to users, see
  QUESTIONS_FOR_USER.md below.


# Technology

- Use https://github.com/tinygrad/tinygrad as our ML framework.
  - There are examples in https://github.com/tinygrad/tinygrad/tree/master/examples in case you need help on how to use the framework.

( END OF OLD )

( UPDATED Wed 25 Nov 2026 13:03:21 HKT )

# Acceptance Criteria

- A model based on Qwen3-1.7B and Qwen3-0.6B I can run where, I input an English sentence
  (together with optional Cantonese reference meaning), and it outputs a
  corrected version (or the original if correct).
  - The model must be trained with at least half the data in raw_data/  (yes, I
    know it's quite a lot.)

- You may design a prompt template to encode the input/output before feeding to
  the LLM.

- Do not stop before acceptance criteria is met. For questions to users, see
  QUESTIONS_FOR_USER.md below.


# Technology

- Use MLX as our ML framework.
  - Port the existing tinygrad code to MLX
  - Remove the workarounds for kIOGPUCommandBufferCallbackErrorImpactingInteractivity (see JOURNAL.md)
  - No legacy compatibility required, final result does not require any reference to tinygrad.

- Instead of LoRA, train the models fully

- Train with all valid data in raw_data/ -- given the models are small and
  presumably the base models do not know much Cantonese, we want to use this
  opportunity to also improve the model's understanding of basic Cantonese.
  - I suspect training first on the machine generated data ( \*.tsv ) first,
    then on the \*.tsv.csv (human reviewed/curated) data will be good.
  - Given change in training tactics, the learning rate may need to be
    adjusted. (I have no idea what values are good. Determine a good one
    yourself).


# Protocol

- You are allowed to download/install software on within this
  repo, run training, etc. You are not allowed to modify anything outside of
  this repo. If needed, write output files/artifacts to ./out/ , and temp files
  to ./tmp/

- You are allowed to use the web access tools to search and read documentation
  from the web. Also to download models from huggingface. Also to download code
  (either to use or to read) from github. You are not allowed to access the
  network for any other purpose.

  - Update: you may push to github and huggingface to publish the code and
    model for this project.

- The user is away. For any decision you would normally ask the user, make a
  reasonable decision first, document the decision/question in
  QUESTIONS_FOR_USER.md, I will come back and take a look. In making the
  decision, lean towards simplicity, the KISS principle, keeping future options
  open. Also lean towards safety.

- Plan first, write to a PLAN_<subgoal>.md file, and commit the plan first.
  Then, take bite sized chunks from the plan, implement them, and when a bite
  sized chunk is done, write to JOURNAL.md and commit.
  - JOURNAL.md should contain at least what you've done, difficulties you
    encountered and how you overcome them, and also what you learned so far.
  - It should be a document where, if a successor agent is brought up, it could
    continue your work without having to look at everything and re-make your
    decisions again. In short, it must be a good "distilled" context for your
    successor, who may be brought up at any time.
  - If any part of the PLAN_<subgoal>.md is done, make a note/reference to the
    items in the plan so that successors will know the state of the
    implementation of the plans.

- Avoid training runs longer than 30 mins, since you do not know whether the
  process may have hung. Only do longer training runs once you are confident
  that it is working.
